# okc: problem=muse_attention_decode
"""Gated attention decode in one fused Triton kernel.

Each program handles one (pack of KV heads, context split, request): it normalizes (and on
sliding layers rotates) its queries straight from the fused projection output and attends its
fixed slice of the paged FP8 cache with an online softmax. One extra program per (request,
pack) builds the block's new keys and values, stores them in the cache and attends them
(causal inside the block). Partials go to a workspace and are merged in a two-level tree: the
last program to arrive in a group of splits merges the group, and the last group merger
merges the groups and applies the per-head RMSNorm and sigmoid gate. Arrival counters decide
who merges, and the merger resets them, so the kernel is a single launch that replays
cleanly in a CUDA graph."""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


LOG2E = 1.4426950408889634
RMS_EPS = 1e-5


@triton.jit
def _rms_rope(X, off, mask, FREQ, cd, pos, THETA: tl.constexpr):
    """Scaleless RMSNorm over each row's 64 unmasked entries at X + off ([R, C] offsets of
    head-dim index cd [C]), then interleaved RoPE at positions pos [R]."""
    x = tl.load(X + off, mask=mask, other=0.).to(tl.float32)
    if THETA:
        xp = tl.load(X + off + ((cd ^ 1) - cd)[None, :], mask=mask, other=0.).to(tl.float32)
        inv_freq = tl.load(FREQ + cd)
    inv = tl.rsqrt(tl.sum(x * x, 1) / 64 + 1e-5)
    x = x * inv[:, None]
    if THETA:
        xp = xp * inv[:, None]
        ang = pos.to(tl.float32)[:, None] * inv_freq[None, :]
        sn, cs = _sincos(ang)
        x = x * cs + tl.where((cd % 2 == 0)[None, :], -xp, xp) * sn
    return x


@triton.jit
def _sincos(ang):
    """sin and cos of fp32 angles up to ~1e6: two-step reduction by 2*pi (6.28125 is exact in
    a few bits, so k * 6.28125 is exact), then the hardware approximations on [-pi, pi]."""
    k = libdevice.rint(ang * 0.15915494309189535)
    r = libdevice.fma(-k, 6.28125, ang)
    r = libdevice.fma(-k, 0.0019353071795864769, r)
    sn = tl.inline_asm_elementwise("sin.approx.f32 $0, $1;", "=f,f", [r], dtype=tl.float32, is_pure=True, pack=1)
    cs = tl.inline_asm_elementwise("cos.approx.f32 $0, $1;", "=f,f", [r], dtype=tl.float32, is_pure=True, pack=1)
    return sn, cs


@triton.jit
def _stamp(TS, pid, k: tl.constexpr, dep):
    t = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [dep.to(tl.int32)],
                                  dtype=tl.int64, is_pure=False, pack=1)
    tl.store(TS + pid * 8 + k, t)


@triton.jit
def _prefetch_l2(ptrs):
    tl.inline_asm_elementwise("prefetch.global.L2 [$1];", "=r,l", [ptrs], dtype=tl.int32,
                              is_pure=False, pack=1)


@triton.jit
def _softmax_step(qk, vis, m, l, acc, v):
    qk = tl.where(vis, qk, -float("inf"))
    m_new = tl.maximum(m, tl.max(qk, 1))
    safe = tl.where(m_new == -float("inf"), 0., m_new)
    alpha = tl.exp2(m - safe)
    p = tl.exp2(qk - safe[:, None])
    l = l * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
    return m_new, l, acc


@triton.jit
def _tile(q, pg, start, k_hi, seq, rl, m, l, acc, TAB, KC, VC, kn, c, h0,
          RS: tl.constexpr, PAGE: tl.constexpr, BN: tl.constexpr, WINDOW: tl.constexpr, L: tl.constexpr,
          MASKED: tl.constexpr, PREF: tl.constexpr, QK8: tl.constexpr):
    """Attend one block of BN cached keys from `start`; MASKED handles the slice's ragged end
    and, on sliding verify blocks, the window's start. With PREF, pg holds this block's page-table
    entries on entry and the next block's on return."""
    pos = start + kn
    if MASKED:
        pm = pos < k_hi
        if not PREF:
            pg = tl.load(TAB + pos // PAGE, mask=pm, other=0)
        row = pg.to(tl.int64) * PAGE + pos % PAGE
        kp = row[:, None] * RS + h0 * 64 + c[None, :]
        k = tl.load(KC + kp, mask=pm[:, None], other=0.)
        v = tl.load(VC + kp, mask=pm[:, None], other=0.)
    else:
        if not PREF:
            pg = tl.load(TAB + pos // PAGE)
        row = pg.to(tl.int64) * PAGE + pos % PAGE
        kp = row[:, None] * RS + h0 * 64 + c[None, :]
        k = tl.load(KC + kp)
        v = tl.load(VC + kp)
    if PREF:
        pg = tl.load(TAB + (pos + BN) // PAGE, mask=(pos + BN) < k_hi, other=0)
    if QK8:
        qk = tl.dot(q, tl.trans(k))
    else:
        qk = tl.dot(q, tl.trans(k.to(tl.float16)))
    if MASKED:
        vis = pm[None, :]
        if WINDOW and L > 1:
            vis = vis & (pos[None, :] > (seq + rl)[:, None] - WINDOW)
        m, l, acc = _softmax_step(qk, vis, m, l, acc, v.to(tl.float16))
    else:
        m_new = tl.maximum(m, tl.max(qk, 1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
        m = m_new
    return m, l, acc, pg


@triton.jit
def _merge(WS_ACC, WS_ML, first, count, ROWS: tl.constexpr, SB: tl.constexpr, MST: tl.constexpr):
    """Online merge of `count` partials (m, l [ROWS], acc [ROWS, 64]) from index `first`."""
    rr = tl.arange(0, ROWS)
    d = tl.arange(0, 64)
    mx = tl.full((ROWS,), -float("inf"), tl.float32)
    ls = tl.zeros((ROWS,), tl.float32)
    o = tl.zeros((ROWS, 64), tl.float32)
    for c0 in tl.range(0, count, SB, num_stages=MST):
        sc = c0 + tl.arange(0, SB)
        v = sc < count
        idx = first + sc
        ms = tl.load(WS_ML + idx[:, None] * (2 * ROWS) + rr[None, :], mask=v[:, None],
                     other=-float("inf"), cache_modifier=".cg")
        lp = tl.load(WS_ML + idx[:, None] * (2 * ROWS) + ROWS + rr[None, :], mask=v[:, None],
                     other=0., cache_modifier=".cg")
        a = tl.load(WS_ACC + (idx[:, None, None] * ROWS + rr[None, :, None]) * 64 + d[None, None, :],
                    mask=v[:, None, None], other=0., cache_modifier=".cg")
        m_new = tl.maximum(mx, tl.max(ms, 0))
        safe = tl.where(m_new == -float("inf"), 0., m_new)
        al = tl.exp2(mx - safe)
        w = tl.exp2(ms - safe[None, :])
        ls = ls * al + tl.sum(w * lp, 0)
        o = o * al[:, None] + tl.sum(w[:, :, None] * a, 0)
        mx = m_new
    return mx, ls, o


@triton.jit
def _put(WS_ACC, WS_ML, idx, m, l, acc, r, c, mask, ROWS: tl.constexpr):
    tl.store(WS_ML + idx * (2 * ROWS) + r, m, mask=r < ROWS)
    tl.store(WS_ML + idx * (2 * ROWS) + ROWS + r, l, mask=r < ROWS)
    tl.store(WS_ACC + (idx * ROWS + r)[:, None] * 64 + (c % 64)[None, :], acc, mask=mask)


@triton.jit
def _fused_attn(QKV, FREQ, LOC, KC, VC, KS, VS, TABLE, SEQ, WS_ACC, WS_ML, CNT1, CNT2, OUT, TS,
                TABLE_STRIDE, OFF2,
                L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, P: tl.constexpr,
                PAGE: tl.constexpr, WINDOW: tl.constexpr, THETA: tl.constexpr, QSCALE: tl.constexpr,
                SPAN: tl.constexpr, NSPLIT: tl.constexpr, NTOT: tl.constexpr, GS: tl.constexpr,
                NG: tl.constexpr, QM: tl.constexpr, NKR: tl.constexpr, BN: tl.constexpr,
                SB: tl.constexpr, QK8: tl.constexpr, PREF: tl.constexpr, GEARLY: tl.constexpr,
                WS: tl.constexpr, MST: tl.constexpr, PROF: tl.constexpr):
    """One program per (pack of P KV heads, split, request). Rows are (token, head in pack,
    query head in group); columns are the pack's P heads' 64 dims side by side, with each
    row's query block-diagonal so one dot serves all P heads. Splits 0..NSPLIT-1 attend fixed
    slices of the cached context; split NTOT-1 (the extra one, or the only one) attends the
    block's new tokens."""
    hp = tl.program_id(0)
    s = tl.program_id(1)
    b = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    W: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    ROWS: tl.constexpr = L * P * G
    RS: tl.constexpr = HKV * 64  # cache row stride (elements)
    PC: tl.constexpr = P * 64
    CH: tl.constexpr = (SPAN + NSPLIT * BN - 1) // (NSPLIT * BN) * BN  # keys per split
    h0 = hp * P
    pid = (b * NTOT + s) * (HKV // P) + hp
    if PROF:
        _stamp(TS, pid, 0, pid)

    r = tl.arange(0, QM)
    c = tl.arange(0, PC)
    cp = c // 64
    cd = c % 64
    rl = r // (P * G)
    rp = (r // G) % P
    rg = r % G
    rvalid = r < ROWS
    blk = (cp[None, :] == rp[:, None])
    kn = tl.arange(0, BN)

    seq = tl.load(SEQ + b)
    ks = tl.load(KS)
    vs = tl.load(VS)
    # This split's slice of the cached context (empty for the extra split).
    if WINDOW:
        k_lo = seq - (WINDOW - 1) + s * CH
        k_hi = tl.minimum(seq, k_lo + CH)
        k_lo = tl.maximum(k_lo, 0)
    else:
        k_lo = s * CH
        k_hi = tl.minimum(seq, k_lo + CH)
    if PREF:
        # Page-table entries run one block ahead of the keys they address.
        pg = tl.load(TABLE + b * TABLE_STRIDE + (k_lo + kn) // PAGE, mask=(k_lo + kn) < k_hi, other=0)
    if GEARLY:
        gate = _gate(QKV, b, h0, tl.arange(0, ROWS), L, HQ, HKV, P)
    else:
        gate = 0.
    if s == NTOT - 1:
        rr = tl.arange(0, ROWS)
        _prefetch_l2(QKV + (b * L + rr // (P * G)) * W + (HQ + 2 * HKV) * 64
                     + ((h0 + (rr // G) % P) * G + rr % G) * 64)

    qoff = (b * L + rl)[:, None] * W + ((h0 + rp) * G + rg)[:, None] * 64 + cd[None, :]
    q = _rms_rope(QKV, qoff, rvalid[:, None] & blk, FREQ, cd, seq + rl, THETA)
    if QK8:
        q = (q * (QSCALE * ks)).to(tl.float8e4nv)
    else:
        q = (q * (QSCALE * ks)).to(tl.float16)
    if PROF:
        _stamp(TS, pid, 1, tl.sum(tl.sum(q.to(tl.float32), 1), 0) + seq)

    m = tl.full((QM,), -float("inf"), tl.float32)
    l = tl.zeros((QM,), tl.float32)
    acc = tl.zeros((QM, PC), tl.float32)
    if not PREF:
        pg = kn
    k_first = k_lo
    if WINDOW and L > 1:
        if s == 0:
            # The window's first keys are out of reach of the block's later tokens.
            m, l, acc, pg = _tile(q, pg, k_lo, k_hi, seq, rl, m, l, acc, TABLE + b * TABLE_STRIDE, KC, VC,
                                  kn, c, h0, RS, PAGE, BN, WINDOW, L, True, PREF, QK8)
            k_first = k_lo + BN
    k_full = k_first + tl.maximum(k_hi - k_first, 0) // BN * BN
    for start in tl.range(k_first, k_full, BN, warp_specialize=WS):
        m, l, acc, pg = _tile(q, pg, start, k_hi, seq, rl, m, l, acc, TABLE + b * TABLE_STRIDE, KC, VC,
                              kn, c, h0, RS, PAGE, BN, WINDOW, L, False, PREF, QK8)
    if k_full < k_hi:
        m, l, acc, pg = _tile(q, pg, k_full, k_hi, seq, rl, m, l, acc, TABLE + b * TABLE_STRIDE, KC, VC,
                              kn, c, h0, RS, PAGE, BN, WINDOW, L, True, PREF, QK8)
    if PROF:
        _stamp(TS, pid, 2, tl.sum(l, 0))

    if s == NTOT - 1:
        # The block's own keys and values, rows (token, head in pack), block-diagonal:
        # built here, stored, and attended causally.
        jr = tl.arange(0, NKR)
        jl = jr // P
        jp = jr % P
        jv = jl < L
        jblk = jv[:, None] & (cp[None, :] == jp[:, None])
        tok = (b * L + jl)[:, None] * W
        vx = tl.load(QKV + tok + (HQ + HKV + h0) * 64 + c[None, :], mask=jblk, other=0.).to(tl.float32)
        loc = tl.load(LOC + b * L + jl, mask=jv, other=0)
        kx = _rms_rope(QKV, tok + (HQ + h0) * 64 + c[None, :], jblk, FREQ, cd, seq + jl, THETA)
        k8 = (kx / ks).to(tl.float8e4nv)
        v8 = (vx / vs).to(tl.float8e4nv)
        cpo = loc[:, None] * RS + h0 * 64 + c[None, :]
        tl.store(KC + cpo, k8, mask=jblk)
        tl.store(VC + cpo, v8, mask=jblk)
        if QK8:
            qk = tl.dot(q, tl.trans(k8))
        else:
            qk = tl.dot(q, tl.trans(k8.to(tl.float16)))
        vis = jv[None, :] & (jl[None, :] <= rl[:, None]) & (jp[None, :] == rp[:, None])
        m, l, acc = _softmax_step(qk, vis, m, l, acc, v8.to(tl.float16))
    if PROF:
        _stamp(TS, pid, 3, tl.sum(l, 0))

    bh = b * (HKV // P) + hp
    if NTOT == 1:
        o = acc / l[:, None]
        if P > 1:  # collapse each row's own block onto 64 columns
            o = tl.sum(tl.reshape(tl.where(blk, o, 0.), (QM, P, 64)), 1)
        _epilogue(o, vs, QKV, OUT, b, h0, r, 0., False, L, HQ, HKV, P)
    else:
        _put(WS_ACC, WS_ML, bh * NTOT + s, m, l, acc, r, c, rvalid[:, None] & blk, ROWS)
        tl.debug_barrier()
        g = s // GS
        if NG == 1:
            old = tl.atomic_add(CNT2 + bh, 1, sem="acq_rel")
            if PROF:
                _stamp(TS, pid, 4, old)
            if old == NTOT - 1:
                tl.store(CNT2 + bh, 0)
                mx, ls, o = _merge(WS_ACC, WS_ML, bh * NTOT, NTOT, ROWS, SB, MST)
                if PROF:
                    _stamp(TS, pid, 5, tl.sum(ls, 0))
                _epilogue(o / ls[:, None], vs, QKV, OUT, b, h0, tl.arange(0, ROWS), gate, GEARLY, L, HQ, HKV, P)
                if PROF:
                    _stamp(TS, pid, 6, pid)
        else:
            count = tl.minimum(GS, NTOT - g * GS)
            old = tl.atomic_add(CNT1 + bh * NG + g, 1, sem="acq_rel")
            if PROF:
                _stamp(TS, pid, 4, old)
            if old == count - 1:
                tl.store(CNT1 + bh * NG + g, 0)
                mx, ls, o = _merge(WS_ACC, WS_ML, bh * NTOT + g * GS, count, ROWS, SB, MST)
                rr = tl.arange(0, ROWS)
                _put(WS_ACC, WS_ML, OFF2 + bh * NG + g, mx, ls, o, rr, tl.arange(0, 64),
                     tl.full((ROWS, 64), 1, tl.int1), ROWS)
                tl.debug_barrier()
                old2 = tl.atomic_add(CNT2 + bh, 1, sem="acq_rel")
                if PROF:
                    _stamp(TS, pid, 5, old2)
                if old2 == NG - 1:
                    tl.store(CNT2 + bh, 0)
                    mx, ls, o = _merge(WS_ACC, WS_ML, OFF2 + bh * NG, NG, ROWS, SB, MST)
                    _epilogue(o / ls[:, None], vs, QKV, OUT, b, h0, rr, gate, GEARLY, L, HQ, HKV, P)
                    if PROF:
                        _stamp(TS, pid, 6, pid)


@triton.jit
def _gate(QKV, b, h0, r, L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, P: tl.constexpr):
    G: tl.constexpr = HQ // HKV
    W: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    d = tl.arange(0, 64)
    head = (h0 + (r // G) % P) * G + r % G
    tok = b * L + r // (P * G)
    return tl.load(QKV + tok[:, None] * W + (HQ + 2 * HKV) * 64 + head[:, None] * 64 + d[None, :],
                   mask=(r < L * P * G)[:, None], other=0.).to(tl.float32)


@triton.jit
def _epilogue(o, vs, QKV, OUT, b, h0, r, gate, HAVE_GATE: tl.constexpr,
              L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, P: tl.constexpr):
    """Per-head RMSNorm of the BF16 attention output, times sigmoid(gate); rows r are
    (token, head in pack, query head in group)."""
    G: tl.constexpr = HQ // HKV
    W: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    d = tl.arange(0, 64)
    rl = r // (P * G)
    head = (h0 + (r // G) % P) * G + r % G
    rvalid = r < L * P * G
    tok = b * L + rl
    if not HAVE_GATE:
        gate = _gate(QKV, b, h0, r, L, HQ, HKV, P)
    o = (o * vs).to(tl.bfloat16).to(tl.float32)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / 64 + 1e-5)[:, None]
    o = o * tl.sigmoid(gate)
    tl.store(OUT + tok[:, None] * (HQ * 64) + head[:, None] * 64 + d[None, :],
             o.to(tl.bfloat16), mask=rvalid[:, None])


_COUNTERS = {}
_FREQ = {}
TUNE = {}  # optional overrides (development)
LAST = None


def _counters(n, device):
    buf = _COUNTERS.get(device)
    if buf is None or buf.numel() < n:
        buf = torch.zeros(max(n, 1 << 16), dtype=torch.int32, device=device)
        _COUNTERS[device] = buf
    return buf


def _inv_freq(theta, device):
    """RoPE inverse frequencies per head dim (pairs share one), as the reference builds them."""
    key = (theta, device)
    if key not in _FREQ:
        inv = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
        _FREQ[key] = inv.repeat_interleave(2).contiguous()
    return _FREQ[key]


def _plan(batch, length, hkv, span, window):
    """Launch shape per case, tuned on H200 under CUDA-graph replay with a cold L2."""
    if "nsplit" in TUNE:
        return dict(TUNE)
    if length == 1 and window:  # sliding decode: latency-bound
        if batch < 8:
            t = dict(pack=1, bn=128, warps=8, stages=2, pref=True, nsplit=16)
        elif batch <= 16:
            t = dict(pack=1, bn=128, warps=4, stages=3, nsplit=8)
        else:
            t = dict(pack=1, bn=64, warps=4, stages=4, nsplit=8)
        t["gearly"] = True
    elif length == 1:  # full decode: bandwidth-bound, two KV heads per program (128-byte rows)
        if batch < 8:
            t = dict(pack=2, bn=64, warps=4, stages=6, gs=16, sb=16)
        else:
            t = dict(pack=2, bn=128, warps=4, stages=5)
        t["nsplit"] = max(1, 256 // batch)
        t["gearly"] = True
        t["qk8"] = True
    elif window:  # sliding verify
        if batch < 8:
            t = dict(pack=1, bn=64, warps=8, stages=3, nsplit=32)
        else:
            t = dict(pack=1, bn=64, warps=4, stages=2, pref=True, nsplit=max(1, 128 // batch), sb=1)
        t["qk8"] = True
    else:  # full verify: 64 query rows per KV head; capped registers keep 4 programs per SM
        t = dict(pack=1, bn=64, warps=4, stages=2, pref=True, qk8=True, sb=4 if batch < 8 else 2,
                 nsplit=max(1, min(128, 256 // batch)), maxnreg=None if batch < 8 else 120)
    t["nsplit"] = max(1, min(t["nsplit"], triton.cdiv(span, t["bn"])))
    return t


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, page, window = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"], cfg["window"]
    qkv = data["qkv"]
    device = qkv.device
    table = data["page_table"]
    span = window - 1 if window else table.shape[1] * page
    t = _plan(batch, length, hkv, span, window)
    pack = min(t.get("pack", 1), hkv)
    nsplit, bn, warps, stages = t["nsplit"], t.get("bn", 64), t.get("warps", 4), t.get("stages", 3)
    rows = length * pack * (hq // hkv)
    qm = max(16, triton.next_power_of_2(rows))
    packs = hkv // pack
    ntot = nsplit + 1 if nsplit > 1 else 1
    sb = t.get("sb", max(1, (64 * 32 * warps) // (rows * 64)))
    gs = t.get("gs", sb)
    ng = triton.cdiv(ntot, gs)
    if ng == 1:
        gs = ntot
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=device)
    slots = batch * packs * (ntot + ng)
    ws_acc = torch.empty((slots * rows * 64,), dtype=torch.float32, device=device)
    ws_ml = torch.empty((slots * 2 * rows,), dtype=torch.float32, device=device)
    cnt = _counters(batch * packs * (ng + 1), device)
    freq = _inv_freq(cfg["rope_theta"], device) if cfg["rope_theta"] else ws_ml
    global LAST
    LAST = _fused_attn[(packs, ntot, batch)](
        qkv, freq, data["cache_loc"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        table, data["seq_lens"], ws_acc, ws_ml, cnt, cnt[batch * packs * ng:], out, t.get("ts", out),
        table.stride(0), batch * packs * ntot,
        length, hq, hkv, pack, page, window, bool(cfg["rope_theta"]),
        cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2E,
        span, nsplit, ntot, gs, ng, qm, max(16, triton.next_power_of_2(length * pack)), bn,
        sb, t.get("qk8", False), t.get("pref", False), t.get("gearly", False) and ntot > 1, t.get("ws", False), t.get("mst", 1), "ts" in t,
        num_warps=warps, num_stages=stages, **({"maxnreg": t["maxnreg"]} if t.get("maxnreg") else {}))
    return out
