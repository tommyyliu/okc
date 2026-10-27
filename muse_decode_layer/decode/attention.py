"""Fused QK normalization/cache stores and dense split-context attention."""

import torch
import triton
import triton.language as tl

from triton.language.extra.cuda import libdevice
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from .config import ATTENTION_BACKEND
from .projections import _project

@triton.jit
def _qkv_norm_store(QKV, Q, KC, VC, POS, LOC, KS, VS, FREQ,
                    FACTOR: tl.constexpr, THETA: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 64)
    # Eighteen heads: sixteen query heads and two key/value heads.
    x = tl.load(QKV + t * 2304 + h * 64 + d).to(tl.float32)
    x *= tl.rsqrt(tl.sum(x * x, 0) / 64 + 1e-5)
    if THETA != 0:
        partner = tl.load(QKV + t * 2304 + h * 64 + (d ^ 1)).to(tl.float32)
        # The partner uses the same per-head reduction.
        raw = tl.load(QKV + t * 2304 + h * 64 + d).to(tl.float32)
        partner *= tl.rsqrt(tl.sum(raw * raw, 0) / 64 + 1e-5)
        pos = tl.load(POS + t).to(tl.float32)
        angle = pos * tl.load(FREQ + d // 2)
        x = x * libdevice.cos(angle) + tl.where(d % 2 == 0, -partner, partner) * libdevice.sin(angle)
    if h < 16:
        tl.store(Q + t * 1024 + h * 64 + d, x * FACTOR)
    else:
        slot = tl.load(LOC + t)
        scale_k = tl.load(KS)
        scale_v = tl.load(VS)
        v = tl.load(QKV + t * 2304 + 1152 + (h - 16) * 64 + d).to(tl.float32)
        tl.store(KC + slot * 128 + (h - 16) * 64 + d, x / scale_k)
        tl.store(VC + slot * 128 + (h - 16) * 64 + d, v / scale_v)

@triton.jit
def _attention_parts(Q, KC, VC, TABLE, SEQ, KS, VS, PART, MAX, SUM,
                     LENGTH: tl.constexpr, PAGE: tl.constexpr, TABLE_STRIDE: tl.constexpr,
                     WINDOW: tl.constexpr, SCALE: tl.constexpr, SPLITS: tl.constexpr,
                     QM: tl.constexpr, BN: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    split = tl.program_id(2)
    r = tl.arange(0, QM)
    d = tl.arange(0, 64)
    nk = tl.arange(0, BN)
    seq = tl.load(SEQ + b)
    first = tl.maximum(0, seq - WINDOW + 1) if WINDOW else 0
    end = seq + LENGTH
    # Dynamic lengths bound the loop; unused page-table entries are never read.
    chunk = tl.cdiv(end - first, SPLITS * BN) * BN
    start = first + split * chunk
    stop = tl.minimum(end, start + chunk)
    query_pos = seq + r // 8
    q = tl.load(Q + (b * LENGTH + r[:, None] // 8) * 1024 + h * 512
                + (r[:, None] % 8) * 64 + d[None, :], r[:, None] < LENGTH * 8, other=0)
    key_scale = tl.load(KS)
    value_scale = tl.load(VS)
    m = tl.full((QM,), -float("inf"), tl.float32)
    l = tl.full((QM,), 0, tl.float32)
    acc = tl.full((QM, 64), 0, tl.float32)
    for off in range(start, stop, BN):
        pos = off + nk
        page = tl.load(TABLE + b * TABLE_STRIDE + pos // PAGE, pos < stop, other=0)
        slot = page * PAGE + pos % PAGE
        k = tl.load(KC + slot[None, :] * 128 + h * 64 + d[:, None],
                    pos[None, :] < stop, other=0.0).to(tl.bfloat16)
        logits = tl.dot(q, k).to(tl.float32) * (SCALE * key_scale)
        visible = (pos[None, :] < stop) & (pos[None, :] <= query_pos[:, None])
        if WINDOW:
            visible &= pos[None, :] > query_pos[:, None] - WINDOW
        logits = tl.where(visible, logits, -float("inf"))
        new_m = tl.maximum(m, tl.max(logits, 1))
        # Padded rows and fully masked pieces have a finite subtraction origin.
        safe_m = tl.where(new_m == -float("inf"), 0., new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(logits - safe_m[:, None])
        v = tl.load(VC + slot[:, None] * 128 + h * 64 + d[None, :],
                    pos[:, None] < stop, other=0.0).to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        l = l * alpha + tl.sum(p, 1)
        m = new_m
    base = ((b * 2 + h) * SPLITS + split) * QM + r
    tl.store(MAX + base, m)
    tl.store(SUM + base, l)
    tl.store(PART + base[:, None] * 64 + d[None, :], acc * value_scale)

@triton.jit
def _attention_finish(PART, MAX, SUM, QKV, OUT,
                      LENGTH: tl.constexpr, SPLITS: tl.constexpr, QM: tl.constexpr,
                      BS: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1)
    b = t // LENGTH
    r = (t % LENGTH) * 8 + head % 8
    s = tl.arange(0, BS)
    d = tl.arange(0, 64)
    base = ((b * 2 + head // 8) * SPLITS + s) * QM + r
    m = tl.load(MAX + base, s < SPLITS, other=-float("inf"))
    l = tl.load(SUM + base, s < SPLITS, other=0)
    p = tl.load(PART + base[:, None] * 64 + d[None, :], s[:, None] < SPLITS, other=0)
    alpha = tl.exp(m - tl.max(m, 0))
    denom = tl.sum(l * alpha, 0)
    x = tl.sum(p * alpha[:, None], 0) / denom
    # BF16 is the normal attention-output boundary, before the per-head norm.
    x = x.to(tl.bfloat16).to(tl.float32)
    x *= tl.rsqrt(tl.sum(x * x, 0) / 64 + 1e-5)
    gate = tl.load(QKV + t * 2304 + 1280 + head * 64 + d).to(tl.float32)
    tl.store(OUT + t * 1024 + head * 64 + d, x * tl.sigmoid(gate))

def _split_attention(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    qkv = _project(data["attn_input"], state, "qkv_gate")
    q = torch.empty((tokens, 16, 64), dtype=torch.bfloat16, device=qkv.device)
    _qkv_norm_store[(tokens, 18)](
        qkv, q, data["k_cache"], data["v_cache"], data["positions"], data["cache_loc"],
        data["k_scale"], data["v_scale"], state["rope_freq"],
        cfg["qk_scale_factor"], cfg["rope_theta"], num_warps=4)
    max_span = data["page_table"].shape[1] * cfg["page_size"]
    if cfg["window"]:
        max_span = min(max_span, cfg["window"] + length - 1)
    splits = min(triton.next_power_of_2(triton.cdiv(max_span, 256)),
                 triton.next_power_of_2(max(1, 128 // batch)))
    qm = max(16, triton.next_power_of_2(length * 8))
    shape = (batch, 2, splits, qm)
    partial = torch.empty((*shape, 64), dtype=torch.float32, device=qkv.device)
    maxima = torch.empty(shape, dtype=torch.float32, device=qkv.device)
    sums = torch.empty_like(maxima)
    _attention_parts[(batch, 2, splits)](
        q, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"], partial, maxima, sums,
        length, cfg["page_size"], data["page_table"].stride(0), cfg["window"],
        cfg["sm_scale"], splits, qm, 128, num_warps=4, num_stages=2)
    out = torch.empty((tokens, 1024), dtype=torch.bfloat16, device=qkv.device)
    _attention_finish[(tokens, 16)](
        partial, maxima, sums, qkv, out, length, splits, qm,
        triton.next_power_of_2(splits), num_warps=4)
    return out


@triton.jit
def _attention_gate(ATTN, QKV, OUT):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 64)
    # Preserve the same BF16 output boundary as the split-attention control.
    x = tl.load(ATTN + t * 1024 + h * 64 + d).to(tl.bfloat16).to(tl.float32)
    x *= tl.rsqrt(tl.sum(x * x, 0) / 64 + 1e-5)
    gate = tl.load(QKV + t * 2304 + 1280 + h * 64 + d).to(tl.float32)
    tl.store(OUT + t * 1024 + h * 64 + d, x * tl.sigmoid(gate))


def _fa3_attention(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    qkv = _project(data["attn_input"], state, "qkv_gate")
    # Round the normalized query to E4M3. Its scale factor belongs in the
    # softmax scale, exactly as on the SGLang FA3 path, not inside the FP8 cast.
    q = torch.empty((tokens, 16, 64), dtype=torch.float8_e4m3fn, device=qkv.device)
    _qkv_norm_store[(tokens, 18)](
        qkv, q, data["k_cache"], data["v_cache"], data["positions"], data["cache_loc"],
        data["k_scale"], data["v_scale"], state["rope_freq"],
        1.0, cfg["rope_theta"], num_warps=4)
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, 2),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, 2))
    attn = flash_attn_with_kvcache(
        q=q,
        k_cache=data["k_cache"],
        v_cache=data["v_cache"],
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(attn, tuple):
        attn = attn[0]
    out = torch.empty((tokens, 1024), dtype=torch.bfloat16, device=qkv.device)
    _attention_gate[(tokens, 16)](attn, qkv, out, num_warps=4)
    return out


def _attention(data, state):
    if ATTENTION_BACKEND == "full_fa3" and data["config"]["window"] == 0:
        return _fa3_attention(data, state)
    return _split_attention(data, state)

