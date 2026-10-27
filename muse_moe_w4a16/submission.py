# okc: problem=muse_moe_w4a16
"""Routed NVFP4 experts with in-register decode.

prepare() rearranges the bits of each 32-bit word of E2M1 codes (lossless: same codes,
same bytes, same FP8 block scales) so a word decodes to 4 BF16x2 pairs with a few
shift/mask ops and one BF16x2 multiply by the block scale. kernel_fn then runs:
  _route     per local expert, the (token, slot) pairs routed to it
  _gate_up   gate/up projections + SwiGLU, BF16 intermediate
  _down      down projection, BF16 output per pair
  _finalize  post-expert RMSNorm and routing-weighted sum per token
Accumulation is FP32; projection outputs and the SwiGLU product round to BF16 as in SGLang.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


LOCAL_START = 0
LOCAL_COUNT = 32
NW = 4  # warps per GEMM program; each owns 16 rows (REP row tiles of 16)

# Prepared weights. For row tile rt (16 rows), K step kc (256), chunk ch (64), lane (g, c) and
# v in 0..3, one 32-bit word holds 8 E2M1 values: rows rt*16 + 8*h8 + g, columns
# kc*256 + ch*64 + c*16 + 4*v + 2*k8 + k1, h8/k8/k1 in {0, 1}. That is one mma.m16n8k16
# A fragment (virtual k = v*16 + 8*k8 + 2*c + k1 of the chunk), and lane (g, c) owns whole
# 16-value scale blocks. Words are stored [rt][kc][ch][g][c][v], so each lane reads 16
# contiguous bytes per chunk. Register j = 2*k8 + h8 of the fragment uses bit slot j:
# (sign, e1, e0, m) bit positions in each 16-bit half (k1 = 0 low, 1 high).
_SLOT_BITS = ((15, 8, 7, 6), (12, 5, 4, 3), (9, 2, 1, 0), (14, 13, 11, 10))

# A masked slot is BF16x2 e2m1 * 2^-126 (0.5 is subnormal); times the scale's BF16x2
# (scale * 2^118) it is e2m1 * scale * 2^-8, exact. Outputs: 8 BF16 in fragment order.
_DECODE = gl.constexpr(r"""
{
    .reg .b32 r0, r1, r2, r3, t1, t2;
    and.b32 r0, $8, 0x81C081C0;
    mul.rn.bf16x2 r0, r0, $9;
    shl.b32 t1, $8, 3;
    and.b32 r1, t1, 0x81C081C0;
    mul.rn.bf16x2 r1, r1, $10;
    shl.b32 t1, $8, 6;
    and.b32 r2, t1, 0x81C081C0;
    mul.rn.bf16x2 r2, r2, $9;
    shr.b32 t1, $8, 4;
    and.b32 t1, t1, 0x00C000C0;
    shr.b32 t2, $8, 5;
    and.b32 t2, t2, 0x01000100;
    or.b32 t1, t1, t2;
    shl.b32 t2, $8, 1;
    and.b32 t2, t2, 0x80008000;
    or.b32 r3, t1, t2;
    mul.rn.bf16x2 r3, r3, $10;
    mov.b32 {$0, $1}, r0;
    mov.b32 {$2, $3}, r1;
    mov.b32 {$4, $5}, r2;
    mov.b32 {$6, $7}, r3;
}
""")

# Prepared block scales are one byte each, ((e + 9) << 3) | m for scale 2^e * (1 + m/8)
# (every E4M3 value but zero; zero-scale blocks get zero codes). Byte b -> BF16 bits
# b * 16 + 0x7600 is scale * 2^118. The input holds rows g (low byte) and g + 8; the outputs
# are each replicated to BF16x2.
_SCALES = gl.constexpr(r"""
{
    .reg .b32 t;
    cvt.u32.u16 t, $2;
    prmt.b32 $0, t, 0, 0x4040;
    mad.lo.u32 $0, $0, 16, 0x76007600;
    prmt.b32 $1, t, 0, 0x4141;
    mad.lo.u32 $1, $1, 16, 0x76007600;
}
""")


@gluon.jit
def _idx4(N0: gl.constexpr, N1: gl.constexpr, N2: gl.constexpr, N3: gl.constexpr, L: gl.constexpr):
    a0 = gl.arange(0, N0, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, L))))
    a1 = gl.arange(0, N1, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, L))))
    a2 = gl.arange(0, N2, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, L))))
    a3 = gl.arange(0, N3, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, L))))
    i0 = gl.expand_dims(gl.expand_dims(gl.expand_dims(a0, 1), 2), 3)
    i1 = gl.expand_dims(gl.expand_dims(gl.expand_dims(a1, 0), 2), 3)
    i2 = gl.expand_dims(gl.expand_dims(gl.expand_dims(a2, 0), 1), 3)
    i3 = gl.expand_dims(gl.expand_dims(gl.expand_dims(a3, 0), 1), 2)
    return i0, i1, i2, i3


@gluon.jit
def _idx3(N0: gl.constexpr, N1: gl.constexpr, N2: gl.constexpr, L: gl.constexpr):
    a0 = gl.arange(0, N0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    a1 = gl.arange(0, N1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    a2 = gl.arange(0, N2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    i0 = gl.expand_dims(gl.expand_dims(a0, 1), 2)
    i1 = gl.expand_dims(gl.expand_dims(a1, 0), 2)
    i2 = gl.expand_dims(gl.expand_dims(a2, 0), 1)
    return i0, i1, i2


@gluon.jit
def _weights(words, scales, BN: gl.constexpr, A_LAYOUT: gl.constexpr):
    """words int32 [BN/2, 4, 4, 4] ((rt, g), ch, c, v), scales int16 [BN/2, 4, 4] -> the
    A operand, BF16 [BN, 256] (rows rt*16 + 8*h8 + g, virtual k ch*64 + v*16 + 8*k8 + 2*c + k1)."""
    sa, sb = gl.inline_asm_elementwise(_SCALES, "=r,=r,h", [scales], dtype=(gl.int32, gl.int32),
                                       is_pure=True, pack=1)
    sa = gl.broadcast(gl.expand_dims(sa, 3), words)[0]
    sb = gl.broadcast(gl.expand_dims(sb, 3), words)[0]
    o0, o1, o2, o3, o4, o5, o6, o7 = gl.inline_asm_elementwise(
        _DECODE, "=h,=h,=h,=h,=h,=h,=h,=h,r,r,r", [words, sa, sb], dtype=(gl.bfloat16,) * 8,
        is_pure=True, pack=1)
    # (h8, k8) = (0, 0), (1, 0), (0, 1), (1, 1); each pair is (k1 = 0, k1 = 1).
    k8_0 = gl.join(gl.join(o0, o1), gl.join(o2, o3))   # [..., k1, h8]
    k8_1 = gl.join(gl.join(o4, o5), gl.join(o6, o7))
    t = gl.join(k8_0, k8_1)                            # [i, ch, c, v, k1, h8, k8]
    t = gl.reshape(t, [BN // 16, 8, 4, 4, 4, 2, 2, 2])  # [rt, g, ch, c, v, k1, h8, k8]
    t = gl.permute(t, [0, 6, 1, 2, 4, 7, 3, 5])         # [rt, h8, g, ch, v, k8, c, k1]
    return gl.convert_layout(gl.reshape(t, [BN, 256]), A_LAYOUT, assert_trivial=True)


@gluon.jit
def _activations(xs, BM: gl.constexpr, B_LAYOUT: gl.constexpr):
    """xs BF16 [BM, 4, 4, 16] (n, ch, c, (v, k8, k1)) -> the B operand, BF16 [256, BM]."""
    t = gl.reshape(xs, [BM, 4, 4, 4, 2, 2])           # [n, ch, c, v, k8, k1]
    t = gl.permute(t, [1, 3, 4, 2, 5, 0])             # [ch, v, k8, c, k1, n]
    return gl.convert_layout(gl.reshape(t, [256, BM]), B_LAYOUT, assert_trivial=True)


@gluon.jit
def _expert_gemm(X, W, S, GS, TILES, LIST, COUNT, OUT, M, LOCAL_START: gl.constexpr,
                 NW: gl.constexpr, GATED: gl.constexpr, BM: gl.constexpr, REP: gl.constexpr,
                 LW: gl.constexpr, LX: gl.constexpr, MMA: gl.constexpr):
    """One (n tile, expert m-tile) of a projection. GATED: out[pair, n] = bf16(silu(gate) * up)
    from w13 (gate rows n, up rows 4096 + n), x rows = tokens. Else: out[pair, n] from w2,
    x rows = pairs. Both round the projection to BF16 first."""
    BN: gl.constexpr = 16 * NW * REP
    A_LAYOUT: gl.constexpr = gl.DotOperandLayout(0, MMA, 2)
    B_LAYOUT: gl.constexpr = gl.DotOperandLayout(1, MMA, 2)
    LS: gl.constexpr = gl.SliceLayout(3, LW)
    nt = gl.program_id(0)
    tile = gl.program_id(1)
    if tile < gl.load(TILES):
        info = gl.load(TILES + 1 + tile)
        e = info >> 16
        m0 = info & 0xFFFF
        rows_left = gl.load(COUNT + e) - m0
        ROWS: gl.constexpr = 8192 if GATED else 4096
        rt0 = nt * (BN // 16)

        # Words: [i = (rt, g), ch, c, v]; word offset rt*8192 + kc*512 + ch*128 + g*16 + c*4 + v.
        i, ch, c, v = _idx4(BN // 2, 4, 4, 4, LW)
        w_off = (rt0 + i // 8) * 8192 + (i % 8) * 16 + ch * 128 + c * 4 + v
        w_ptr = W + e.to(gl.int64) * (ROWS * 512) + w_off
        # Scales (int16 = rows g, g + 8): offset rt*2048 + kc*128 + g*16 + c*4 + ch.
        si, sch, sc = _idx3(BN // 2, 4, 4, LS)
        s_off = (rt0 + si // 8) * 2048 + (si % 8) * 16 + sc * 4 + sch
        s_ptr = S + e.to(gl.int64) * (ROWS * 128) + s_off

        # Activations: [n, ch, c, j], column ch*64 + c*16 + j.
        n, xch, xc, xj = _idx4(BM, 4, 4, 16, LX)
        valid = n < rows_left
        pair = gl.load(LIST + e * M + m0 + n, mask=valid, other=0)
        if GATED:
            x_row = pair // 8
        else:
            x_row = pair
        x_ptr = X + x_row * 4096 + xch * 64 + xc * 16 + xj
        x_mask = gl.broadcast(valid, x_ptr)[0]

        acc = gl.zeros([BN, BM], gl.float32, MMA)
        if GATED:
            acc_u = gl.zeros([BN, BM], gl.float32, MMA)
            wu_ptr = w_ptr + 256 * 8192
            su_ptr = s_ptr + 256 * 2048
        words = gl.load(w_ptr)
        scales = gl.load(s_ptr)
        xs = gl.load(x_ptr, mask=x_mask, other=0.)
        if GATED:
            words_u = gl.load(wu_ptr)
            scales_u = gl.load(su_ptr)
        for kc in range(1, 16):
            # Prefetch step kc while step kc - 1 computes.
            nwords = gl.load(w_ptr + kc * 512)
            nscales = gl.load(s_ptr + kc * 128)
            nxs = gl.load(x_ptr + kc * 256, mask=x_mask, other=0.)
            if GATED:
                nwords_u = gl.load(wu_ptr + kc * 512)
                nscales_u = gl.load(su_ptr + kc * 128)
            b = _activations(xs, BM, B_LAYOUT)
            acc = mma_v2(_weights(words, scales, BN, A_LAYOUT), b, acc)
            if GATED:
                acc_u = mma_v2(_weights(words_u, scales_u, BN, A_LAYOUT), b, acc_u)
                words_u = nwords_u
                scales_u = nscales_u
            words = nwords
            scales = nscales
            xs = nxs
        b = _activations(xs, BM, B_LAYOUT)
        acc = mma_v2(_weights(words, scales, BN, A_LAYOUT), b, acc)
        if GATED:
            acc_u = mma_v2(_weights(words_u, scales_u, BN, A_LAYOUT), b, acc_u)

        # The weights were decoded as value * 2^-8.
        gs = gl.load(GS + LOCAL_START + e) * 256.0
        y = (acc * gs).to(gl.bfloat16)
        if GATED:
            g = y.to(gl.float32)
            u = (acc_u * gs).to(gl.bfloat16).to(gl.float32)
            y = (g / (1.0 + gl.exp(-g)) * u).to(gl.bfloat16)
        r = gl.arange(0, BN, layout=gl.SliceLayout(1, MMA))
        on = gl.arange(0, BM, layout=gl.SliceLayout(0, MMA))
        ovalid = on < rows_left
        opair = gl.load(LIST + e * M + m0 + on, mask=ovalid, other=0)
        o_ptr = OUT + gl.expand_dims(opair, 0) * 4096 + nt * BN + gl.expand_dims(r, 1)
        gl.store(o_ptr, y, mask=gl.broadcast(gl.expand_dims(ovalid, 0), o_ptr)[0])


def _layouts(bm, rep):
    """(LW, LX, MMA) for BM tokens and REP row tiles per warp."""
    bn = 16 * NW * rep
    # Words [BN/2, 4, 4, 4]: v and ch in registers, lane = (g, c), warps over row tiles.
    lw = gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 0, 1], [0, 0, 0, 2], [0, 1, 0, 0], [0, 2, 0, 0]]
        + [[8 * NW * (1 << b), 0, 0, 0] for b in range(rep.bit_length() - 1)],
        lane_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]],
        warp_bases=[[8 * (1 << b), 0, 0, 0] for b in range(NW.bit_length() - 1)],
        block_bases=[], shape=[bn // 2, 4, 4, 4])
    # Activations [BM, 4, 4, 16]: 16 contiguous columns and ch in registers, lane = (n % 8, c),
    # all warps hold all of it.
    lx = gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 1, 0, 0], [0, 2, 0, 0]]
        + [[8 * (1 << b), 0, 0, 0] for b in range((bm // 8).bit_length() - 1)],
        lane_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]],
        warp_bases=[[0, 0, 0, 0]] * (NW.bit_length() - 1),
        block_bases=[], shape=[bm, 4, 4, 16])
    mma = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[NW, 1], instr_shape=[16, 8])
    return lw, lx, mma


@triton.jit
def _route(IDS, COUNT, LIST, TILES, M, LOCAL_START: tl.constexpr, P: tl.constexpr,
           BM: tl.constexpr, MT: tl.constexpr):
    """Per local expert: its (token * 8 + slot) pairs in order and their count; then the
    (expert << 16 | first pair) of each BM-pair tile, after their number."""
    offs = tl.arange(0, P)
    ids = tl.load(IDS + offs, mask=offs < M * 8, other=-1) - LOCAL_START
    e = tl.arange(0, 32)
    hit = ids[None, :] == e[:, None]
    pos = tl.cumsum(hit.to(tl.int32), 1) - 1
    tl.store(LIST + e[:, None] * M + pos, offs[None, :], mask=hit)
    cnt = tl.sum(hit.to(tl.int32), 1)
    tl.store(COUNT + e, cnt)
    nt = (cnt + BM - 1) // BM
    start = tl.cumsum(nt, 0) - nt
    j = tl.arange(0, MT)
    tl.store(TILES + 1 + start[:, None] + j[None, :], e[:, None] * 65536 + j[None, :] * BM,
             mask=j[None, :] < nt[:, None])
    tl.store(TILES, tl.sum(nt, 0))


@triton.jit
def _finalize(Y, ROUTE_W, IDS, NORM_W, OUT, LOCAL_START: tl.constexpr, LOCAL_COUNT: tl.constexpr):
    """out[t] = sum over local slots of weight * rmsnorm(bf16(y * (1 + w)))."""
    t = tl.program_id(0)
    s = tl.arange(0, 8)
    d = tl.arange(0, 4096)
    expert = tl.load(IDS + t * 8 + s)
    local = (expert >= LOCAL_START) & (expert < LOCAL_START + LOCAL_COUNT)
    # Slots routed elsewhere were never written: mask them before any arithmetic.
    y = tl.load(Y + (t * 8 + s[:, None]) * 4096 + d[None, :], mask=local[:, None], other=0.).to(tl.float32)
    w = tl.load(NORM_W + d)
    scaled = (y * w[None, :]).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(scaled * scaled, 1) / 4096 + 1e-8)
    route = tl.where(local, tl.load(ROUTE_W + t * 8 + s), 0.)
    tl.store(OUT + t * 4096 + d, tl.sum(scaled * (inv * route)[:, None], 0).to(tl.bfloat16))


def _codes(packed):
    """uint8 [R, K/2] -> E2M1 codes uint8 [R, K]."""
    return torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1)


def _scale_codes(sf):
    """float8_e4m3fn [R, K/16] -> (prepared scale codes uint8, nonzero mask). Lossless for
    every positive E4M3 value: code ((e + 9) << 3) | m for 2^e * (1 + m/8), e >= -9."""
    f = sf.float()
    if (f < 0).any() or not torch.isfinite(f).all():
        raise ValueError("block scales must be finite and non-negative")
    nz = f > 0
    e = torch.floor(torch.log2(torch.where(nz, f, torch.ones_like(f))))
    m = torch.round((torch.where(nz, f, torch.ones_like(f)) / torch.exp2(e) - 1) * 8)
    assert torch.equal(torch.exp2(e) * (1 + m / 8) * nz, f)
    code = (((e + 9) * 8 + m) * nz).to(torch.uint8)
    return code, nz


def _prepare_matrix(packed, sf):
    """One expert's matrix: (words uint8 [R*K/2], scales uint8 [R*K/16]) in kernel order."""
    rows, k = packed.shape[0], packed.shape[1] * 2
    codes = _codes(packed)
    scode, nz = _scale_codes(sf)
    # A zero-scale block is all zeros: give it zero codes so its (unused) scale code is moot.
    codes = (codes.view(rows, k // 16, 16) * nz[..., None]).view(rows, k)
    # [rt, h8, g, kc, ch, c, v, k8, k1] -> [rt, kc, ch, g, c, v, k8, h8, k1]
    codes = codes.view(rows // 16, 2, 8, k // 256, 4, 4, 4, 2, 2).permute(0, 3, 4, 2, 5, 6, 7, 1, 8)
    codes = codes.reshape(-1, 8).to(torch.int64)
    word = torch.zeros(codes.shape[0], dtype=torch.int64, device=packed.device)
    for p in range(8):
        slot, high = divmod(p, 2)
        c = codes[:, p]
        for bit, pos in zip((3, 2, 1, 0), _SLOT_BITS[slot]):
            word |= ((c >> bit) & 1) << (16 * high + pos)
    words = word.view(torch.uint8).view(-1, 8)[:, :4].reshape(-1)  # little-endian low 4 bytes
    # Scales [rt, h8, g, kc, ch, c] -> [rt, kc, g, c, ch, h8].
    scales = scode.view(rows // 16, 2, 8, k // 256, 4, 4).permute(0, 3, 2, 5, 4, 1).reshape(-1)
    return words, scales


def prepare(weights: Weights):
    out = {}
    for name in ("w13", "w2"):
        words, scales = zip(*(_prepare_matrix(w, s) for w, s in zip(
            weights[name][LOCAL_START:LOCAL_START + LOCAL_COUNT],
            weights[name + "_sf"][LOCAL_START:LOCAL_START + LOCAL_COUNT])))
        out[name] = torch.stack(words).view(torch.int32)
        out[name + "_s"] = torch.stack(scales).view(torch.int16)
        out[name + "_gs"] = weights[name + "_gs"].float().contiguous()
    out["norm_w"] = weights["post_norm_w"].float() + 1.0
    torch.cuda.synchronize()
    return out


def _config(m):
    """(BM, REP gate/up, REP down) for m tokens."""
    if m <= 8:
        return 8, 1, 1
    return 16, 1, 1


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    assert cfg["local_start"] == LOCAL_START and cfg["local_count"] == LOCAL_COUNT
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    m = x.shape[0]
    dev = x.device
    bm, rep1, rep2 = _config(m)
    max_tiles = min(LOCAL_COUNT, m * 8) + triton.cdiv(m * 8, bm)
    count = torch.empty((LOCAL_COUNT,), dtype=torch.int32, device=dev)
    pairs = torch.empty((LOCAL_COUNT, m), dtype=torch.int32, device=dev)
    tiles = torch.empty((1 + max_tiles,), dtype=torch.int32, device=dev)
    _route[(1,)](ids, count, pairs, tiles, m, LOCAL_START, P=triton.next_power_of_2(m * 8), BM=bm,
                 MT=triton.next_power_of_2(triton.cdiv(m, bm)), num_warps=4)
    inter = torch.empty((m * 8, 4096), dtype=torch.bfloat16, device=dev)
    _expert_gemm[(4096 // (16 * NW * rep1), max_tiles)](
        x, state["w13"], state["w13_s"], state["w13_gs"], tiles, pairs, count, inter, m, LOCAL_START, NW,
        True, bm, rep1, *_layouts(bm, rep1), num_warps=NW)
    y = torch.empty((m * 8, 4096), dtype=torch.bfloat16, device=dev)
    _expert_gemm[(4096 // (16 * NW * rep2), max_tiles)](
        inter, state["w2"], state["w2_s"], state["w2_gs"], tiles, pairs, count, y, m, LOCAL_START, NW,
        False, bm, rep2, *_layouts(bm, rep2), num_warps=NW)
    out = torch.empty_like(x)
    _finalize[(m,)](y, data["topk_weights"].contiguous(), ids, state["norm_w"], out,
                    LOCAL_START, LOCAL_COUNT, num_warps=8)
    return out
