import triton
import triton.language as tl

@triton.jit
def _gemv(X, W, Y, P, N: tl.constexpr, K: tl.constexpr, S: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """One token: coalesced weight reads and FP32 reduction over K."""
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    ks = split * (K // S) + tl.arange(0, BK)
    end = (split + 1) * (K // S)
    acc = tl.full((BN,), 0, tl.float32)
    for _ in range(tl.cdiv(K // S, BK)):
        x = tl.load(X + ks, ks < end, other=0).to(tl.float32)
        w = tl.load(W + ns[:, None] * K + ks[None, :], (ns[:, None] < N) & (ks[None, :] < end), other=0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        ks += BK
    if S == 1:
        tl.store(Y + ns, acc, ns < N)
    else:
        tl.store(P + split * N + ns, acc, ns < N)

@triton.jit
def _gemm(X, WH, WL, Y, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, S: tl.constexpr, ROUTER: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """BF16 tensor cores; independent FP32 high/low router accumulators."""
    ms = tl.program_id(0) * BM + tl.arange(0, BM)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    ks = split * (K // S) + tl.arange(0, BK)
    high = tl.full((BM, BN), 0, tl.float32)
    low = tl.full((BM, BN), 0, tl.float32)
    for _ in range(K // (S * BK)):
        x = tl.load(X + ms[:, None] * K + ks[None, :], ms[:, None] < M, other=0)
        offsets = ns[None, :] * K + ks[:, None]
        wh = tl.load(WH + offsets, ns[None, :] < N, other=0)
        high = tl.dot(x, wh, high)
        if ROUTER:
            wl = tl.load(WL + offsets, ns[None, :] < N, other=0)
            low = tl.dot(x, wl, low)
        ks += BK
    if ROUTER:
        high += low
    offsets = ms[:, None] * N + ns[None, :]
    mask = (ms[:, None] < M) & (ns[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, high, mask)
    else:
        tl.store(P + split * M * N + offsets, high, mask)

@triton.jit
def _reduce(P, Y, TOTAL: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    splits = tl.arange(0, S)
    values = tl.load(P + splits[:, None] * TOTAL + offsets[None, :], offsets[None, :] < TOTAL, other=0)
    tl.store(Y + offsets, tl.sum(values, axis=0), offsets < TOTAL)

@triton.jit
def _gemv_full(X, W, Y, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, BLOCK_K: tl.constexpr):
    ns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    ks = tl.arange(0, BLOCK_K)
    x = tl.load(X + ks, ks < K, other=0).to(tl.float32)
    w = tl.load(W + ns[:, None] * K + ks[None, :], (ns[:, None] < N) & (ks[None, :] < K), other=0).to(tl.float32)
    sums = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + ns, sums, ns < N)

@triton.jit
def _stream_gemm(X, W, Y, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    mt = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    ms = tile % mt * BM + tl.arange(0, BM)
    ns = tile // mt * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    chunk: tl.constexpr = tl.cdiv(K, S)
    ks = split * chunk + tl.arange(0, BK)
    end = tl.minimum((split + 1) * chunk, K)
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(tl.cdiv(chunk, BK)):
        x = tl.load(X + ms[:, None] * K + ks[None, :], (ms[:, None] < M) & (ks[None, :] < end), other=0)
        w = tl.load(W + ns[None, :] * K + ks[:, None], (ns[None, :] < N) & (ks[:, None] < end), other=0)
        acc = tl.dot(x, w, acc)
        ks += BK
    offsets = ms[:, None] * N + ns[None, :]
    mask = (ms[:, None] < M) & (ns[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, acc, mask)
    else:
        tl.store(P + split * M * N + offsets, acc, mask)

@triton.jit
def _exponent_ranges(W, Bases, Sizes):
    group = tl.program_id(0)
    lanes = tl.arange(0, 256)
    bits = tl.load(W + group * 256 + lanes).to(tl.uint16, bitcast=True).to(tl.uint32)
    exponent = bits >> 7 & 255
    lo = tl.min(exponent, axis=0)
    hi = tl.max(exponent, axis=0)
    compact = hi - lo <= 15
    tl.store(Bases + group, tl.where(compact, lo, 256))
    tl.store(Sizes + group, tl.where(compact, 384, 512))

@triton.jit
def _pack_lossless(W, Payload, Offsets, Bases):
    group = tl.program_id(0)
    lanes = tl.arange(0, 256)
    bits = tl.load(W + group * 256 + lanes).to(tl.uint16, bitcast=True).to(tl.uint32)
    base = tl.load(Bases + group).to(tl.uint32)
    start = tl.load(Offsets + group)
    sign_mantissa = bits & 127 | bits >> 8 & 128
    tl.store(Payload + start + lanes, sign_mantissa.to(tl.uint8))
    if base < 256:
        pairs = lanes * 2
        even_bits = tl.load(W + group * 256 + pairs, lanes < 128, other=0)
        odd_bits = tl.load(W + group * 256 + pairs + 1, lanes < 128, other=0)
        even = even_bits.to(tl.uint16, bitcast=True).to(tl.uint32) >> 7 & 255
        odd = odd_bits.to(tl.uint16, bitcast=True).to(tl.uint32) >> 7 & 255
        code = even - base | odd - base << 4
        tl.store(Payload + start + 256 + lanes, code.to(tl.uint8), lanes < 128)
    else:
        exponent = bits >> 7 & 255
        tl.store(Payload + start + 256 + lanes, exponent.to(tl.uint8))

@triton.jit
def _decode_lossless(Payload, Offsets, Bases, groups, lanes, valid):
    start = tl.load(Offsets + groups, valid, other=0)
    base = tl.load(Bases + groups, valid, other=256).to(tl.uint32)
    compact = base < 256
    sm = tl.load(Payload + start + lanes, valid, other=0).to(tl.uint32)
    index = tl.where(compact, lanes // 2, lanes)
    code = tl.load(Payload + start + 256 + index, valid, other=0).to(tl.uint32)
    delta = code >> (lanes & 1) * 4 & 15
    exponent = tl.where(compact, base + delta, code)
    bits = sm & 127 | (sm & 128) << 8 | exponent << 7
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)

@triton.jit
def _lossless_gemv(X, Payload, Offsets, Bases, Y, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, GROUPS_K: tl.constexpr):
    ns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    kg = tl.arange(0, GROUPS_K)
    lanes = tl.arange(0, 256)
    groups = ns[:, None, None] * (K // 256) + kg[None, :, None]
    valid = (ns[:, None, None] < N) & (kg[None, :, None] < K // 256)
    w = _decode_lossless(Payload, Offsets, Bases, groups, lanes[None, None, :], valid)
    w = tl.reshape(w, (ROWS, GROUPS_K * 256)).to(tl.float32)
    ks = tl.arange(0, GROUPS_K * 256)
    x = tl.load(X + ks, ks < K, other=0).to(tl.float32)
    sums = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + ns, sums, ns < N)

@triton.jit
def _lossless_gemm(X, Payload, Offsets, Bases, Y, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, S: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    mt = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    ms = tile % mt * BM + tl.arange(0, BM)
    ns = tile // mt * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    k0 = split * (K // S)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(K // (S * BK)):
        ks = k0 + kk
        x = tl.load(X + ms[:, None] * K + ks[None, :], ms[:, None] < M, other=0)
        groups = ns[:, None] * (K // 256) + k0 // 256
        lanes = k0 % 256 + kk[None, :]
        w = _decode_lossless(Payload, Offsets, Bases, groups, lanes, ns[:, None] < N)
        acc = tl.dot(x, tl.trans(w), acc)
        k0 += BK
    offsets = ms[:, None] * N + ns[None, :]
    mask = (ms[:, None] < M) & (ns[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, acc, mask)
    else:
        tl.store(P + split * M * N + offsets, acc, mask)

@triton.jit
def _decode_aligned(Payload, Offsets, Bases, groups, lanes, valid):
    start = tl.multiple_of(tl.load(Offsets + groups, valid, other=0), (128, 1))
    base = tl.load(Bases + groups, valid, other=256).to(tl.uint32)
    compact = base < 256
    sm = tl.load(Payload + start + lanes, valid, other=0).to(tl.uint32)
    index = tl.where(compact, lanes // 2, lanes)
    code = tl.load(Payload + start + 256 + index, valid, other=0).to(tl.uint32)
    delta = code >> (lanes & 1) * 4 & 15
    exponent = tl.where(compact, base + delta, code)
    bits = sm & 127 | (sm & 128) << 8 | exponent << 7
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)

@triton.jit
def _decode_matrix(Payload, Offsets, Bases, Y, GROUPS: tl.constexpr, G: tl.constexpr, ALIGNED: tl.constexpr):
    groups = tl.program_id(0) * G + tl.arange(0, G)
    lanes = tl.arange(0, 256)
    if ALIGNED:
        w = _decode_aligned(Payload, Offsets, Bases, groups[:, None], lanes[None, :], groups[:, None] < GROUPS)
    else:
        w = _decode_lossless(Payload, Offsets, Bases, groups[:, None], lanes[None, :], groups[:, None] < GROUPS)
    tl.store(Y + groups[:, None] * 256 + lanes[None, :], w, groups[:, None] < GROUPS)

@triton.jit
def _decode_group(Payload, Offsets, Bases, Y):
    """One format branch per block, with each packed exponent byte read once."""
    group = tl.program_id(0)
    start = tl.multiple_of(tl.load(Offsets + group), 128)
    base = tl.load(Bases + group).to(tl.uint32)
    lanes = tl.arange(0, 256)
    sm = tl.load(Payload + start + lanes).to(tl.uint32)
    if base < 256:
        pairs = tl.arange(0, 128)
        packed = tl.load(Payload + start + 256 + pairs).to(tl.uint32)
        shifts = tl.arange(0, 2) * 4
        exponent = base + tl.reshape(packed[:, None] >> shifts[None, :] & 15, (256,))
    else:
        exponent = tl.load(Payload + start + 256 + lanes).to(tl.uint32)
    bits = sm & 127 | (sm & 128) << 8 | exponent << 7
    w = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(Y + group * 256 + lanes, w)

@triton.jit
def _copy_matrix(W, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(W + offsets, offsets < TOTAL, other=0)
    tl.store(Y + offsets, w, offsets < TOTAL)
