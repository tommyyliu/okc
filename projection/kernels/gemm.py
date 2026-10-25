"""Projection GPU kernels; tune launch settings in projection/configs.py."""

import triton
import triton.language as tl


@triton.jit
def _gemm(X, WH, WL, Y, P, M: tl.constexpr, N: tl.constexpr,
          K: tl.constexpr, S: tl.constexpr, ROUTER: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """BF16 tensor cores; independent FP32 high/low router accumulators."""
    ms = tl.program_id(0) * BM + tl.arange(0, BM)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    ks = split * (K // S) + tl.arange(0, BK)
    high = tl.full((BM, BN), 0, tl.float32)
    low = tl.full((BM, BN), 0, tl.float32)
    for _ in range(K // (S * BK)):
        x = tl.load(X + ms[:, None] * K + ks[None, :],
                    ms[:, None] < M, other=0)
        offsets = ns[None, :] * K + ks[:, None]
        wh = tl.load(WH + offsets, ns[None, :] < N, other=0)
        high = tl.dot(x, wh, high)
        if ROUTER:
            wl = tl.load(WL + offsets, ns[None, :] < N, other=0)
            low = tl.dot(x, wl, low)
        ks += BK
    if ROUTER:
        high += low
    if S > 1:
        tl.extra.cuda.gdc_launch_dependents()
    offsets = ms[:, None] * N + ns[None, :]
    mask = (ms[:, None] < M) & (ns[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, high, mask)
    else:
        tl.store(P + split * M * N + offsets, high, mask)



@triton.jit
def _stream_gemm(X, W, Y, P, M: tl.constexpr, N: tl.constexpr,
                 K: tl.constexpr, S: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Adjacent programs consume the same weight tile for different token
    # tiles. This also works for irregular M and the sliced unpooler.
    mt = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    ms = (tile % mt) * BM + tl.arange(0, BM)
    ns = (tile // mt) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    chunk: tl.constexpr = tl.cdiv(K, S)
    ks = split * chunk + tl.arange(0, BK)
    end = tl.minimum((split + 1) * chunk, K)
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(tl.cdiv(chunk, BK)):
        x = tl.load(X + ms[:, None] * K + ks[None, :],
                    (ms[:, None] < M) & (ks[None, :] < end), other=0)
        w = tl.load(W + ns[None, :] * K + ks[:, None],
                    (ns[None, :] < N) & (ks[:, None] < end), other=0)
        acc = tl.dot(x, w, acc)
        ks += BK
    if S > 1:
        tl.extra.cuda.gdc_launch_dependents()
    offsets = ms[:, None] * N + ns[None, :]
    mask = (ms[:, None] < M) & (ns[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, acc, mask)
    else:
        tl.store(P + split * M * N + offsets, acc, mask)



@triton.jit
def _weight_major_gemm(W, X, Y, M: tl.constexpr, N: tl.constexpr,
                       K: tl.constexpr, BM: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr):
    # A=[output features, K], B=[K, tokens], C=[output features, tokens].
    # Input weights stay row-major. Only output addressing transposes C.
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ms = tl.program_id(1) * BM + tl.arange(0, BM)
    ks = tl.arange(0, BK)
    acc = tl.full((BN, BM), 0, tl.float32)
    for _ in range(tl.cdiv(K, BK)):
        w = tl.load(W + ns[:, None] * K + ks[None, :],
                    (ns[:, None] < N) & (ks[None, :] < K), other=0)
        x = tl.load(X + ms[None, :] * K + ks[:, None],
                    (ms[None, :] < M) & (ks[:, None] < K), other=0)
        acc = tl.dot(w, x, acc)
        ks += BK
    tl.store(Y + ms[None, :] * N + ns[:, None], acc,
             (ms[None, :] < M) & (ns[:, None] < N))
