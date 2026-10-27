"""Experimental storage layouts; each kernel retains BF16/FP32 arithmetic."""

import triton
import triton.language as tl


@triton.jit
def _weight_major_tiled_gemm(X, W, Y, P, M: tl.constexpr, N: tl.constexpr,
                             K: tl.constexpr, S: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # W is [N/BN, K/BK, BN, BK], with each weight tile contiguous.
    tl.static_assert(N % BN == 0)
    tl.static_assert(K % (S * BK) == 0)
    mt = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    n_tile = tile // mt
    ms = (tile % mt) * BM + tl.arange(0, BM)
    ni = tl.arange(0, BN)
    ki = tl.arange(0, BK)
    ns = n_tile * BN + ni
    split = tl.program_id(1)
    ks = split * (K // S) + ki
    woffsets = n_tile * BN * K + split * (K // S) * BN + ni[:, None] * BK + ki[None, :]
    acc = tl.full((BN, BM), 0, tl.float32)
    for _ in range(K // (S * BK)):
        w = tl.load(W + woffsets, ns[:, None] < N, other=0)
        x = tl.load(X + ms[None, :] * K + ks[:, None],
                    ms[None, :] < M, other=0)
        acc = tl.dot(w, x, acc)
        woffsets += BN * BK
        ks += BK
    if S > 1:
        tl.extra.cuda.gdc_launch_dependents()
    offsets = ms[None, :] * N + ns[:, None]
    mask = (ms[None, :] < M) & (ns[:, None] < N)
    if S == 1:
        tl.store(Y + offsets, acc, mask)
    else:
        tl.store(P + split * M * N + offsets, acc, mask)


@triton.jit
def _weight_major_native_gemm(W, X, Y, M: tl.constexpr, N: tl.constexpr,
                              K: tl.constexpr, BM: tl.constexpr,
                              BN: tl.constexpr, BK: tl.constexpr):
    # Same dot and loads as _weight_major_gemm; Y is physically [N,M].
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
    tl.store(Y + ns[:, None] * M + ms[None, :], acc,
             (ms[None, :] < M) & (ns[:, None] < N))


@triton.jit
def _router_interleaved_gemm(X, W, Y, P, M: tl.constexpr, N: tl.constexpr,
                             K: tl.constexpr, S: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Logical expert tile is BN. Physical features interleave hi0,lo0,hi1,lo1.
    tl.static_assert(K % (S * BK) == 0)
    ms = tl.program_id(0) * BM + tl.arange(0, BM)
    pairs = tl.program_id(1) * (2 * BN) + tl.arange(0, 2 * BN)
    split = tl.program_id(2)
    ks = split * (K // S) + tl.arange(0, BK)
    acc = tl.full((BM, 2 * BN), 0, tl.float32)
    for _ in range(K // (S * BK)):
        x = tl.load(X + ms[:, None] * K + ks[None, :],
                    ms[:, None] < M, other=0)
        w = tl.load(W + pairs[None, :] * K + ks[:, None],
                    pairs[None, :] < 2 * N, other=0)
        acc = tl.dot(x, w, acc)
        ks += BK
    # Add only after independent FP32 K accumulation of both components.
    combined = tl.sum(tl.reshape(acc, (BM, BN, 2)), axis=2)
    if S > 1:
        tl.extra.cuda.gdc_launch_dependents()
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = ms[:, None] * N + ns[None, :]
    mask = (ms[:, None] < M) & (ns[None, :] < N)
    if S == 1:
        tl.store(Y + offsets, combined, mask)
    else:
        tl.store(P + split * M * N + offsets, combined, mask)
