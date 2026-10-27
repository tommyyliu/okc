"""Projection GPU kernels; tune launch settings in projection/configs.py."""

import triton
import triton.language as tl


@triton.jit
def _gemv_full(X, W, Y, N: tl.constexpr, K: tl.constexpr,
               ROWS: tl.constexpr, BLOCK_K: tl.constexpr):
    ns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    ks = tl.arange(0, BLOCK_K)
    x = tl.load(X + ks, ks < K, other=0).to(tl.float32)
    w = tl.load(W + ns[:, None] * K + ks[None, :],
                (ns[:, None] < N) & (ks[None, :] < K), other=0).to(tl.float32)
    # Neither inputs nor weights are rounded before multiplication.
    sums = tl.sum(w * x[None, :], axis=1)
    tl.store(Y + ns, sums, ns < N)


@triton.jit
def _gemv_chunked(X, W, Y, N: tl.constexpr, K: tl.constexpr,
                  ROWS: tl.constexpr, BLOCK_K: tl.constexpr):
    # Keep a short FP32 vector live across K chunks, then reduce it once.
    # Router weights are loaded directly as FP32; BF16 operands are only widened.
    ns = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lane_k = tl.arange(0, BLOCK_K)
    acc = tl.full((ROWS, BLOCK_K), 0, tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        ks = start * BLOCK_K + lane_k
        x = tl.load(X + ks, ks < K, other=0).to(tl.float32)
        w = tl.load(W + ns[:, None] * K + ks[None, :],
                    (ns[:, None] < N) & (ks[None, :] < K), other=0).to(tl.float32)
        acc += w * x[None, :]
    tl.store(Y + ns, tl.sum(acc, axis=1), ns < N)


@triton.jit
def _router_direct(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BLOCK_K: tl.constexpr):
    # One expert and BM tokens per CTA. W is the original FP32 router tensor.
    expert = tl.program_id(0)
    ms = tl.program_id(1) * BM + tl.arange(0, BM)
    ks = tl.arange(0, BLOCK_K)
    woffsets = expert * K + ks
    w = tl.load(W + woffsets, (expert < N) & (ks < K), other=0).to(tl.float32)
    xoffsets = ms[:, None] * K + ks[None, :]
    x = tl.load(X + xoffsets, (ms[:, None] < M) & (ks[None, :] < K), other=0).to(tl.float32)
    products = x * w[None, :]
    sums = tl.sum(products, axis=1)
    offsets = ms * N + expert
    tl.store(Y + offsets, sums, (ms < M) & (expert < N))
