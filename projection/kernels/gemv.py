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
