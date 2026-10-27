"""Projection GPU kernels; tune launch settings in projection/configs.py."""

import triton
import triton.language as tl


@triton.jit
def _tma_weight_major(WD, XD, Y, P, M: tl.constexpr, N: tl.constexpr,
                      K: tl.constexpr, S: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tl.static_assert(K % (S * BK) == 0)
    mt: tl.constexpr = tl.cdiv(M, BM)
    tile = tl.program_id(0)
    m0 = (tile % mt) * BM
    n0 = (tile // mt) * BN
    split = tl.program_id(1)
    k0 = split * (K // S)
    acc = tl.full((BN, BM), 0, tl.float32)
    for ki in range(K // (S * BK)):
        # Descriptors use zero padding for rows beyond irregular M/N.
        w = WD.load([n0, k0 + ki * BK])
        x = XD.load([m0, k0 + ki * BK])
        acc = tl.dot(w, x.T, acc)
    if S > 1:
        tl.extra.cuda.gdc_launch_dependents()
    ns = n0 + tl.arange(0, BN)
    ms = m0 + tl.arange(0, BM)
    offsets = ms[None, :] * N + ns[:, None]
    mask = (ms[None, :] < M) & (ns[:, None] < N)
    if S == 1:
        tl.store(Y + offsets, acc, mask)
    else:
        tl.store(P + split * M * N + offsets, acc, mask)
