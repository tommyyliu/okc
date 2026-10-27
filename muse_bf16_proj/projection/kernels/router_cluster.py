"""Experimental single-launch router with cluster-local split-K reduction."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def _router_cluster(X, WH, WL, Y, M: gl.constexpr, N: gl.constexpr,
                    K: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
                    BK: gl.constexpr, C: gl.constexpr):
    gl.static_assert(C == gl.num_ctas())
    gl.static_assert((C == 2) | (C == 4) | (C == 8))
    gl.static_assert(K % (C * BK) == 0)
    gl.static_assert(BN == 16)
    gl.static_assert((BM == 16) | (BM == 32))
    # A logical batch is one disjoint K slice, placed on one physical CTA.
    # Keeping the CTA dimension explicit lets gl.sum generate the cluster
    # communication and synchronization rather than relying on global atomics.
    cga: gl.constexpr = [[1, 0, 0], [2, 0, 0], [4, 0, 0]][:C.bit_length() - 1]
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 4, 8], [1, 4, 1], [2, 1, 0], cga_layout=cga)
    acc_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, 2, 2], instr_shape=[1, 16, 8], cga_layout=cga)
    lhs_layout: gl.constexpr = gl.DotOperandLayout(0, acc_layout, 2)
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, acc_layout, 2)
    mt = gl.cdiv(M, BM)
    tile = gl.program_id(0)
    m0 = (tile % mt) * BM
    n0 = (tile // mt) * BN
    slices = gl.arange(0, C, layout=gl.SliceLayout(1, gl.SliceLayout(2, load_layout)))
    ns = n0 + gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(2, load_layout)))
    ms = m0 + gl.arange(0, BM, layout=gl.SliceLayout(0, gl.SliceLayout(2, load_layout)))
    ks = gl.arange(0, BK, layout=gl.SliceLayout(0, gl.SliceLayout(1, load_layout)))
    high = gl.zeros((C, BN, BM), gl.float32, layout=acc_layout)
    low = gl.zeros((C, BN, BM), gl.float32, layout=acc_layout)
    for step in range(K // (C * BK)):
        k = slices[:, None, None] * (K // C) + step * BK + ks[None, None, :]
        woffsets = ns[None, :, None] * K + k
        xoffsets = ms[None, :, None] * K + k
        wh = gl.load(WH + woffsets, ns[None, :, None] < N, other=0)
        wl = gl.load(WL + woffsets, ns[None, :, None] < N, other=0)
        x = gl.load(X + xoffsets, ms[None, :, None] < M, other=0)
        xdot = gl.convert_layout(x.permute((0, 2, 1)), rhs_layout)
        high = mma_v2(gl.convert_layout(wh, lhs_layout), xdot, high)
        low = mma_v2(gl.convert_layout(wl, lhs_layout), xdot, low)
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2], [1, 4, 8], [1, 4, 1], [2, 1, 0], cga_layout=cga)
    partials = gl.convert_layout(high + low, reduce_layout)
    # The reduced axis is exactly the axis sharded across the cluster. The
    # compiler owns the distributed-shared-memory exchange and cluster barriers.
    combined = gl.sum(partials, axis=0)
    # After reducing C, all CTAs have the same logical output. Replicated CTA
    # bases let the compiler select unique store owners, avoiding output races.
    store_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    values = combined
    out_n = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, store_layout))
    out_m = m0 + gl.arange(0, BM, layout=gl.SliceLayout(0, store_layout))
    offsets = out_m[None, :] * N + out_n[:, None]
    mask = (out_m[None, :] < M) & (out_n[:, None] < N)
    gl.store(Y + offsets, values, mask)
