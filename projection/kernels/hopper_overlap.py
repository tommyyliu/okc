"""Hopper pipeline retaining one WGMMA group while retiring the previous slot."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, warpgroup_mma, warpgroup_mma_init, warpgroup_mma_wait

from .hopper_ws import _hopper_ws_load


@gluon.jit
def _hopper_overlap_compute(Y, P, wbuf, xbuf, empty, ready, n0, m0, split,
                            M: gl.constexpr, N: gl.constexpr, S: gl.constexpr,
                            BM: gl.constexpr, BN: gl.constexpr, STEPS: gl.constexpr,
                            BUFFERS: gl.constexpr):
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BM, 16])
    acc = warpgroup_mma_init(gl.zeros((BN, BM), gl.float32, layout=mma_layout))
    for step in range(STEPS):
        slot = step % BUFFERS
        phase = (step // BUFFERS) & 1
        mbarrier.wait(ready.index(slot), phase)
        acc = warpgroup_mma(wbuf.index(slot), xbuf.index(slot).permute((1, 0)),
                            acc, is_async=True)
        # Two groups may now be queued. Retire the older one, leaving the newest
        # accumulator in the async proxy; init is a type wrapper, not a reset.
        acc = warpgroup_mma_init(warpgroup_mma_wait(num_outstanding=1, deps=(acc,)))
        retired = (step + BUFFERS - 1) % BUFFERS
        mbarrier.arrive(empty.index(retired), count=1, pred=step >= 1)
    values = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    mbarrier.arrive(empty.index((STEPS - 1) % BUFFERS), count=1)
    if S > 1:
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;", "=r", [],
            dtype=gl.int32, is_pure=False, pack=1)
    store_layout: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [2, 2], [0, 1])
    values = gl.convert_layout(values, store_layout)
    ns = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, store_layout))
    ms = m0 + gl.arange(0, BM, layout=gl.SliceLayout(0, store_layout))
    offsets = ms[None, :] * N + ns[:, None]
    mask = (ms[None, :] < M) & (ns[:, None] < N)
    if S == 1:
        gl.store(Y + offsets, values, mask)
    else:
        gl.store(P + split * M * N + offsets, values, mask)


@gluon.jit
def _hopper_overlap_gemm(WD, XD, Y, P, M: gl.constexpr, N: gl.constexpr,
                         K: gl.constexpr, S: gl.constexpr, BM: gl.constexpr,
                         BN: gl.constexpr, BK: gl.constexpr, BUFFERS: gl.constexpr):
    gl.static_assert(BM == 64 and BN == 64)
    gl.static_assert(BUFFERS >= 2 and K % (S * BK) == 0)
    mt = gl.cdiv(M, BM)
    tile = gl.program_id(0)
    split = gl.program_id(1)
    m0 = (tile % mt) * BM
    n0 = (tile // mt) * BN
    k0 = split * (K // S)
    wbuf = gl.allocate_shared_memory(WD.dtype, [BUFFERS] + WD.block_type.shape, WD.layout)
    xbuf = gl.allocate_shared_memory(XD.dtype, [BUFFERS] + XD.block_type.shape, XD.layout)
    empty = gl.allocate_shared_memory(gl.int64, [BUFFERS, 1], mbarrier.MBarrierLayout())
    ready = gl.allocate_shared_memory(gl.int64, [BUFFERS, 1], mbarrier.MBarrierLayout())
    for slot in gl.static_range(BUFFERS):
        mbarrier.init(empty.index(slot), count=1)
        mbarrier.init(ready.index(slot), count=1)
    gl.warp_specialize([
        (_hopper_overlap_compute, (Y, P, wbuf, xbuf, empty, ready, n0, m0, split,
                                   M, N, S, BM, BN, K // (S * BK), BUFFERS)),
        (_hopper_ws_load, (WD, XD, wbuf, xbuf, empty, ready, n0, m0, k0,
                           K // (S * BK), BK, BUFFERS)),
    ], [1], [24])
    for slot in gl.static_range(BUFFERS):
        mbarrier.invalidate(empty.index(slot))
        mbarrier.invalidate(ready.index(slot))
