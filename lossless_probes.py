"""Development kernels for decoder benchmarks; never imported by submissions."""

import triton
import triton.language as tl

from submission_triton_lossless import _decode_lossless


@triton.jit
def _decode_aligned(Payload, Offsets, Bases, groups, lanes, valid):
    # The packer emits 384/512-byte blocks, so every start is 128B aligned.
    start = tl.multiple_of(tl.load(Offsets + groups, valid, other=0), (128, 1))
    base = tl.load(Bases + groups, valid, other=256).to(tl.uint32)
    compact = base < 256
    sm = tl.load(Payload + start + lanes, valid, other=0).to(tl.uint32)
    index = tl.where(compact, lanes // 2, lanes)
    code = tl.load(Payload + start + 256 + index, valid, other=0).to(tl.uint32)
    delta = (code >> ((lanes & 1) * 4)) & 15
    exponent = tl.where(compact, base + delta, code)
    bits = (sm & 127) | ((sm & 128) << 8) | (exponent << 7)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _decode_matrix(Payload, Offsets, Bases, Y, GROUPS: tl.constexpr,
                   G: tl.constexpr, ALIGNED: tl.constexpr):
    groups = tl.program_id(0) * G + tl.arange(0, G)
    lanes = tl.arange(0, 256)
    if ALIGNED:
        w = _decode_aligned(Payload, Offsets, Bases, groups[:, None],
                            lanes[None, :], groups[:, None] < GROUPS)
    else:
        w = _decode_lossless(Payload, Offsets, Bases, groups[:, None],
                             lanes[None, :], groups[:, None] < GROUPS)
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
        exponent = base + tl.reshape((packed[:, None] >> shifts[None, :]) & 15, (256,))
    else:
        exponent = tl.load(Payload + start + 256 + lanes).to(tl.uint32)
    bits = (sm & 127) | ((sm & 128) << 8) | (exponent << 7)
    w = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(Y + group * 256 + lanes, w)


@triton.jit
def _copy_matrix(W, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(W + offsets, offsets < TOTAL, other=0)
    tl.store(Y + offsets, w, offsets < TOTAL)
