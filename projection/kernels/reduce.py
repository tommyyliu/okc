"""Projection GPU kernels; tune launch settings in projection/configs.py."""

import triton
import triton.language as tl


@triton.jit
def _reduce(P, Y, TOTAL: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    # Completion, including visibility of every producer's partial stores.
    tl.extra.cuda.gdc_wait()
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    splits = tl.arange(0, S)
    values = tl.load(P + splits[:, None] * TOTAL + offsets[None, :],
                     offsets[None, :] < TOTAL, other=0)
    tl.store(Y + offsets, tl.sum(values, axis=0), offsets < TOTAL)
