"""Signal PDL consumers before producer math; consumers retain their GDC wait.

The existing producers remain unchanged and also signal at their epilogue.
Repeated triggers are allowed; only the first signal from each CTA matters.
"""

import triton
import triton.language as tl

from .gemm import _gemm, _weight_major_stream_gemm
from .tma import _tma_weight_major


@triton.jit
def _gemm_early(X, WH, WL, Y, P, M: tl.constexpr, N: tl.constexpr,
                K: tl.constexpr, S: tl.constexpr, ROUTER: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    if S > 1:
        tl.extra.cuda.gdc_launch_dependents()
    _gemm(X, WH, WL, Y, P, M, N, K, S, ROUTER, BM, BN, BK)


@triton.jit
def _weight_major_stream_gemm_early(X, W, Y, P, M: tl.constexpr, N: tl.constexpr,
                                   K: tl.constexpr, S: tl.constexpr,
                                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    if S > 1:
        tl.extra.cuda.gdc_launch_dependents()
    _weight_major_stream_gemm(X, W, Y, P, M, N, K, S, BM, BN, BK)


@triton.jit
def _tma_weight_major_early(WD, XD, Y, P, M: tl.constexpr, N: tl.constexpr,
                           K: tl.constexpr, S: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    if S > 1:
        tl.extra.cuda.gdc_launch_dependents()
    _tma_weight_major(WD, XD, Y, P, M, N, K, S, BM, BN, BK)
