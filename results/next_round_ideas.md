# Ideas after the round-three layout result

**Implemented follow-up:** the [expanded 21-idea map](optimization_idea_map.md)
and [round-four experiment](multi_category_round4_experiment.md) supersede the
shortlist below. One candidate now tests four independent layout groups; its
GPU check is pending.

The consolidation check is running. Planning below uses its **hypothetical
3858.27 us** score, reconstructed from base `67a1b145d83b4f89` with selected
round-three per-case ratios. Replace that reference with the actual passing
consolidation result before finalizing the next candidate. No check was started
while preparing this plan, and no in-flight candidate source was modified.

The new evidence favors exploring physical layout: packed pooler M128 improved
9.3%, while guessed larger K tiles and split reductions lost. The source also
reveals two untested opportunities: weight-major kernels transpose their stores
into a contiguous `[M,N]` result, and router GEMM carries separate high/low dots.

## Ranked experiments

| Rank | Experiment | Reason to try | Scope and interpretation |
|---|---|---|---|
| 1 | Compute into natural `[N,M]` output and return a transpose view | Small implementation change. A view adds no GPU copy. It can change cuBLAS's chosen algorithm and remove an output-layout conversion in our weight-major kernel. | Test reversed cuBLAS on large QKV and full unpooler; test natural stores in existing O orientation. Preserve the currently winning weight layout within each case. The local contract checks shape/dtype and values, not output contiguity. A gain is a hypothesis until measured. |
| 2 | Prepare tile-major BF16 weights for six small QKV/pooler cases | These cases account for about 27.5% of projected score sensitivity. Pooler producer time dominates, so improving loads/main loop has more room than shortening its reducer. | Pack `[N/64,K/128,64,128]` once, with contiguous 64-by-128 weight tiles. Preserve logical dot orientation, BM/BN/BK, splits, stages, FP32 accumulation and output layout for the first comparison. Change only physical weight storage/addressing. |
| 3 | Interleave router high/low weights and compute one wider dot | Test the two-component arithmetic as a single wider matrix multiplication. This could change load/layout scheduling while keeping the same total arithmetic and FP32 accuracy. | Prepare `[expert,component,K]` BF16 weights, flatten expert/component into the GEMM output-feature dimension, accumulate both components independently in FP32, and add each pair in the producer epilogue. Preserve logical expert tile sizes, splits, PDL and scratch shape. Test M8/16/32/128/256; M1 remains original FP32 GEMV. |
| 4 | A real Hopper load/compute pipeline for larger GEMMs | Large QKV/pooler still has meaningful headroom; earlier TMA kernels were ordinary descriptor loops with no explicit warp-specialized pipeline. | Prototype a documented Hopper-compatible descriptor/persistent pipeline. Audit actual async instructions, register/shared-memory use and spills before allocating a GPU check. Persistence by itself does not fix a grid that is too small. More work and higher risk than the first three. |
| 5 | Single-pass router with K reduction inside each CTA | Removing partials and the second kernel is a larger scheduling change than varying splits. | Use original FP32 weights with a CTA owning an expert and a small token group, reusing its weight vector across tokens. Need enough expert CTAs to fill the GPU and a bounded token group to control register pressure. Existing chunked M1 failure warns that another serial K loop can be slow. Prototype after the wider-dot experiment. |
| 6 | Tensor-core or packed-layout GEMV for BF16 single-token cases | The four BF16 M1 cases represent about 14.7% of projected sensitivity. A different reduction layout could improve memory throughput. | Tensor-core padding and split-K can add wasted math and a reducer; simple reuse of the current small-batch launch is not an established improvement. Defer until the weight-packing experiment gives useful evidence. |

## Concrete details for the cheap layout experiment

Instead of `torch.mm(x, w.T)`, try `torch.mm(w, x.T).T`. Where the retained
weight is a contiguous `[K,N]` copy `wt`, try `torch.mm(wt.T, x.T).T` so only
the computation/output orientation changes. The returned tensor remains a
complete `[M,N]` result; its strides differ. Do not append `.contiguous()` to
the experimental return, which would introduce a separate copy.

For custom O, allocate `[N,M]`, store `acc[n,m]` at `n*M+m`, and return its
transpose view. Keep the existing input loads, tensor-core math, tile sizes
and stages. Inspect compiler layout-conversion counts before claiming that
the changed stores remove any conversion.

The local checker `checks.compare` checks shape/dtype, finite values and
relative error; `reference.check_implementation` also checks probe rows.
Neither requires a contiguous result. Fresh outputs are computed on every
call; all activation work remains timed.

## Budget and proposed next check

The first combined candidate should contain three disjoint groups:

1. Reversed cuBLAS for QKV M128/M256 and full unpooler M16/M32/M128/M256,
   plus natural output stores in O M8/M16/M32.
2. Tile-major prepared weights for QKV/pooler M8/M16/M32.
3. Interleaved high/low wider router dots for M8/M16/M32/M128/M256.

Leave large pooler and single-token paths as reference cases. This compares
the new ideas without applying two new changes to the same scored case.
Validate each new path on nearby shapes and inspect compiler resources where
available.

For scale, relative to the projected consolidation reference:

| Group | Sensitivity | Score saving if this group takes 20% less time |
|---|---:|---:|
| Six small QKV/pooler cases | 27.5% | 218 us |
| Large QKV plus full unpooler | 25.5% | 200 us |
| Multi-token router | 9.1% | 71 us |
| Large pooler | 10.4% | 82 us |
| Four single-token BF16 cases | 14.7% | 123 us |

These numbers value a hypothetical improvement; they do not forecast one.
The first two groups each 20% faster plus multi-token router 30% faster would
project about **3335 us**, before any new O gain. Their score savings are not
additive. Further gains would still be needed for the user-reported 2950 us
target. Use the actual consolidation result to refresh these calculations.

Source references:

- [PyTorch tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
  documents transpose/view behavior and when `contiguous()` makes a copy.
- [Triton warp specialization](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html)
  documents load/compute partitions on Hopper and newer, including synchronization
  and resource tradeoffs.
- [Triton 3.8.0 release](https://github.com/triton-lang/triton/releases/tag/v3.8.0)
  records Hopper WGMMA/Gluon and warp-specialization support relevant to the
  recorded judge compiler. Use supported Hopper patterns rather than assuming
  every Blackwell tutorial variant applies.
