# Round five plan after round-four timings

**Completed:** the [round-five candidate](multi_category_round5_experiment.md)
passed correctness at 4321.59 us. Retain the four eight-warp GEMV cases; reject
the six unroll2 settings and three direct-router settings. The details below
record the original planning rationale and resource risks, not a pending plan.

Round four completed as `0fe4f6662e1d458a`: all correctness tests and timed
cases passed, at **3884.83 us / 1.2435046x**, interval [1.2248520x, 1.2621571x].
Its raw response/profile and comparisons are retained as
`multi_category_round4.json/txt` and `multi_category_round4_analysis.json`.
The separate consolidation result is still unavailable locally, so comparisons
use the relevant per-case predecessor jobs.

## What changes the plan

- Tiled small-GEMM weights gave no clear improvement: normalized latency was
  about 0.5–4.4% slower across the six cases. Use original row-major weight
  storage with the passing weight-major arithmetic for the next comparison.
- Native O stores shifted about -2.7% to +1.9%, within unchanged-case variation.
  There is insufficient evidence to prioritize native split-partial layouts.
- Reversed cuBLAS was essentially flat on unpooler and QKV M128; QKV M256
  regressed about 10%. Its profile selects split-K plus a 4.1 us reducer.
  Keep the previously selected library orientation and defer extending reversal
  to large pooler/O.
- Interleaved router M8 regressed about 11%. Other router changes, from roughly
  3–5% faster to flat, are not decisive given unchanged-case variation from
  -3.7% to +4.9%. Keep the original producer, including the repeated M128 split16
  gain, and investigate eliminating the second launch instead.

## Proposed next payload: three disjoint groups, thirteen cases

| Priority | Group | Cases | One new factor |
|---:|---|---:|---|
| 1 | Small QKV/pooler | M8/M16/M32 in both layers: 6 | Unroll the K loop by two on the passing original-storage weight-major kernel |
| 2 | Full-K GEMV | QKV/pooler/unpooler/router M1: 4 | Four to eight warps, preserving full-K reduction and one output row per program |
| 3 | Single-pass FP32 router | Router M8/M16/M32: 3 | One kernel with token-pair weight reuse and internal K reduction, replacing split partials plus reducer |

Each group uses different scored cases. Selected earlier O orientation,
pooler/unpooler transposes and router M128 split16 remain controls; their repeat
observations can confirm or weaken earlier small gains. Large QKV retains its
original library orientation. Build this provisional control explicitly from
passing evidence; do not treat the round-four aggregate as evidence for every
changed layout.

## Implementation sketches

**Small GEMM:** QKV has eight K iterations per partition and pooler sixteen.
Use a two-iteration unroll while keeping dot ordering, BM/BN/BK, split counts,
warps, stages, partial precision and output layout fixed. Higher register use
is the main resource risk. Check generated code/resources when the compiler is
available; if code is unchanged or spills appear, replace this idea with a
bounded pipeline-depth comparison before spending a check. Native partials
remain a later alternative, not a simultaneous change to these six cases.

**GEMV:** use the existing full-K kernel with eight warps instead of four.
The proposed four cases have K4096/8192 and one output row per program. O M1
retains its four-row configuration. The earlier rows2 and chunked-loop failures
did not measure this combination. Keep original FP32 router weights, BF16
weights for the other layers and the existing output dtypes.

**Router prototype:** one CTA owns an expert and two tokens. Load the expert's
original FP32 weight vector, load two BF16 activation rows, widen operands for
FP32 multiplication/reduction and write two FP32 logits directly. Use a full
8192-element K vector to avoid another serial chunked loop, with eight warps
as the initial layout. The product tile has 16384 FP32 elements, averaging 64
elements per thread before accounting for other live values; this is a
resource sketch, not a compiler register measurement. The M8/M16/M32 grids
contain 1024/2048/4096 CTAs. Mask nearby odd token counts such as the M9
outlier test. There is no scratch, reducer, atomic initialization or retained
activation state. Weight traffic and register pressure may still outweigh the
saved launch, so compare complete cold-L2 graph calls.

## Payoff model

Using **measured round-four timings** only to value the affected case groups:

| Group | Sensitivity | Score saving if the group takes 20% less time |
|---|---:|---:|
| Six small QKV/pooler cases | 27.4% | 219 us |
| Four larger-K M1 cases | 14.4% | 121 us |
| Three small-router cases | 5.6% | 44 us |

All three groups 20% faster would project about **3506 us** from the 3884.83 us
round-four measurement. This is a conditional payoff calculation, not a timing
prediction, and savings are not additive. The next control restores unhelpful
round-four experiments, so refresh final estimates for that exact selection.
`build/multi-category-round5/planning-budget.json` records the calculation.
The user-reported 2950 us target will require further gains beyond this scenario.

## Before requesting a check

1. Finalize the per-case control from passing evidence and incorporate the
   separate consolidation response if it becomes available.
2. Build the loop variant and router prototype; verify pointer coverage,
   FP32 arithmetic, output dtype, token-pair masking and cloned-state binding.
3. Audit compilation/resources where available. Torch/Triton are absent locally;
   do not describe CPU checks as GPU compiler or numerical validation.
4. Request one combined payload. Keep the original router path available if
   the prototype cannot be made reviewable without an extra exploratory check.

The larger Hopper load/compute pipeline remains an architectural follow-up
against the best selected cuBLAS layout. More guessed large tiles have little
support from the latest measurements.
