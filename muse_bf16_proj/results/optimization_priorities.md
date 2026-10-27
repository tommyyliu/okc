# Optimization priorities after the control recheck

**Latest update:** nearby weight-major check `67a1b145d83b4f89` passed at
3934.4243 us / 1.2278289x, including new-kernel correctness at pooler M=7.
[The updated opportunity analysis](opportunity_analysis.md) uses that result
and the user's approximate 3000 us target. It calls for broader pooler/QKV
improvements and raises router scheduling as an independent opportunity.
The historical control calculations below remain tied to `706c118665f14447`.

Control `706c118665f14447` measures **1.2197188x**, 95% interval
[1.2014230, 1.2380146]. It passed correctness and is consistent with yesterday's
1.2310594x within the reported intervals. `pdl_recheck.json` is the current
measurement reference; `pdl.json` remains the historical one.

The score is a geometric mean of six load speedups. A case's priority should
depend on its share of its load, not just absolute microseconds or per-case
speedup. The shares below average those within-load fractions over all loads.
The groups in this table do not overlap. Calculations for these groups, each
layer, and all thirty individual cases are in `optimization_priorities.json`.

| Group | Scored cases | Score-weighted latency share | Score if this group takes 20% less time | Score at reported floors |
|---|---:|---:|---:|---:|
| QKV + pooler, M=8/16/32 | 6 | 27.57% | 1.2931x | 1.3497x |
| Full unpooler, M=16/32/128/256 | 4 | 18.03% | 1.2658x | 1.2805x |
| All single-token projections | 5 | 16.67% | 1.2659x | 1.3067x |
| Pooler, M=128/256 | 2 | 10.89% | 1.2475x | 1.2834x |
| QKV, M=128/256 | 2 | 7.07% | 1.2375x | 1.2665x |

These are hypothetical score calculations, not speedup predictions. The task's
floors omit practical inefficiencies and are optimistic uncompressed bounds.
Reaching the recorded 1.61x target requires 24.24% less aggregate latency from
the new control. Even the floor scenario for any individual case improves the
aggregate by less than 3%. A shared improvement across several sizes matters.

## First priority: the six small QKV/pooler cases

Pooler takes 19.23/19.29/20.44 us at M=8/16/32. QKV takes
12.54/13.38/14.42 us. Both use K=8192 and row-major BF16 weights, so a useful
kernel design may transfer between them while retaining separate launch settings.

Five of the six cases use our streaming GEMM; QKV M=32 still uses cuBLAS.
The fresh profile shows the pooler producer consumes about 93% of its profiled
GPU time. Its reducer consumes 1.2–1.5 us, so eliminating that reducer alone is
not a large improvement. Profile timing is a separate pass and should not be
subtracted directly from the score timings.

Investigate weight loads and the GEMM main loop first: K tile size, pipeline
depth, operand orientation, and the number of independent K partitions. Use
separate choices for each (layer, M). Prefer a bounded search over these choices
to another single universal tile guess. Keep cuBLAS as a competitor for every
case, and preserve the measured PDL dispatch unless a candidate wins.

`tools/tune.py` now searches streaming tiles, split-K, pipeline stages, and both
operand orientations while retaining the source dispatch and cuBLAS controls.
The initial streaming and weight-major checks both passed correctness.
Weight-major measured 1.2324506x overall and reduced normalized latency by
2.04–5.51% across the six cases versus streaming. The aggregate confidence
intervals overlap and unchanged cases also vary, so PDL remains the default.
See [the experiment record](small_gemm_experiment.md) for retained evidence and
the next check: `build/small-gemm/weight_major_nearby.py` preserves the scored
launches and extends QKV/pooler dispatch to all M=2..32, exercising the new
kernel on nearby pooler M=7 while collecting another timing observation.

## Second priority: single-token GEMV

The five single-token cases are one complete load. They account for exactly
one-sixth of the score's latency sensitivity. The large-K paths currently use
one output row per program and four warps. Compare multiple rows per program,
warp counts, and a chunked reduction against the full-K reduction. The existing
tuner already supports a small rows/warps sweep. A GPU result, rather than
register count alone, must select the winner.

## Larger GEMMs

Full unpooler is a large latency share but already near cuBLAS and only about
24–29% removable latency to reach the reported floors. Keep it in the
search but do not assume its 18% share translates into an easy 18% improvement.
Large QKV has roughly 2x reported-floor headroom. M=128's cuBLAS profile includes
a roughly 3 us split-K reduction, making it a concrete later scheduling target.

Pause large-pooler TMA work until better profiling or a materially different
design is available. The initial variant lost at both sizes. Split-K helped
M=128 versus that failed variant but still lost to same-run SGLang; M=256
regressed further. These outcomes do not establish that TMA itself is unsuitable,
but do not support continuing to vary split counts without more evidence.

## Experiment discipline

Use `submission.py` as the PDL control. Require correctness on scored and nearby
shapes, compare candidate/SGLang ratios within each run, and retain only per-case
winners. Recheck promising small gains. The next objective is a reproducible
improvement across several small-batch cases, not recovering the failed TMA score.
