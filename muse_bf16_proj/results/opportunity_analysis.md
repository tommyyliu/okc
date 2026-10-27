# Opportunities from the 3934 us check

Source: `small_gemm_weight_major_nearby.json`, job `67a1b145d83b4f89`.
All nine correctness tests and thirty scored cases passed, including the new
weight-major kernel at nearby pooler M=7. Score: 3934.4243 us, 1.2278289x,
95% speedup interval [1.2094115, 1.2462464].

The user's approximate 3000 us competitor score is the target here; it has not
been refreshed from a live leaderboard. Reaching it requires 23.75% less
latency, or a 1.3115x throughput improvement. A hypothetical uniform 20% time
reduction gives 3147.5 us, still above target.

`opportunity_analysis.json` records all calculations. Every scenario rebuilds
weighted load sums before taking the geometric mean. Shares are within-load
weighted latency fractions averaged across six loads. Single-area savings
below are not additive. Floors are optimistic bounds, not performance forecasts.

| Layer, across all token counts | Score sensitivity share | Score saving if 20% less time | Score saving at reported floors |
|---|---:|---:|---:|
| Pooler | 33.97% | 267 us | 416 us |
| QKV | 21.91% | 172 us | 341 us |
| Unpooler | 21.73% | 172 us | 278 us |
| Router | 10.97% | 86 us | 308 us |
| O projection | 11.42% | 90 us | 214 us |

Even putting both pooler and QKV at their reported floors gives about 3175 us.
Closing the complete gap therefore needs improvements beyond those two layers
under this uncompressed floor model. Putting every case at its reported floor
gives 2373 us; the 3000 us target is above that bound, which does not establish
that a practical implementation can achieve it.

## Concrete opportunities

The [pooler-specific ranked ideas](pooler_optimization_ideas.md) compare its
small-token GEMM, single-token GEMV, and large-token cuBLAS cases separately.
Three focused candidates for that sequence are staged under
`build/pooler-round1/`; see its manifest before using an OKC check.

1. **Pooler across sizes.** The largest share. At M=128/256 it takes 25.13/31.45
   us versus floors of 14.64/17.37 us. At small M its producer still occupies
   over 92% of profiled GPU time. Investigate weight layout, main-loop pipeline,
   tile reuse and per-size scheduling. Weight repacking in `prepare` is allowed
   at the original BF16 precision. Earlier TMA candidates lost, so a new design
   needs profiling or a materially different layout rather than more blind
   split-count changes. None of these hypotheses is a measured new speedup.
2. **QKV, especially M=128/256.** These two cases take 17.33/20.02 us versus
   floors of 8.42/9.77 us and still dispatch to cuBLAS. M=128's separate profile
   includes a 3.0 us reducer. Evaluate a dedicated kernel and scheduling for
   these sizes while retaining cuBLAS as the control. Small-QKV improvements
   alone cannot recover the large-size headroom.
3. **Router scheduling.** It takes 5.2–10.1 us versus floors of 1.75–2.68 us.
   The multi-token path splits K 16 or 32 ways; reducers consume 1.3–2.1 us in
   the profiling pass, and at M=16 the reducer is 36% of profiled time. Compare
   tiles and partition counts that reduce scratch and launch overhead while
   preserving the FP32-weight hi/lo arithmetic. The theoretical headroom is
   large for its share, which warrants moving it above its former low priority.
4. **Unpooler and GEMV.** Full unpooler has 18.29% sensitivity but only about
   25–30% latency above its floors as a removable fraction, and is already
   close to same-run cuBLAS. Single-token projections form a whole load with
   16.67% sensitivity; rows per program, warp count and chunked reductions
   remain bounded, useful experiments. All single-token cases at floor would
   save about 270 score us, so this also cannot close the gap alone.

The six small QKV/pooler cases currently represent 27.05% of sensitivity.
Making all six 20% faster saves about 219 score us; even reaching all six
reported floors saves only 335 us. A 4% improvement confined to that group is
roughly a 1.1% aggregate improvement. The weight-major result is useful evidence
about operand layout, but it has the scale of a small contribution to the
934 us gap. Substantial reductions across several kernel families are needed.

Recommended order: finish a bounded small-GEMM sweep, then pursue large
pooler/QKV designs and a separate router scheduling experiment. Keep independent
per-case winners and require repeated normalized gains before default promotion.
