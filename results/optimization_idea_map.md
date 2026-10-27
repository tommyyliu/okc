# Expanded optimization map

This map follows the passing round-three check `327bb5b2fdc6423a`. The selected
consolidation candidate projects 3858.27 us; its result has not landed in the
workspace. The user-reported competitor target is 2950 us. New-kernel benefits
below are hypotheses, not timing predictions.

Ideas 1–4 were tested in [round four](multi_category_round4_experiment.md), which
passed at 3884.83 us without a clear broad gain from the new groups. Tiled weights
and native O stores were inconclusive; reversed QKV M256 and interleaved router
M8 regressed. The [round-five plan](multi_category_round5_plan.md) selects loop
unrolling, full-K GEMV warp count and a single-pass router. Other ideas remain
branches to choose from evidence, rather than an automatic queue of checks.

| # | Idea | Scope | Why it could help / what to measure | Priority |
|---:|---|---|---|---|
| 1 | Prepare contiguous weight tiles | QKV/pooler M8/16/32 | Pack `[N/64,K/128,64,128]`; each producer loads one contiguous tile instead of segments separated by the original K stride. Preserve math, tile sizes, pipeline and splits. | Included now |
| 2 | Reverse the cuBLAS operands and return a view | Large QKV and full unpooler | `(W @ X.T).T` presents a different GEMM/output layout. Preserve the currently selected original/transposed weight storage. Inspect chosen library kernels and total graph time. | Included now |
| 3 | Store O in its native weight-major output layout | O M8/16/32 | Backing output `[N,M]` matches the accumulator; return a transpose view. Test whether stores/layout conversions matter while retaining the dot loop. | Included now |
| 4 | Interleave router hi/lo features into one wider dot | Router M8/16/32/128/256 | Independent FP32 accumulations in adjacent physical features, pairwise addition in the producer epilogue. Same arithmetic, logical grid, splits and scratch; test load/instruction scheduling. | Included now |
| 5 | Native output and partial layout for split weight-major GEMM | Six small QKV/pooler cases | Extend idea 3 to `[S,N,M]` scratch and a native-layout reducer. It changes both producer stores and reducer traversal, so measure the complete call. | Next if native O gains |
| 6 | Simple transposed weights as an alternative to tile-major packing | Small QKV/pooler | Compare contiguous `[K,N]` versus tile-major `[N/BN,K/BK,BN,BK]`; retain a clear logical orientation comparison. Large cuBLAS benefited from storage layout, which does not establish the best custom-kernel layout. | Next after packing result |
| 7 | Explicit loop pipeline depth | Winning small-GEMM layout | Compare `tl.range(..., num_stages=2/3/4)` with the current kernel stage setting. Loop staging and kernel-level staging have different compiler scope. Audit shared memory and spills first. | Medium |
| 8 | Unroll the short K loop | Small QKV/pooler and router | Current small QKV has eight K iterations per split, pooler sixteen. Try an unroll factor of two before four; check whether generated code changes and whether register use grows. | Medium, bounded |
| 9 | Separate weight and activation cache policies | Custom GEMMs | Compare streaming weight loads with `.cg` and activation loads with `.ca`. Look for useful reuse within a replay; every measured replay starts cold. | Medium, one factor |
| 10 | Reorder output/K-partition CTA traversal | Custom split GEMMs | Change which tile/split varies fastest to alter concurrent memory requests and activation reuse. The current weight-major order already places token tiles together; an alternative needs evidence. | Medium after producer win |
| 11 | Tune warps/register allocation from compiler resources | Winning custom kernels | Compare warp count or a register cap only after seeing register/shared-memory pressure. A cap can trade resources for spills; reject configurations that simply move the cost. | Medium, compiler prerequisite |
| 12 | Change GEMV warp count with the full-K reduction retained | BF16 M1 and router M1 | Test two/eight warps against four while keeping row count and arithmetic fixed. This avoids conflating warp scheduling with the failed rows2 or chunked-loop ideas. | Small bounded experiment |
| 13 | Explicit warp/lane layout for GEMV | Primarily pooler/QKV M1 | A Gluon implementation could make vectorization and row/K ownership explicit. Inspect compiled loads and register use before a GPU check. | More implementation work |
| 14 | BF16 tensor-core GEMV | QKV/pooler/unpooler M1 | Pad the logical token tile, retain BF16 weights and FP32 accumulation. Padding math and a possible reducer may outweigh better weight handling. Existing small-batch settings are not a measured M1 win. | Longer shot |
| 15 | Single-pass FP32 router | Small multi-token router | A CTA owns an expert and a small token group, reusing FP32 weights and reducing K internally. Removes scratch/reducer, but token grouping and K loops can create register or latency problems. | Architectural follow-up |
| 16 | Hopper load/compute warp specialization | Large QKV/pooler | Build a genuine supported Hopper pipeline with independent load/compute partitions and asynchronous math. The failed descriptor kernels did not implement that design. Require offline compiler/resource validation first. | High upside, high effort |
| 17 | Persistent scheduling and within-replay weight reuse | Large M128/M256 GEMMs | Reuse a weight tile across token tiles and overlap tile transitions. A persistent launch alone does not solve insufficient parallel work; count CTAs and memory traffic explicitly. | After a competitive pipeline |
| 18 | Change the split-partial memory order | Router or weight-major GEMM | Compare split-major with a layout favoring reducer reads. Current reads are already coalesced within each split, so producer-write cost could exceed any benefit. | Lower priority |
| 19 | Tune reducer block width/warps | Split paths | B128/B512 or two warps versus B256/four. The reducer is a small fraction of pooler time; use only as an independent cheap side experiment. | Low score ceiling |
| 20 | FP32 cuBLAS output followed by BF16 conversion | Library-backed BF16 GEMM | A different output type may select a different library algorithm. The conversion must remain timed; require a GEMM gain that exceeds the added launch/traffic. | Longer shot |
| 21 | A different lossless BF16 encoding | Weight-bandwidth paths | Fixed-layout exponent/mantissa packing could reduce bytes without changing precision. The archived codec measured about 0.1076x overall, so a new design needs evidence that decoding is cheap before another check. | Research backlog |

## Payoff model

Use the consolidation result when available. With the current hypothetical
3858.27 us reference, reducing latency by 20% in each group individually buys:

| Group | Score saving |
|---|---:|
| Six small QKV/pooler cases | 218 us |
| Large QKV plus full unpooler | 200 us |
| Five multi-token router cases | 71 us |
| Large pooler | 82 us |
| Four BF16 single-token cases | 123 us |

These values measure the payoff of a gain, not its probability. Savings are
not additive. The first two groups 20% faster and multi-token router 30% faster
project about 3335 us, before any new O improvement. That scenario would still
need further gains to reach 2950 us.

## Result-driven branches

- If tiled weights improve both small layer families, inspect their generated
  loads and use that layout for a bounded pipeline/unroll comparison.
- If reversed cuBLAS changes algorithms and wins, retain settings by case;
  a single common layout has already proved inappropriate for QKV versus pooler.
- If native O stores win, extend that output-layout idea to split paths.
- If interleaved router weights win, retain the same precision and compare a
  small split/pipeline follow-up. If producer time remains unchanged, prioritize
  the single-pass design rather than more memory interleaving variants.
- Larger pipeline work should compete against the selected cuBLAS layout,
  including the newly passing layouts, rather than against a failed custom tile.

## Primary implementation references

- [PyTorch views](https://docs.pytorch.org/docs/stable/tensor_view.html): transpose
  views and when `contiguous()` copies.
- [Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html):
  `.ca`/`.cg` cache semantics and masked loading.
- [Triton range](https://triton-lang.org/main/python-api/generated/triton.language.range.html):
  loop pipeline/unroll controls and the difference from kernel-level stages.
- [Triton Config](https://triton-lang.org/main/python-api/generated/triton.Config.html):
  warps, stages, register limits and cluster configuration.
- [Gluon warp specialization](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html):
  Hopper-compatible partitioning and resource/synchronization tradeoffs. Check
  the judge's Triton 3.8 support; the generic `tl.range` warp-specialization
  documentation is not a blanket guarantee of Hopper support.
