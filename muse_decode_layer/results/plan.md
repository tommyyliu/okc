# Decode experiment plan

The previous FA3 control passed as `2524b1c4986a49b4`: 21184.86 us / 1.0677577x,
with a 95% speedup interval of [1.0517413, 1.0837740]. The earlier FP8-fixed
control remains at 22056.34 us / 1.0255689x. The corrected user-reported leader
is 9580 us / 2.36x. The remaining uniform latency reduction is 54.78%; a
collection of small projection wins alone will not close this gap.

## Current action after submission 6

Cap 8 passed but regressed 7.69%; reject it and retain the 19046.05 us /
1.1876636x submission-5 control. The next frozen candidate is
`build/single-nvfp4/candidate.py`. It introduces direct NVFP4 expert GEMVs
only for T=1, retaining the passing finalize and all larger-token Marlin work.
Twenty CPU checks pass; GPU compilation/correctness/timing are next.
See [the experiment record](single_nvfp4_experiment.md).

This is the first explicit expert kernel architecture rather than another
block-policy sweep. If it works, assess whether grouped/tensor-core variants
can extend the architecture to multi-token workloads; do not run this GEMV
unchanged over every token. It only changes one of six scored loads, so it
cannot close the whole remaining gap by itself.

## Earlier action after submission 5

The small-token fused finalize passed at **19046.05 us / 1.1876636x**, all 13
tests and 12 timed shapes. It reduces the reported score another 6.01%; the
remaining uniform reduction to 9580 us is 49.70%. Keep this as the measured
default, with exact evidence in `results/small-finalize/`.

Next run the frozen `build/moe-m8/candidate.py`: one constant changes from
(8, 16) to (8,), while fused finalize and every GPU function remain unchanged.
This bounds the value of a further Marlin block-size change before a more
substantial expert implementation. The larger expert GEMMs and full-context
attention still dominate; small finalize was a successful launch reduction,
not a solution to the remaining gap. Sixteen CPU checks pass; GPU validation
is pending. See [the experiment record](moe_m8_experiment.md).

## Earlier action after submission 4

Block cap 16 passed as `f2293cc26af04ec1`: **20264.20 us / 1.1162692x**,
interval [1.0995252, 1.1330132]. All 13 tests and 12 timed shapes passed.
Large verify normalized latency improved by 9.75–16.15%; keep this policy
as the new control. The remaining uniform latency reduction to 9580 us is
52.72%.

The next frozen candidate, `build/small-finalize/candidate.py`, replaces the
expert finalize only at T <= 16 with one Triton kernel. It retains BF16
scaled values, FP32 reductions/weighted sum, and masks remote slots before
reading uninitialized Marlin outputs. It targets six small-token scored
shapes; larger-token finalize remains unchanged. Fifteen CPU checks pass;
GPU checking is next. See [the record](small_finalize_experiment.md).

## Workflow adapted from BF16 projections

Keep editable source separate from standalone exports. Freeze each experiment
with source hashes before checking it. Preserve failed evidence, distinguish
measured winners from promising ideas, compare ratios against same-run SGLang,
and confirm gains that fall inside the reported uncertainty. Build and CPU
checks catch organization regressions; GPU checks establish correctness and
timing. Do not transfer an isolated projection speedup directly into a
whole-layer claim.

The source organization preserves the FP8-fixed candidate's exact definition
ASTs, constant values, launch settings, and preparation. No optimization is
bundled into that refactor. The initial failed run and the passing fix are
listed in [the ledger](README.md).

## Earlier sequence through submission 4

1. The FP8-fix check is complete and retained in `results/fp8-fix/`. Keep it
   as the measured control. Do not spend a duplicate check merely to validate
   a source-only move whose definition hashes are identical.
2. The score reconstruction matches the reported speedup. Read the retained
   analysis/profile before building an experiment; the larger full-attention
   shapes regress while sliding verification is dominated by experts.
3. Prioritize a full-attention ablation against SGLang's FA3 path, keeping the
   custom sliding path unchanged. Use the same cache-write fusion and a
   clearly specified allowed query precision/softmax scale. The baseline
   already provides the correct handling of scheduler metadata, scaled KV
   caches, causal verify, and page tables. This separates custom full-attention
   cost from the rest of the layer before a further custom attention rewrite.
   This check is complete. Larger full shapes improved by 8.69–22.09% in
   normalized latency; unchanged sliding cases vary by about 2% or less.
   Keep the retained `results/full-fa3/` candidate as the new control.
4. Then investigate Marlin block-size policy on the sliding verification
   workload. Profile costs are 386/491 us at verify 16/32, about 72% of summed
   kernel time; keep alignment, both GEMMs, and finalization identifiable.
   A larger expert rewrite needs its own correctness and profiling plan.
   The next candidate is frozen at `build/moe-m16/candidate.py`; it caps the
   existing selector at 16 instead of 48/64 for the scored larger verify
   workloads. See [the experiment record](moe_m16_experiment.md).
5. Keep a control, define
   the affected shapes in advance, and change one factor in those shapes.
   Do not combine attention and expert rewrites before attributing their
   individual effects.
6. Compare normalized per-shape latency and aggregate confidence intervals.
   Confirm small gains, check nearby shapes, and promote only supported
   changes. Update the ledger and plan around the measured result.

## Candidate experiments, conditional on the profile

| Observed cost | Bounded next experiment | What must remain fixed | Main risk |
|---|---|---|---|
| Full-context attention | Vary split count for decode and verify independently; then consider KV tile size in a separate check | Every visible key/value, query precision, online-softmax math, cache scales | More partitions increase merge work and scratch traffic |
| Sliding attention | Compare fewer splits against the current policy; later test a fused single-partition finish if supported by timings | Full causal 2048-token window, all verify queries, scattered-page indexing | Small grids may underuse the GPU; extra launches may dominate |
| Single-token projections | Bounded GEMV warp/rows sweep, retaining the full-K implementation as control | BF16 weights/activations and FP32 accumulation | Register pressure and weight traffic can erase occupancy gains |
| Small QKV/pooler | Isolate launch configuration from arithmetic orientation; compare a bounded split/stage sweep to the current dispatch | Projection dimensions, preparation layout, FP32 partials | Projection-subproblem gains may disappear under layer scheduling |
| Expert alignment/Marlin | Profile alignment, both GEMMs, activation, and finalize separately; first test block-size policy alone | NVFP4 representation, actual local routes, post-expert norm and top-k weights | Sparse/skewed routing changes work per block; a policy win can be shape-specific |
| Router critical path | Measure the current hi/lo path and overlap before considering one fused producer or fused selection | FP32 router accuracy or allowed BF16 hi/lo accumulation, all 256 scores | A faster isolated router can still contend with the pooler |
| Boundary launches | Test a specific consumer fusion only after its profile cost justifies it | BF16 rounding points, FP32 residual stream, norm epsilons, sliced unpool behavior | Changed rounding or fusion may shift routing and downstream error |

The table gives follow-ups to the measured priorities above. At most one
group should change in the next timing experiment unless its scored shapes
are disjoint and the source changes are independently identifiable.

## Validation coverage

The judge's thirteen correctness cases exercise short contexts, scattered
page-size-1 and page-size-64 caches, mixed request lengths, long contexts,
causal eight-token verification, outlier activations, nonunit KV scales,
soft keys, and nontrivial expert placement. Retain all of them. Nearby
checks should also cover odd batches/token counts and the 8/9 and 32/33
projection/unpool dispatch boundaries. No host reads or data-value-dependent
Python branches may enter the timed CUDA graph.

When a matching compiler is available, adapt `bf16_proj/tools/codegen.py`
to compile the attention/cache/projection/boundary signatures for Hopper
before spending checks. Use actual FP8 pointer types and both query lengths;
the first failed check demonstrates why BF16-only compile probes are
insufficient. Compiler success and register counts still do not prove runtime
correctness or performance. The current local Python has no Torch/Triton,
so no GPU or offline Triton compilation has been performed here.

## Payoff calculation

For each load, sum each shape's time times its layer count (46 sliding,
16 full). The score uses the geometric mean of the six load speedups. A
shape's local latency sensitivity is its fraction of its load divided by six.
Use that fraction to value experiments; the longest individual kernel is not
automatically the highest score opportunity.

`tools.analyze` reports a conditional score if one shape becomes 20% faster.
It does not predict that gain or attribute it to a kernel. If a passing score
is `s` us, a sufficient uniform latency reduction to reach the reported target
is `1 - 9580 / s`. For the current 21184.86 us control, this is 54.78%.
The sliding shapes carry about 65.95% of local score sensitivity and full
shapes 34.05%, so improving full attention alone is not the complete plan.
