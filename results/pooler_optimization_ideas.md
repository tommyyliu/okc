# Pooler optimization ideas, ranked

Baseline: latest weight-major nearby check `67a1b145d83b4f89`, score
3934.42 us. Pooler contributes **33.97% of score-weighted latency sensitivity**
across its six cases. The 3000 us comparison target is user supplied and has
not been verified against a live leaderboard. Entries rank likely useful work
by score opportunity, evidence, and experiment cost, not by promised speedup.

| Rank | Idea | Why it ranks here | Evidence and next experiment |
|---:|---|---|---|
| 1 | Tune the proven small-token weight-major kernel by size | Three pooler cases contribute about 16.3% of total sensitivity. A hypothetical 20% reduction in these cases saves 131 score us. We have a promising operand-order result on these shapes. | Sweep one axis at a time from weight-major: BK 64/128/256, stages 2/3/4, BN 32/64/128, S 1/2/4/8, and warp count. Keep the cuBLAS and measured candidates in every comparison. Rerun a fresh control to separate gains from run variance. |
| 2 | Build a conventional streaming GEMM for M=128/256 and compete with cuBLAS | These are pooler’s longest cases: 25.13 and 31.45 us, near or slower than same-run SGLang, with reported floors of 14.64 and 17.37 us. A 20% reduction in just these two saves about 86 score us; their combined floor scenario saves 192 us. | Try the existing non-TMA tensor-core path with fewer output rows/features per CTA and then adjust BK, stages, and split count. Aim for enough independent CTAs without rereading too much of the 67 MB weight. The previous TMA versions at these sizes regressed 22–32%; a plain streaming design is a different test, but has no measured win yet. |
| 3 | Tune the M=1 pooler GEMV | This one case alone contributes 6.92% sensitivity. The current path is a 4096-row full-K reduction at 18.35 us, versus a 13.99 us floor. A hypothetical 20% reduction saves about 56 score us. | `tools.tune` already tests 1/2/4 output rows per program and 4/8 warps for M=1. Compare these graph timings before changing the kernel. If the tuner has not already covered pooler M=1 in this environment, include it in the next local GPU sweep. |
| 4 | Remove or reduce split-K and its reducer on small pooler | The reducer takes roughly 1.2–1.5 us in the separate profile, while the producer accounts for over 92% of profiled time. A reducer-only optimization has limited upside, but lowering S could remove its launch and scratch traffic. | Compare S=1/2/4/8 with BN and BM chosen to keep useful parallelism. S=1 has fewer CTAs and a longer K loop; pair it with narrower output tiles. Measure producer plus reducer together. Do not infer a score gain from producer-only profiles. |
| 5 | Repack pooler weights once in `prepare()` and compare load layouts | Preparation is untimed, and exact BF16 repacking is allowed. This gives a clean way to test whether the current weight layout makes tile loads inefficient, especially for the large cases. | Make one contiguous transposed or tiled BF16 copy, add a kernel that consumes that layout, and compare at identical output tile shapes. Track preparation memory and ensure the copy is bit-preserving. Retain the original-layout kernel as the control. |
| 6 | Change CTA tile order for large-token pooler | Triton’s GEMM guide explains how grouped CTA ordering can improve L2 reuse by reusing input tiles across adjacent output tiles. This may matter at M=128/256, where current GEMM implementations split output over multiple token tiles. | Record the existing order first: `_stream_gemm` maps M tiles inside each N tile. Try a small set of output-tile orders only if profiling shows L2 or HBM inefficiency; compare cold-L2 graph results because each scored replay evicts L2. The current mapping already provides some same-weight reuse, so upside is uncertain. |
| 7 | Try a persistent or asynchronous Hopper pipeline | It could reduce scheduling and memory-wait overhead in the long-K loop. It is more implementation work and can trade shared-memory use or occupancy for overlap. | First collect compiler resource data and GPU bottleneck counters for a winning conventional kernel. Triton’s persistent-matmul tutorial is a useful implementation reference, but it does not establish that this pooler shape benefits. |
| 8 | Revisit TMA only after a different design or diagnostic result | The tested 64×64×128 TMA kernel lost at M=128 and 256. Raising split-K also lost, with large scratch writes and a reducer. Repeating those variants is low promise. | Only return to TMA after identifying a concrete problem in the producer profile or changing tile shape, data layout, or pipeline. |

## Recommended experiment sequence

Three one-factor candidates are ready under `build/pooler-round1/`:

| Candidate | Changes | Cases |
|---|---|---:|
| `pooler-small-split2.py` | Weight-major pooler M=8/16/32: split count 4 → 2 | 3 |
| `pooler-gemv-rows2.py` | Pooler M=1: output rows per program 1 → 2 | 1 |
| `pooler-large-streaming.py` | Pooler M=128/256: non-TMA weight-major kernel vs cuBLAS | 2 |

The most recent check reported 14 daily checks remaining; a live quota lookup
failed because the OKC host was unreachable. Treat that count as a snapshot.
Start with `pooler-large-streaming.py` because it probes the largest unexplored
pooler opportunity. Each `okc check` still measures all thirty scored shapes,
but only two pooler dispatches change. Compare their per-case candidate/SGLang
ratios with the latest base run; keep or discard those settings before spending
a check on the other files. If reducing split-K helps small pooler, follow with
BK=256 on those same three cases. If the large streaming path fails, use the
profile evidence before choosing a new tile family. Do not run all candidates
just to fill the budget.

First check:

```bash
okc check build/pooler-round1/pooler-large-streaming.py --json \
  | tee build/pooler-round1/pooler-large-streaming-check.json
```

Then choose one follow-up from the per-case results: small pooler split-K or
M=1 GEMV. Candidate scope and the base result ID are in
`build/pooler-round1/manifest.json`.

The initial opportunity sequence is M=128/256 streaming first, then M=1 GEMV
and small-token split/tile tuning. If a candidate shows potential, inspect
weight bandwidth, CTA count, registers, and shared-memory use before a larger
scheduling or asynchronous design. Test weight repacking as a separate factor.
Keep settings per token count and promote only repeatable same-run normalized
wins.

An optimistic whole-pooler-at-floor calculation takes the score to about
3519 us, still above 3000 us. The floor is not achievable-performance evidence.
It does show why a pooler-only change cannot be the entire plan for that target.

Triton documents grouped tile ordering as an L2-reuse optimization in its
[matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
and persistent scheduling in its
[persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html).
NVIDIA’s [CUDA best-practices guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
explains that occupancy can hide stalls, while more occupancy by itself does
not guarantee faster execution. These are design references, not evidence of a
pooler speedup on H200.
