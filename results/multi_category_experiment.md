# Multi-category check: per-case findings

Job `bb31d5bd92fb41e1` ran candidate
`build/multi-category-round1/combined.py` on an H200. All nine correctness
tests and all thirty scored cases passed, but its aggregate was poor:
**0.97394x, 4960.05 us**, 95% interval [0.95933x, 0.98855x]. The candidate is
a broad experiment, not a viable submission. Raw response and profile are
retained as `multi_category_round1.json/txt`.

Compare each changed case's candidate/SGLang ratio with the same ratio in base
job `67a1b145d83b4f89`; the runs are on different nodes, so these normalized
comparisons still carry run variance.

| Idea | Cases | Normalized latency change vs base | Decision |
|---|---|---:|---|
| Pooler GEMV rows 1 → 2 | M=1 | 18% slower | Reject rows=2. Keep rows=1. |
| Pooler split count 4 → 2 | M=8/16/32 | 16–18% slower | Reject. The shorter reducer does not repay the slower producer. |
| Pooler non-TMA streaming | M=128/256 | 99% / 154% slower | Reject this tile/configuration. Candidate took 50.0 / 81.5 us vs base 25.1 / 31.4 us. |
| O weight-major orientation | M=8/16/32 | 7–10% faster | Keep as a strong per-case winner. It also won in the earlier orientation run. |
| QKV non-TMA streaming | M=128/256 | 150% / 221% slower | Reject this tile/configuration. Candidate took 42.8 / 66.1 us vs base 17.3 / 20.0 us. |
| Unpooler orientation | M=16/32 | 2% faster | Too small to call from one cross-run comparison. |
| Router split count, halved | M=128 | 5% faster | Promising; profile producer/reducer fell from about 4.8/2.1 to 4.4/1.6 us. Confirm alone. |
| Router split count, halved | M=256 | 13% slower | Reject the S=8 setting; return to S=16. |

The fifteen unchanged cases had a median normalized shift of -0.7%, ranging
from 7.2% slower to 5.5% faster. That variation limits fine-grained conclusions
from cross-run differences. O's repeatable 7–10% gains are larger; the router
M=128 signal merits one isolated check.

Both large streaming candidates spent all profiled GPU time inside
`_weight_major_stream_gemm`. They avoided the reducer but ran far slower than
cuBLAS. Do not repeat these configurations; a new large-pooler or QKV path needs
profiling and a materially different schedule or layout. For the small pooler,
splitting K less also lost: the producer became slower while the reducer was
only a small share.

The successful check used seven of twenty daily checks, leaving thirteen in its
result. An earlier attempt had failed at DNS resolution; the job above completed
successfully.

Two isolated candidates are prepared from the passing base:

1. `build/multi-category-round1/o-orientation-only.py` applies the O orientation
   win at M=8/16/32. This is the better next check because it repeats the
   clearest cross-run signal.
2. `build/multi-category-round1/router-M128-split16-only.py` tests the router
   M=128 improvement independently while leaving M=256 at its prior setting.

The hypothetical O-only score, about 3915 us versus the base 3934 us, applies
cross-run per-case ratios to the base run. It is an estimate, not a measured
score. `multi_category_round1_analysis.json` retains that calculation and all
per-case values. Do not use the combined 4960 us candidate as a new control.

## Second combined candidate

`build/multi-category-round2/candidate.py` preserves the successful small-QKV
base, repeats O orientation and router M=128 as anchors, then changes BK from
128 to 256 in six small weight-major GEMMs. It also tries non-TMA weight-major
large pooler/QKV kernels with BM64, BN64, BK256, and S=2. The longer-K,
split-K configuration is distinct from the very slow, unsplit BK128 kernel
tested above.

The candidate changes fourteen scored cases across the four ideas in
`build/multi-category-round2/manifest.json`; its compile plan has 40 kernel and
reducer jobs. Its first check
attempt also failed before receiving a job ID because `openkernelchallenge.dev`
did not resolve. The subsequent check completed as `9515208aace4455e`:
all nine correctness tests and thirty timed cases passed, at **4450.49 us /
1.08545x**, speedup interval [1.06917x, 1.10173x]. Raw evidence is retained as
`multi_category_round2.json/txt`.

Relative to base `67a1b145d83b4f89`, the aggregate score regressed about 13.1%.
Using each run's official per-case median paired speedup against SGLang:

| Idea | Cases | Normalized latency change vs base | Decision |
|---|---|---:|---|
| BK128 to BK256, small QKV | M=8/16/32 | 21% / 17% / 27% slower | Reject these settings. |
| BK128 to BK256, small pooler | M=8/16/32 | 9% / 10% / 13% slower | Reject these settings. |
| Non-TMA BM64/BN64/BK256/S2 QKV | M=128/256 | 59% / 98% slower | Reject this configuration. |
| Non-TMA BM64/BN64/BK256/S2 pooler | M=128/256 | 24% / 76% slower | Reject this configuration. |
| O orientation | M=8/16/32 | 10% / 11% / 10% faster | Repeatable winner. Carry forward. |
| Router split32 to split16 | M=128 | 6% faster | Repeated promising result. Carry forward. |

Router M=256 retained its base split16 setting in this round. Its roughly 3%
normalized movement is an unchanged-case observation, not a new optimization.
Retaining only the O and router M=128 changes projects about **3905.9 us**,
28.5 us below base. This is a cross-run estimate, not a measured combined score.
The earlier experiment tables used ratios of medians; the round-two table above
uses official median paired ratios as specified by the task.

[Round three](multi_category_round3_experiment.md) tests prepared weight layout,
router tiles that preserve CTA count while reducing splits, and chunked GEMV.
