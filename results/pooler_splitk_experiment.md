# Pooler split-K scheduling experiment — correctness passed, still slower

Check `a63715f545fc478d` passed all nine preliminary tests and thirty scored
cases at 1.1842914x, 95% interval [1.1665271, 1.2020558]. Pooler M=128 took
28.71 us against same-run SGLang's 24.82 us; M=256 took 45.47 us against
31.39 us. The separate profiler measured producer/reducer times of 28.2/3.4 us
and 43.0/4.3 us respectively. Neither configuration should replace cuBLAS.
See `pooler_splitk.json` and `pooler_splitk.txt` for retained evidence.

## Original experiment design

The first TMA pooler experiment lost by 32.4% and 22.0% against same-run SGLang
at M=128 and M=256. This follow-up tests more independent K partitions before
changing the kernel layout or abandoning TMA. It is a scheduling hypothesis,
not a claim that the profile established an occupancy bottleneck.

| Setting | Failed TMA | New M=128 | New M=256 |
|---|---:|---:|---:|
| K partitions | 1 | 4 | 2 |
| Producer programs | 128 / 256 | 512 | 512 |
| K-loop iterations per program | 64 | 16 | 32 |
| FP32 partial buffer | None | 8 MiB | 8 MiB |

Both cases retain BM=64, BN=64, BK=128, four warps, and three stages. The existing
PDL reducer combines FP32 partials before storing BF16 output. This adds one
reduction launch and 16 MiB of partial write/read traffic per scored call. A
win requires the shorter independent reductions to pay for that added work.

Only pooler at 33–256 tokens changes from the PDL default. Nearby token counts
use the same rule: increase split-K until the producer grid has at least 512
programs, capped at eight partitions. The scored settings are explicit in
`projection/configs.py`. Kernels, prepare, and precision are unchanged.

```bash
python -m tools.build --preset pooler_splitk --output build/pooler_splitk.py
okc check build/pooler_splitk.py --json | tee build/pooler_splitk-check.json
```

Inspect both changed cases against their same-run SGLang baselines. Also inspect
the producer and reducer profile separately. A faster producer with a slower
total would show that reduction overhead erases the scheduling benefit. If
total time remains materially behind SGLang at both sizes, stop this sequence
of pooler variants and move to another kernel family or obtain detailed GPU
profiling before further TMA tuning.

CPU dispatch/signature tests pass for scored and nearby sizes. The subsequent
GPU check above validates the scored cases; local nearby GPU checks were not run.
