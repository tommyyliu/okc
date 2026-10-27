# October 26 follow-up: five bounded checks

All five candidates passed **nine correctness tests and thirty timed cases**.
None improved on the previous selected candidate's **3764.88 us / 1.2831221x**.
Retain [that exact source](round7_selected_source.py) as the experimental
reference. No new setting was promoted, and PDL remains the historical default.

The five remaining daily checks are now used. The final quota response reports
**0 checks and 10 scored submissions remaining**. No scored submission was made.

## Results against the selected reference

| Trial | Job | Score, us | Speedup | Score change |
|---|---|---:|---:|---:|
| TMA, two stages | `99492464985742e0` | 3933.79 | 1.2280256x | +4.49% |
| TMA, four stages | `72ee9e56e9c74c82` | 3799.77 | 1.2713403x | +0.93% |
| Wider TMA geometry | `76820b2557f04afc` | 3836.20 | 1.2592679x | +1.89% |
| Earlier producer signals | `edb8c83cf4f74dfa` | 3825.61 | 1.2627539x | +1.61% |
| Router warp counts | `a7d02a5dc3194517` | 3819.72 | 1.2646999x | +1.46% |

Lower scores are better. Raw results, readable profiles, resource logs and local
compiler reports are retained as `round8_<trial>.*`.
[The manifest](round8_manifest.json) records sources and hashes;
[the analysis](round8_analysis.json) includes all thirty cases and unchanged-case
variation for every run.

Per-case changes below use `reference.paired_ratio / trial.paired_ratio - 1`
against each run's own SGLang baseline. They are cross-run comparisons, not
direct paired A/B timings between our candidates. Small differences and
aggregate point estimates remain subject to run variance.

## TMA stages and geometry

The two-stage and four-stage trials keep BM/BN/BK and K splits fixed across the
six small QKV/pooler cases. They use the same TMA kernel as the prior trial.
The selected reference already uses TMA on QKV M32; its other five cases use
weight-major GEMM. QKV M32 therefore isolates pipeline depth against the
selected reference, while the other five comparisons include the operand-path
change. Comparing the two new TMA trials isolates their pipeline depth.

| Case | Two stages | Four stages | Wider geometry, three stages |
|---|---:|---:|---:|
| QKV M8 | +11.1% | +14.2% | +1.0% |
| QKV M16 | +11.8% | +6.5% | +2.0% |
| QKV M32 | +20.7% | +15.1% | +11.7% |
| Pooler M8 | +16.8% | +1.8% | +4.6% |
| Pooler M16 | +17.4% | +0.7% | +6.2% |
| Pooler M32 | +17.3% | +1.2% | +6.0% |

The reference's **three stages remain the best tested setting on QKV M32**.
Two stages reduced shared memory from 73752 to 49168 bytes for that case,
with 42 registers and no reported spills, but lost substantially. Four stages
used 98336 shared bytes and also lost. Lower shared-memory use alone did not
improve the schedule. Unchanged-case median movement was +0.09% and -0.64%
for the two-stage and four-stage runs, respectively.

The wider geometry doubles BN64 to BN128, halves BK128 to BK64, and doubles
the K split count. This preserves producer CTA count, weight bytes per CTA,
K-loop iteration count and useful matrix arithmetic per CTA. It halves logical
activation bytes per CTA but doubles FP32 partial-buffer traffic. These are
coupled geometry changes, not a one-field ablation. The new kernel used
55320/61464 shared bytes at BM16/BM32, 42/60 registers and no reported spills.
It passed the nearby pooler M7 test, but the scored cases supplied no clear gain.

## Earlier reduction scheduling

The `round8_early` preset changes twelve split-K producer entries. Thin JIT
wrappers issue a PDL signal before the existing producer math; the original
producers and their later signals remain unchanged. Producer grids, tiles,
precision, allocation and reduction launches match the selected reference.

The reducer still calls `gdc_wait` before its first global load. The intent is
to overlap consumer scheduling, not to make partial results readable early.
NVIDIA documents that the trigger enables dependent scheduling, while the
dependent kernel must separately wait for producer completion and visibility.
Only the first trigger from each CTA is required for scheduling eligibility.
See [NVIDIA's PDL description](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

Local assembly audits verify the initial signal precedes tensor-core math,
the producer uses a normal launch, and the consumer retains PDL and a wait
before memory reads. All required GPU tests passed, including router M9
outliers. The twelve changed cases ranged from -1.6% to +5.9%; unchanged cases
had median +1.04%. Small apparent wins at router M32/M256 and unpooler M8 are
insufficient evidence to select them. Keep the existing signal placement.

## Router launch geometry

The last check increases router M1 from sixteen to thirty-two warps and changes
only the three B32 reducers from four warps to one. Producer math and producer
launches on M8/M16/M32 remain identical to the reference.

| Router M | Reference, us | Trial, us | Normalized latency change |
|---|---:|---:|---:|
| 1 | 3.805 | 4.058 | +5.6% |
| 8 | 5.569 | 5.612 | +0.2% |
| 16 | 5.674 | 5.841 | +3.7% |
| 32 | 6.327 | 6.247 | -0.1% |

Thirty-two-warps GEMV compiled with 27 registers, 16384 shared bytes and no
spills. The one-warp reducer used 40 registers and 128 shared bytes, compared
with the old four-warp reducer's 19 registers and 512 shared bytes. The physical
resource reduction did not establish an end-to-end gain. Unchanged-case median
movement was +1.57%. Keep **sixteen warps for M1 and four reducer warps**.

## Validation and staged next check

All **54 CPU tests pass**. Tests cover unchanged original GPU bodies, every
preset's compiler-manifest/launch agreement, current descriptor bindings and
the early-signal wrapper's identical grids, storage and consumer launches.
Every new GPU configuration compiled locally with pinned Triton **3.8.0**
before submission. The generated PDL file is current. GPU correctness is
established by the five complete remote results, not by CPU checks or AOT
compilation. `git diff --check` passes.

One next candidate is staged without consuming a check:
`build/next-qkv-split4/candidate.py`. It changes **only QKV M32 from S8 to S4**,
retaining BM32/BN64/BK128, four warps and three stages. Nearby QKV M17..32 uses
the same S4 setting. Other scored dispatch is identical to the exact round-seven
source. The manifest labels it **unmeasured**. Its source preset is `qkv_split4`;
the retained [plan](next_qkv_split4_plan.json) and
[compiler audit](next_qkv_split4_codegen.json) permit regeneration and review.

The hypothesis is that a longer TMA loop and half the partial traffic can repay
the reduction from 288 to 144 producer CTAs. Prior non-TMA reduced-split losses
do not establish this TMA configuration's performance. Local compilation and
PDL ordering audits passed at M17/M31/M32, without spills; GPU correctness and
timing remain pending.

When the daily quota resets at 00:00 UTC, the prepared command is:

```bash
python -m tools.build --preset qkv_split4 \
  --output build/next-qkv-split4/candidate.py
okc check build/next-qkv-split4/candidate.py --json \
  > build/next-qkv-split4/check.json
```
