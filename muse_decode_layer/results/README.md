# Decode evidence ledger

| Candidate | Evidence | Status | Performance conclusion |
|---|---|---|---|
| Initial fused decode | [`submission1.json`](../submission1.json), job `918336d7e75f4911`; [log](../profiles/submission.918336d7e75f4911.log) | Preparation passed; all 13 tests failed at the same attention compilation error | No timing or speedup |
| FP8 mask fix | [`submission2.json`](../submission2.json), job `71b479393ecd4532`; [retained analysis](fp8-fix/analysis.json) | All 13 correctness tests and 12 benchmarks passed | 22056.34 us / 1.0255689x; 95% interval [1.0101854, 1.0409524] |
| Source organization | `decode/`, `tools/`, `tests/fixtures/fp8_fix.json` | Eight CPU checks pass; exact AST equivalence to the FP8-fixed candidate | Organization only; no claimed speedup |
| Full FA3 experiment | [Experiment record](full_fa3_experiment.md); [retained evidence](full-fa3/check.json), job `2524b1c4986a49b4` | All 13 correctness tests and 12 benchmarks passed | 21184.86 us / 1.0677577x; 95% interval [1.0517413, 1.0837740]; current default |
| Expert block cap 16 | [Experiment record](moe_m16_experiment.md); [retained evidence](moe-m16/check.json), job `f2293cc26af04ec1` | All 13 tests and 12 benchmarks passed | 20264.20 us / 1.1162692x; interval [1.0995252, 1.1330132]; current default policy |
| Small expert finalize | [Experiment record](small_finalize_experiment.md); [retained evidence](small-finalize/check.json), job `418dc106c212475a` | All 13 tests and 12 benchmarks passed | 19046.05 us / 1.1876636x; interval [1.1698486, 1.2054785]; current default |
| Expert block cap 8 | [Experiment record](moe_m8_experiment.md); [retained response](moe-m8/check.json), job `7844a7a3dbef43b6` | All 13 tests and 12 benchmarks passed | 20511.35 us / 1.1028188x; rejected, 7.69% slower |
| Direct single-token NVFP4 | [Experiment record](single_nvfp4_experiment.md); `build/single-nvfp4/candidate.py` | Twenty CPU checks pass; GPU check pending | No measured result yet |

The first failure was Triton's inability to cast the integer mask-fill tensor
directly to FP8 E4M3. It occurred in `_attention_parts` while loading K;
the V load had the same issue and was corrected too. Static submission checks
accepted the initial candidate. That acceptance does not establish GPU
compilation, numerical correctness, or timing.

The fixed, pre-refactor source is retained as
[`fp8-fix/candidate.py`](fp8-fix/candidate.py); its SHA256 is in the AST checkpoint.
The generated export has different formatting/import order, while preserving
all original definitions and constants. Check 2 measures the pre-refactor fixed
candidate. Do not infer a separate performance improvement from the refactor.

The projection subproblem's historical wins informed the initial dispatch,
but they do not establish a measured benefit in this whole-layer workload.
The whole-layer expert and attention costs, routing, cache access, and overlap
need their own passing measurements.

The corrected user-reported leaderboard reference is **9580 us / 2.36x**.
The passing run's inferred SGLang baseline is 22620.30 us, consistent with that
reference. Reaching 9580 us requires **56.57% less aggregate latency** from this
original control. From the current FA3 default, the remaining reduction is
**54.78%**. No live leaderboard query has been made. Follow [plan.md](plan.md).

## Submission 6 and current priority

Block cap 8 passed correctness but regressed to 20511.35 us / 1.1028188x,
7.69% slower than submission 5. Large verify normalized latency regressed
13.11–30.56%; decode 32 sliding also regressed 8.96%. Reject the cap and
retain the 19046.05 us control. Do not keep sweeping smaller Marlin blocks.

The next experiment starts a direct NVFP4 expert implementation on the
single-token workload: no alignment, fused gate/up plus SwiGLU, a direct
second projection, then the existing masked finalize. Larger token counts
retain Marlin. This is a scoped first kernel design, not an expected solution
to the remaining 49.70% gap. See [the record](single_nvfp4_experiment.md).

## Earlier submission 5 result

Fused small-token finalize passed at 19046.05 us / 1.1876636x, reducing the
reported score by 6.01% from submission 4. Changed small-token shapes improve
by 8.82–16.87% in candidate/SGLang latency ratios. The new kernel takes about
3.0–3.3 us in the profiles. Larger unchanged verify cases vary 3.24–7.58%
worse, so not every cross-run change is attributable to this kernel. The
aggregate speedup intervals nevertheless do not overlap.

Keep the fused finalize. The next candidate changes only Marlin's block-size
choices to (8,), retaining all kernel math. The default remains the exact
passing submission 5. The remaining reduction to 9580 us is 49.70%.

## Earlier submission 4 result

Block cap 16 reduces the score another 4.35%. Large verify normalized latency
improves by 16.15%/14.84% on sliding batch 16/32 and 12.16%/9.75% on full
batch 16/32. Unchanged cases vary from -0.74% to +2.67%. The speedup intervals
do not overlap. The current gap to 9580 us is 52.72% less latency.

The next experiment fuses small-token expert normalization and weighted sum,
retaining all routes and masking uninitialized remote slots before loading.
Larger T retains the library finalize. See [the record](small_finalize_experiment.md).

## Earlier FA3 result and priority


Full decode 16/32 normalized latency improved by 8.69%/11.20%; full verify
16/32 improved by 19.17%/22.09%. Full decode 1 regressed by 3.04%, and full
verify 1 improved by 2.05%. Unchanged sliding shapes moved between -1.04%
and +1.89%. These are comparisons of candidate/SGLang ratios within each run,
not raw profile times. The aggregate speedup intervals do not overlap.

The FA3 default reduces the reported score by 3.95%. Expert GEMMs still take
69.2%/74.3% of summed profiled kernel time on sliding verify 16/32. All sliding
shapes together carry 65.95% of total score sensitivity, so expert work is
the next focus. The bounded block-size
experiment is ready; the original Marlin implementation and all other stages
are unchanged. See [the retained FA3 analysis](full-fa3/analysis.json).

## What the passing control shows

Sliding layers beat same-run SGLang by 1.01–1.18x. Full layers at batches
16/32 lose at 0.79–0.93x. Their `_attention_parts` profile time ranges from
227.5 to 663.8 us and accounts for 40–70% of the summed profiled kernel time.
Marlin accounts for about 72% of that time in sliding verify 16/32. These
profiles support attention and expert work as the next main areas; they do
not support spending the first experiments on another broad projection rewrite.

The twelve shape sensitivities sum to one. Sliding shapes account for about
63.2% of score sensitivity, with full shapes about 36.8%; the 46/16 layer
weights matter. Profile shares measure a different quantity and are subject
to overlap. See the [raw profile](fp8-fix/profile.txt),
[response](fp8-fix/check.json), and [analysis](fp8-fix/analysis.json).
