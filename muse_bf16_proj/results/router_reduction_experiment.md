# Small-router reduction geometry check

Status: check **3f90adf02bb14808 passed**, including all nine correctness tests
and thirty scored cases. Measured **3848.12 us**, **1.25537x** (95% interval
1.23654–1.27420x). The completed result was found in the workspace at handoff;
no additional check was launched. PDL remains the default; confirmation pending.

**Corrected control comparison:** the earlier keepers check, job
`13ae262ccdfb40bf`, was found in `build/multi-category-round3/keepers-check.jso`.
It measured **3838.82 us**, so the new aggregate is **9.30 us / 0.24% slower**.
The original comparison below used round six and missed this measured control.
Against keepers, normalized M8/M16/M32 router gains are **5.7%/4.1%/6.0%**.
The remaining difference includes four changed GEMV warp counts and timing
movement in 23 unchanged configurations. Used GPU kernel bodies and original
weight preparation are identical; no selected implementation was dropped.
The 0.24% score difference is within reported uncertainty.

On the new run's baselines, a sequential counterfactual calculation assigns
about -11 us to the three router gains, -18 us to the four GEMV changes in
aggregate, +42 us to unchanged-case ratio movement, and -4 us to baseline
distribution movement, leaving the measured +9 us difference. These are
order-dependent cross-run calculations, not proof of causal regressions.
Largest unchanged normalized slowdowns include QKV M8 (+5.3%), QKV M32 (+4.0%)
and router M128 (+5.4%). See [measured-control comparison](router_reduction_vs_keepers.json)
and [retained keepers result](keepers.json). Confirm against the staged banked
control before promoting this change.

M8/M16/M32 router times are **5.51/5.97/6.29 us**, respectively **11.9%/9.4%/6.3%
lower normalized latency** than round six. These are cross-run comparisons.
The 23 configurations matching round six vary by a median +0.72%, range
-1.76% to +6.61%. The reducer profile falls from roughly 1.6/1.7/1.8 us to
1.1/1.2/1.3 us. The logged B32 reducer uses 19 registers and 512 shared bytes,
versus the prior B256 record's 56 registers and 2048 bytes; both report zero
spills. This supports a reducer-geometry contribution without proving occupancy.

Retained evidence: [full result](router_reduction.json),
[profile](router_reduction.txt), [resource log](router_reduction.log),
[comparison calculations](router_reduction_analysis.json).

Run from `muse_bf16_proj/`:

```bash
okc check build/router-reduction/candidate.py --json \
  | tee build/router-reduction/check.json
```

This complete projection candidate tests whether exposing more independent
reducer blocks reduces small-router latency. It follows the
[first-principles diagnosis](first_principles_analysis.md).

## Change and hypothesis

Relative to the selected `banked` configuration, only router M8/M16/M32 reducer
geometry changes. Each reducer block handles **32 output values instead of
256**. Four warps per block, split-K count, producer launches, FP32 scratch
layout, FP32 accumulation, BF16 hi/lo operands, preparation and PDL synchronization
retain their previous settings.

| Router M | Producer blocks | K splits | Old reducer blocks | New reducer blocks | Partial storage |
|---|---:|---:|---:|---:|---:|
| 8 | 128 | 32 | 8 | 64 | 256 KiB |
| 16 | 128 | 32 | 16 | 128 | 512 KiB |
| 32 | 128 | 32 | 32 | 256 | 1024 KiB |

The producer and reducer GPU source bodies are unchanged. Specializing the same
reducer for B32 changes its tensor shape, compiled layout/resource use and grid.
The candidate emits existing scalar compiler-resource diagnostics once per
resource signature; these are not performance counters.

Nearby router sizes `1 < M <= 32` use the B32 policy, including the M9 outlier
correctness test. Router M1 and M>32, and every other projection, retain the
selected configuration. All thirty scored producer choices are preserved.
PDL remains the default; `router_reduce` is an experimental preset.

The prior profiles attributed approximately 1.6/1.7/1.8 us to the reducer at
M8/M16/M32, or 37–40% of summed kernel time. B32 exposes eight times as many
blocks and reduces work per block without changing scratch volume or adding
a launch. Extra block scheduling might instead outweigh the shorter blocks.

If reducer and complete-call times improve together, that supports reducer
geometry as a useful contributor. A faster reducer with flat complete-call time
means the end-to-end dependency/timing path needs investigation. A flat/slower
reducer means a larger grid alone did not solve its cost. None of these outcomes
alone proves an occupancy or launch bottleneck.

The broad 4–5 us small-router goal is not a prediction for this change. This
probe retains both launches and their dependency. A clean negative result can
still narrow the next architectural experiment.

## Comparison

The matching router control is in the passing
[round-six result](multi_category_round6.txt), job `34425ce43f3e4f6c`:

| Router M | Previous scored latency | Previous SGLang / candidate ratio |
|---|---:|---:|
| 8 | 6.36 us | 4.154x |
| 16 | 6.59 us | 2.246x |
| 32 | 6.56 us | 2.289x |

Use exact retained `paired_ratio` fields, copied into the build manifest.
Cross-run normalized latency change is `previous_ratio / new_ratio - 1`.
Inspect unchanged cases for run variation; small cross-run differences are not
established gains. Compare producer/reducer profiles separately and preserve
the adjacent `profiles/submission.<job>.log` for compiler resource records.
Profile sums are not official paired score times.

Do not attribute aggregate improvement over round six to this reducer:
the candidate also returns its four unsuccessful large QKV/pooler paths to
selected settings. Its other choices retain selected GEMV8, O orientation,
weight-major small QKV/pooler and prepared pooler/unpooler transposes. The
selected aggregate of approximately **3854.64 us** on round six's baseline is
counterfactual, not measured. Judge this change from its three per-case results.

`build/router-reduction/control.py` is staged for a direct control check. The
candidate now has a passing GPU result; the control's aggregate is still unmeasured.

## Validation and reproduction

All **44 CPU tests passed**, including unchanged checkpoint GPU bodies and
launches, compile-manifest/launch agreement across every preset, three-case
scope, nearby dispatch, standalone source inspection, and reducer addresses
with odd tails and all 32 partials. These do not establish GPU correctness.

Local ahead-of-time compilation reported `No module named 'triton'`. Planned
jobs and that failure are in `build/router-reduction/codegen/report.json`.
The completed remote check subsequently established compilation and GPU
correctness. Additional nearby graph validation has not been run locally.

```bash
python -m tools.build --preset router_reduce --output build/router-reduction/candidate.py
python -m tools.build --preset banked --output build/router-reduction/control.py
python -m unittest discover -s tests -v
python -m tools.build --check

# Compiler audit when Triton is available; no GPU needed:
python -m tools.codegen --source build/router-reduction/candidate.py \
  --layers router --tokens 8 9 16 32 --output build/router-reduction/codegen

# Additional numerical/graph validation when a GPU runtime is available:
python -m tools.validate build/router-reduction/candidate.py --nearby
```

Source: `projection/configs.py` preset `router_reduce`; runtime and compiler audit
recognize `dot_reduce32`. Candidate/control SHA256 hashes, exact baseline ratios
and compile plans are in `build/router-reduction/manifest.json`.
