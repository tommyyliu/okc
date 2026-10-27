# Retained results

[Five additional checks](round8_experiment.md) tested TMA depths, wider geometry,
earlier PDL signaling and router warp counts. All passed correctness; none beat
the selected **3764.88 us** reference. Results are retained as `round8_*.json`,
`.txt`, `.log` and compiler reports. All 54 CPU tests pass. The daily quota now
has **0 checks and 10 scored submissions remaining**. A QKV M32 S4 follow-up
is compiled and staged for the next reset, with no GPU result yet.

[October 26 autonomous experiments](round7_experiment.md) supersede the pending
control, connectivity and cluster statuses in the historical notes below.
The measured banked control is **3830.40 us**. Both cluster variants and the
deeper WGMMA pipeline passed correctness and regressed. Six-case small TMA
measured **3824.02 us**, a difference within uncertainty. Doubling small-GEMM
splits lost. The selected confirmation passed at **3764.88 us / 1.28312x**,
retaining router M1 at sixteen warps, B32 small-router reducers and QKV M32 TMA.
Its aggregate interval overlaps the control's; PDL remains the default.

| Preset | Original job | Measured speedup | Status |
|---|---|---:|---|
| Selective | `027b6db0ef044348` | 1.2048418× | Historical control |
| Orientation | `8fbfde82af1a4761` | 1.2150730× | Historical O-projection experiment |
| PDL | `e6f6e1af0bd446a1` | 1.2310594× | Default source configuration |
| Combined | — | — | PDL plus O orientation; GPU check pending |
| TMA | — | — | Combined plus dense TMA projections; GPU check pending |
| Pooler TMA | `dc36c8ef363340e3` | 1.1739579× | Correctness passed; both changed cases regressed |
| Pooler split-K | `a63715f545fc478d` | 1.1842914× | Correctness passed; both changed cases still lose |
| PDL recheck | `706c118665f14447` | 1.2197188× | Fresh control; correctness passed |
| Small streaming | `829a3acc7d344f38` | 1.2132923× | Correctness passed; QKV M=32 alone gives no clear gain |
| Small weight-major | `5aa7a0e662a84bfd` | 1.2324506× | Correctness passed; six promising cases, confirmation pending |
| Weight-major nearby | `67a1b145d83b4f89` | 1.2278289× | Correctness passed, including new kernel at pooler M=7; 3934.42 us |
| Multi-category round 1 | `bb31d5bd92fb41e1` | 0.9739420× | Correctness passed; 4960.05 us, reject combined candidate |
| Multi-category round 2 | `9515208aace4455e` | 1.0854527× | Correctness passed; 4450.49 us, reject new BK256 settings |
| Multi-category round 3 | `327bb5b2fdc6423a` | 1.2274639× | Correctness passed; 3935.59 us; layout gains canceled by router/GEMV regressions |
| Multi-category round 4 | `0fe4f6662e1d458a` | 1.2435046× | Correctness passed; 3884.83 us; no clear broad gain from the four new groups |
| Multi-category round 5 | `6456b973b73a4a26` | 1.1178293× | Correctness passed; 4321.59 us; GEMV8 wins, unroll/direct router regress |
| Banked control, Oct 26 | `6ddee3c328e34835` | 1.2611733× | Correctness passed; 3830.40 us |
| Cluster BK256, fixed | `5e18ccab93f940f7` | 1.2137396× | Correctness passed; 3980.10 us, reject |
| Cluster BK64 | `e40c0396e29b49a0` | 1.1535114× | Correctness passed; 4187.91 us, reject |
| Small TMA | `0f311fd4ab4342af` | 1.2632767× | Correctness passed; 3824.02 us, no established broad gain |
| Hopper overlap | `c02532ce190544f5` | 1.0542127× | Correctness passed; 4582.38 us, reject |
| Occupancy probes | `ed6f332cdf94424b` | 1.2495731× | Correctness passed; 3865.96 us, retain only router M1 signal |
| Round 7 selected | `f5ab0410cf1e4449` | 1.2831221× | Correctness passed; 3764.88 us, experimental reference |
| Round 8 TMA stages 2 | `99492464985742e0` | 1.2280256× | Correctness passed; 3933.79 us, reject |
| Round 8 TMA stages 4 | `72ee9e56e9c74c82` | 1.2713403× | Correctness passed; 3799.77 us, no selected gain |
| Round 8 wider TMA | `76820b2557f04afc` | 1.2592679× | Correctness passed; 3836.20 us, reject |
| Round 8 earlier PDL | `edb8c83cf4f74dfa` | 1.2627539× | Correctness passed; 3825.61 us, no clear gain |
| Round 8 router warps | `a7d02a5dc3194517` | 1.2646999× | Correctness passed; 3819.72 us, retain previous geometry |

PDL's historical 95% interval is [1.21259, 1.24953]. These are check results,
not leaderboard submissions. The two small-GEMM results measure generated
files from the refactored source and passed nine correctness cases and thirty
timed shapes; older rows retain their historical source provenance.

`pdl.json` and `selective.json` contain complete check responses. Their `.txt`
files contain readable per-shape timings and profiles. Orientation is retained
as `orientation.profile.json` plus `orientation.txt`; the original complete
response was not available. Full earlier evidence is in `pre-reorganization`.

The recorded target is 1.61×, not a refreshed live leaderboard. From PDL this
requires 23.54% less aggregate latency. Pooler, QKV, and unpooler account for
approximately 78% of per-load latency, averaged over the six loads. Use
`python -m tools.budget` to reconstruct the score and floor scenarios.

The failed lossless implementation measured 0.1076×. Its evidence and decoder
diagnostics are archived in Git rather than kept in the active implementation.

The fresh control and per-case score sensitivity inform
[optimization_priorities.md](optimization_priorities.md). The 1.61x target
remains historical rather than a refreshed leaderboard query.

[Small-batch QKV/pooler](small_gemm_experiment.md) has passed its initial GPU
checks. Weight-major improves the normalized latency of all six target cases
versus streaming in one comparison; confirmation and new-kernel nearby checks
are pending. Raw responses/profiles are retained as `small_gemm_streaming.*`
and `small_gemm_weight_major.*`; `small_gemm_comparison.json` holds the analysis.
No settings from this experiment have been promoted into PDL.

[The latest opportunity analysis](opportunity_analysis.md) quantifies the gap
from the nearby check to the user's approximate 3000 us target. Its scenarios
are retained in `opportunity_analysis.json` and are not measured predictions.

[The combined multi-category check](multi_category_experiment.md) passed at
0.97394x with all correctness tests. Its per-case timings reject the tested
large streaming and reduced-split pooler settings, while confirming the O
orientation gains and suggesting a router M=128 split reduction. The raw result
and per-case analysis are retained as `multi_category_round1.json/txt` and
`multi_category_round1_analysis.json`.

The second candidate completed after the initial DNS failure. BK256 and large
split-K non-TMA kernels regressed; O orientation and router M=128 gains repeated.
Raw response/profile are retained as `multi_category_round2.json/txt`.

[Round three](multi_category_round3_experiment.md) completed at 3935.59 us.
Prepared pooler/unpooler transposes improved normalized latency by 2–9%, but
narrower router tiles and chunked GEMV regressed. O/router M128 gains repeated.
Raw response/profile and per-case analysis are retained as
`multi_category_round3.json/txt` and `multi_category_round3_analysis.json`.
The consolidated `build/multi-category-round3/keepers.py` is ready for one user
check. Its estimated 3858 us score is hypothetical, not a measured improvement.
The user-reported competitor target is 2950 us; no fresh leaderboard query has
been made.

[Next-round ideas](next_round_ideas.md) ranks natural output layouts, tile-major
BF16 weight packing, wider router high/low dots, and a Hopper pipeline. The
budget uses the pending consolidation estimate, not a new measured result.

The [expanded 21-idea map](optimization_idea_map.md) adds concrete load, pipeline,
GEMV, reduction and scheduling follow-ups. [Round four](multi_category_round4_experiment.md)
completed at 3884.83 us. Tiled weights/native O outputs were inconclusive,
reversed QKV M256 and interleaved router M8 regressed, and other changes were
small. Raw response/profile and comparisons are retained as
`multi_category_round4.json/txt` and `multi_category_round4_analysis.json`.

[Round-five planning](multi_category_round5_plan.md) now targets K-loop unrolling,
full-K GEMV warp count and a single-pass FP32 router. The
[round-five candidate](multi_category_round5_experiment.md) passed correctness
at 4321.59 us. Eight-warps GEMV improved all four M1 cases by 4.6–7.6% versus
the latest matching run; unrolling slowed the small GEMMs 10–25%, and direct
FP32 router lost strongly. `build/multi-category-round5/isolated/gemv8.py` is
the clean carry-forward source. Its 3830 us score estimate is not measured.
Raw response/profile and analysis are retained as `multi_category_round5.json/txt`
and `multi_category_round5_analysis.json`.

[Architectural redesigns](architectural_redesigns.md) prioritizes an explicit
Hopper producer/consumer GEMM pipeline, followed by weight-panel reuse. It
separately evaluates tensor-core router reduction and exact compressed-weight
decoding. The fourteen-case dense-engine scope has 63.4% projected sensitivity;
a floor-capped 20% improvement would save about 492 us. `redesign_budget.json`
records the conditional model, not a measured or predicted speedup.

[Round six: explicit Hopper producer/consumer GEMM](multi_category_round6_experiment.md)
passed GPU compilation and all nine correctness tests at 4168.94 us. All four
new paths regressed (31–76%); none is selected. Resource logs show zero spills.

[First-principles analysis](first_principles_analysis.md) motivates the next
[small-router reduction check](router_reduction_experiment.md), which passed at
3848.12 us (job `3f90adf02bb14808`). Only M8/M16/M32 reducer geometry changes
relative to the selected banked control. Normalized latency fell 11.9%/9.4%/6.3%
on those cases; direct control confirmation is pending. All 44 CPU tests pass.
See [the October 25 handoff](handoff_2026-10-25.md) to resume.

Correction: the measured keepers check `13ae262ccdfb40bf` was found under
`build/multi-category-round3/keepers-check.jso` and retained as `keepers.json/txt/log`.
It measured 3838.82 us; the router candidate is 0.24% slower overall, within
uncertainty. Against that run, its three router gains are 4–6%, offset by slower
unchanged cases. [The direct historical comparison](router_reduction_vs_keepers.json)
records the actual launch differences and counterfactual attribution.

The [cluster-router experiment](router_cluster_experiment.md) completed with
passing GPU correctness after frontend/layout fixes, but both configurations
lost to the two-kernel B32 control. Plans and conditional score scenarios are
retained in `router_cluster_plan.json`; measured outcomes are in
`router_cluster_fixed.*`, `router_cluster_bk64.*` and `round7_analysis.json`.

[The OKC networking diagnosis](okc_network_diagnosis.md) describes the earlier
restricted session. Connectivity is restored and agent-run GPU checks succeed.
