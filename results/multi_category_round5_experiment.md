# Round five: loop scheduling and a single-pass router

**Completed:** job `6456b973b73a4a26` passed all correctness and timed cases at
**4321.59 us / 1.1178293x**, with interval [1.1010618x, 1.1345967x]. This combined
candidate is a regression. Retained evidence is `multi_category_round5.json/txt`
and `multi_category_round5_analysis.json`.

## Findings

| Experiment | Normalized latency change vs matching predecessor | Decision |
|---|---:|---|
| M1 QKV eight warps | 7.4% faster | Carry forward for confirmation. |
| M1 pooler eight warps | 4.6% faster | Carry forward for confirmation. |
| M1 unpooler eight warps | 4.8% faster | Carry forward for confirmation. |
| M1 router eight warps | 7.6% faster | Carry forward for confirmation. |
| QKV unroll2, M8/M16/M32 | 21.0% / 20.8% / 25.1% slower | Reject these unroll settings. |
| Pooler unroll2, M8/M16/M32 | 10.3% / 9.5% / 9.7% slower | Reject these unroll settings. |
| Direct router, M8/M16/M32 | 67.9% / 151.7% / 329.9% slower | Reject this direct FP32 design. |

Comparisons use the official paired per-case ratios versus each same-run
SGLang reference. GEMV compares with `0fe4f6662e1d458a`, unroll with
`327bb5b2fdc6423a`, and direct router with `9515208aace4455e`; each predecessor
matches the original setting for that group. The six cases whose configurations
match both rounds four and five shifted from 3.1% faster to 3.0% slower, with
median 0.9% slower. The coherent four-case GEMV result is larger than those
control shifts. Against two additional original four-warp runs, the GEMV
improvements range from 3.4% to 9.7%, retaining the same direction in every case.

The direct router's scored effective times grew from 9.9 to 16.0 to 28.3 us
as M increased 8 to 16 to 32; its predecessor took about 5.8/6.4/6.7 us.
It removed a launch but replaced tensor-core work with repeated FP32 vector
reductions. This rejects the tested architecture, rather than establishing
that every single-kernel router is unsuitable. Another router design should
retain tensor-core throughput and improve reuse. Timings do not separate the
direct kernel's arithmetic, register and repeated-load costs.

Unrolling preserved numerical correctness but materially slowed the producer.
Compiler registers/spills/shared-memory data are absent, so do not assert that
spilling or occupancy caused the loss. This result warrants resource reports
before another related compiler-scheduling experiment.

The existing `build/multi-category-round5/isolated/gemv8.py` already combines
the four eight-warp cases with the selected earlier gains and restores the
original unroll/router paths. It is the clean carry-forward source, not an
additional requested check. Applying selected per-case ratios to the historical
base projects **3830.27 us**, about 28 us below the previous 3858 us consolidation
estimate. This is an estimate, not a measured combined score.

## Original candidate and validation record

The candidate implements the
[round-five plan](multi_category_round5_plan.md), restoring the selected earlier
settings for the unhelpful round-four layout variants. Its control matches all
thirty scored launches in the prior consolidation candidate; that combination
has not produced a local response file, so no measured control aggregate is
claimed. Passing evidence is retained from the earlier checks.

Run from the project root:

```bash
okc check build/multi-category-round5/candidate.py --json \
  | tee build/multi-category-round5/check.json
```

Rebuild with:

```bash
python -m tools.build --preset round5 --output build/multi-category-round5/candidate.py
```

## Three independent experiments

| Group | Scored cases | Change | Relevant predecessor evidence |
|---|---|---|---|
| Weight-major K-loop unroll | QKV/pooler M8/M16/M32: 6 | Original storage/math/addressing, `tl.range(..., loop_unroll_factor=2)` | Original weight-major launches measured unchanged in `327bb5b2fdc6423a` |
| Full-K GEMV scheduling | QKV/pooler/unpooler/router M1: 4 | Eight warps rather than four; one output row and the existing full-K kernel | These full-K/four-warp launches were unchanged in `0fe4f6662e1d458a` |
| Single-pass router | Router M8/M16/M32: 3 | One expert and two tokens per CTA, original FP32 weights, full-K FP32 reduction, direct FP32 output | Original split producer/reducer measured in `9515208aace4455e` |

The thirteen changed cases do not overlap. Seventeen scored calls match the
selected control. These include earlier O orientation, pooler/unpooler prepared
transposes and router M128 split16, plus unchanged large QKV and router M256.
O M1 keeps its original four-output-row/four-warp configuration.

The direct router has one output allocation and one kernel launch, replacing
the temporary split partials and PDL reducer on its three cases. Its grids are
1024/2048/4096 CTAs at M8/M16/M32. It reuses an expert's FP32 vector across two
tokens and masks an odd final token, including the M9 outlier correctness case.
All input-dependent work remains in the timed call; fresh prepared-state weights
are read directly. It neither caches activation work nor stores altered-precision
router weights. The original hi/lo path remains selected at larger token counts.

The unroll variant's AST is identical to the passing weight-major kernel after
normalizing the loop iterator and function name. Its tile sizes, split counts,
stages, FP32 accumulation, partials, masks, PDL signaling and stores are unchanged.
The compiler chooses the actual scheduling. The option is present in the
[pinned Triton 3.8.0 source](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/core.py).

## Validation and remaining uncertainty

All **36 CPU tests pass**, including:

- Actual router address and mask expressions evaluated on odd M3/M9/M17, padded
  K, FP32 weights, BF16 activations, large activations, zero rows and exact probes;
  rounded FP32 products/tree reductions are compared with an FP64 sum.
- Direct router grid, one-allocation/one-launch behavior, output precision and
  current cloned-state binding.
- Unroll-kernel arithmetic/address AST equality, independent case scope and
  preservation of all checkpoint launches/JIT bodies.
- All preset host launches matching compiler signatures, constants and options.

The generated default build is current. The candidate has a **29-job GPU compile
plan**, compared with 32 jobs for its split-router control. These are planned
jobs, not completed GPU compilations. Torch/Triton and a GPU are absent locally;
installing pinned Triton into `/tmp` failed at PyPI DNS resolution. No OKC check
was attempted by the agent. The subsequent user-run check above established
GPU compilation and correctness; it supplies the performance evidence.

The router's weight traffic and register use may outweigh the saved launch,
especially at M32. Unrolling can increase resource use without improving runtime.
CPU math checks did not establish GPU numerical behavior or a speedup. The
completed OKC result above passed correctness and decides each case independently.

Preparation adds only the selected 128 MiB of BF16 pooler/unpooler transpose
copies beyond original weights/hi-lo tensors. The failed tiled/interleaved weight
copies are not used by this preset. Nearby QKV/pooler M2..32 use the unroll path;
nearby router M2..32 uses direct FP32, while larger fallback follows the selected
control policy.

`build/multi-category-round5/manifest.json` records source hashes, all case
changes, comparison jobs and validation limitations. `control.py` preserves
the selected control settings. Three `isolated/*.py` exports are available for
failure localization; only the combined candidate is requested now. Their
dispatches were compared on scored and sampled nearby shapes without GPU checks.

Analyze official per-case paired ratios against same-run SGLang, using matching
predecessor settings and unchanged-case variation. The prior planning payoff
model is conditional, not a predicted score for this new kernel.
