# Muse whole decode layer

Optimize one TP8/EP8 decoder layer on H200. Read [TASK.md](TASK.md) for the
computation, precision rules, correctness limits, and scoring formula.

The measured default, **`full_fa3_m16_finalize`**, passed all **13 correctness
tests and 12 timed shapes** in check `418dc106c212475a`: **19046.05 us /
1.1876636x**, interval [1.1698486, 1.2054785]. It uses FA3 for full attention,
custom split attention for sliding, Marlin block cap 16, and fused expert
finalization for T <= 16. The exact checked source and evidence are retained
in `results/small-finalize/`. The default selects that policy, with the new
direct expert path disabled; its added inactive code means the generated
file is no longer byte-identical to the retained checked source.

The corrected user-reported target is **9580 us / 2.36x**, requiring **49.70%
less latency** from this control. No live leaderboard query has been made.

Block cap 8 failed its performance experiment: 20511.35 us / 1.1028188x,
about 7.69% slower than the default. The next candidate uses direct NVFP4
expert GEMVs only for single-token decode, with Marlin retained for all
larger token counts. It is frozen at `build/single-nvfp4/candidate.py`;
GPU validation is pending. See [the next experiment](results/single_nvfp4_experiment.md).

## Editable source

| Path | Responsibility |
|---|---|
| `decode/spec.py` | Input contracts, fixed dimensions, expert constants |
| `decode/config.py` | Full-attention backend preset: `control` or `full_fa3` |
| `decode/projections.py` | GEMV, split-K BF16 GEMM, PDL reduction, projection dispatch |
| `decode/attention.py` | QK norm/RoPE/cache store, split-context attention, merge/norm/gate |
| `decode/boundaries.py` | Expert-input norm and attention/FFN residual boundaries |
| `decode/experts.py` | NVFP4 Marlin repacking, routing alignment, expert evaluation |
| `decode/nvfp4_gemv.py` | Experimental direct single-token NVFP4 expert path |
| `decode/runtime.py` | Constant-only weight preparation, router/pooler overlap, layer execution |
| `tools/build.py` | Deterministic standalone export and optional source-hash manifest |
| `tools/analyze.py` | Check status, load score reconstruction, shape sensitivity, comparisons |
| `tests/` | CPU refactor equivalence and build/analysis checks |
| `results/` | Evidence ledger and experiment plan |
| `build/` | Ignored frozen candidates, manifests, and temporary analyses |
| `submission.py` | Generated judge entrypoint; edit `decode/` instead |

`baseline.py`, `reference.py`, `checks.py`, `nvfp4.py`, and `TASK.md` remain the
challenge references. The builder includes no project/reference imports in
the standalone export. Source modules have explicit relative imports for
development; the builder strips those imports and deduplicates external ones.

## Build and verify

Run from this directory; these commands require only Python:

```bash
python -m tools.build
python -m tools.build --check
python -m unittest discover -s tests -v
```

The checkpoint in `tests/fixtures/fp8_fix.json` binds the retained split-attention
implementation, existing GPU kernels, preparation, and layer execution to the
passing fixed candidate. The split implementation is now named
`_split_attention`; the `_attention` wrapper selects the preset. Checks
explicitly allow the new FA3 path and test its host dispatch/arguments.
Source equivalence does not establish GPU compilation or numerical correctness.

Kernel math, precision boundaries, and dispatch choices remain in their stage
modules. `tools.build --preset control` exports split attention for all layers;
`--preset full_fa3` selects FA3 only when the window is zero. The default
`--preset full_fa3_m16_finalize` caps Marlin token blocks at 16 and enables
the passing small-token expert finalize.

## Freeze an experiment

Build a named candidate before starting its check, so later edits cannot make
the result's source ambiguous:

```bash
python -m tools.build --preset single_nvfp4 \
  --output build/single-nvfp4/candidate.py --manifest build/single-nvfp4/manifest.json
okc check build/single-nvfp4/candidate.py --json > submission7.json

# Retain a dispatch-equivalent control without replacing submission.py:
python -m tools.build --preset control --output build/control/candidate.py \
  --manifest build/control/manifest.json
```

Each export is self-contained. Its optional manifest records the exact exported
file's SHA256 and the source-module hashes. Retain a useful check response,
candidate/manifest, and generated profile together in a named `results/`
experiment directory. Existing `submission1.json`, `submission2.json`, and
`profiles/` paths are preserved. The passing control's source, response,
profile, and analysis are retained in `results/fp8-fix/`, `results/full-fa3/`, and `results/moe-m16/`.

## Read a check

```bash
python -m tools.analyze submission1.json
python -m tools.analyze submission2.json --output build/check2-analysis.json

# After both runs pass, compare matching shapes normalized to same-run SGLang:
python -m tools.analyze build/experiment.check.json \
  --compare build/control.check.json --output build/comparison.json
```

Empty output files are reported as pending. Failed checks report their first
failure and yield no timing analysis. Passing checks reconstruct the geometric
mean of load speedups from `weight * median_ns` and `weight * baseline_ns`,
then rank whole shapes by their share of the score's latency sensitivity.
The tool verifies that reconstruction against the reported speedup. Comparison
ratios below 1 mean lower candidate/SGLang latency than the previous run.

Use the generated kernel profile to identify stages within a costly shape.
Profile times come from a separate pass, and router/pooler overlap means their
individual times cannot simply be added to predict end-to-end latency. Judge
CUDA-graph timings with cold L2 are the performance evidence.

See [the experiment plan](results/plan.md) for the next steps and
[the evidence ledger](results/README.md) for the current state.
