# Muse BF16 projections

Thirty scored cases: five projections at M = 1, 8, 16, 32, 128, 256. Start with
[TASK.md](TASK.md) for the computation, precision requirements, and scoring rules.

The default is the **PDL candidate measured at 1.2310594×**. This refactor retains
its GPU math, preparation, and launch settings; the rebuilt file has not yet had
a new GPU check. Combined and TMA are experimental presets, not measured upgrades.

## Where to work

| Path | Purpose |
|---|---|
| `projection/configs.py` | One explicit setting per scored `(layer, M)`; preset overrides and nearby-size fallback |
| `projection/shapes.py` | Dimensions and scored token counts |
| `projection/kernels/gemv.py` | Single-token reduction |
| `projection/kernels/gemm.py` | Streaming, router hi/lo, and weight-major GEMM |
| `projection/kernels/tma.py` | Experimental Hopper TMA GEMM |
| `projection/kernels/reduce.py` | PDL split-K reduction |
| `projection/runtime.py` | Weight preparation and kernel launches |
| `tools/` | Build, tuning, GPU validation, compile audit, score analysis |
| `results/` | Retained benchmark evidence and status notes |
| `tests/` | CPU regression and tooling checks |
| `submission.py` | Generated standalone PDL submission; edit the source modules instead |

`baseline.py`, `reference.py`, `checks.py`, and `TASK.md` are the original challenge
files. They are development references, never included in a submission.

## Build and check

Run commands from this directory:

```bash
python -m tools.build
python -m tools.build --check
python -m unittest discover -s tests -v

# On an H200, using the judge's Torch/Triton versions:
python -m tools.validate submission.py --nearby
mkdir -p build
okc check submission.py --json > build/pdl-check.json
```

Use Triton 3.8.0 to match the recorded judge environment, which ran
PyTorch 2.14.1+cu130.
CPU tests and building require only Python; GPU execution requires Torch,
Triton, and Hopper for PDL/TMA.

Build the unmeasured experiments separately:

```bash
python -m tools.build --preset combined --output build/combined.py
python -m tools.build --preset tma --output build/tma.py
```

The builder embeds project sources in dependency order and strips relative
imports. Every output is a single file with the OKC header; neither project
modules nor benchmark/reference tools are needed by the judge.

## Tune one case

Choose the `(layer, M)` row in `projection/configs.py`. Launch tuples are
documented there; `BM` means tokens and `BN` means output features for every
family. Presets share GPU implementations, so editing a kernel affects every
case using it. Editing a configuration affects that exact case. Non-scored
sizes retain the previous range-based policy in `fallback_config`.

```bash
python -m tools.tune --plan --layers pooler --tokens 128 256
python -m tools.tune --layers pooler --tokens 128 --output build/pooler-tuned.py
python -m tools.validate build/pooler-tuned.py --nearby
okc check build/pooler-tuned.py --json > build/pooler-check.json
```

The existing tuner searches GEMV and streaming GEMM settings, preserving the
selected source's configuration as a control. It does **not yet sweep TMA,
orientation, or router GEMM settings**; extend `options()` in `tools/tune.py`
when investigating those families. Use `--source build/tma.py` to retain TMA as
the control. Partial sweeps preserve all unswept choices and export exact-shape
overrides into a new standalone file. Promote measured winners into
`projection/configs.py`, then rebuild. Generated tuning files are not the source
of truth.

Timing uses cold-L2 CUDA graphs with paired comparisons and a fresh-input
confirmation. Recheck finalists with the judge and validate nearby sizes before
promotion; a small historical timing difference is not an established gain.

## Inspect and analyze

```bash
python -m tools.codegen --preset tma --layers pooler --tokens 128 256 --plan
python -m tools.codegen --preset tma --layers pooler --tokens 128 256
python -m tools.budget --result results/pdl.json --target 1.61
```

Codegen derives its manifest from source configurations, compiles for sm_90a,
and reports registers, spills, and PTX instruction counts. Compilation needs
Triton but no GPU. It does not establish correctness or runtime performance.
Builds, tuning reports, compiler artifacts, and new OKC profiles are ignored by
Git. Copy useful measured evidence into `results/` deliberately.

## History

Commit `9ed8e4b`, tagged `pre-reorganization`, captures the complete previous
project, including superseded candidates, old result files, diagnostics, and
compiler artifacts. The working tree keeps the active implementations and
selected evidence. To inspect an old file without restoring it:

```bash
git show pre-reorganization:submission_triton_lossless.py
git show pre-reorganization:EXPERIMENTS.md
```

The repository was initialized in `muse_bf16_proj/` because this sandbox exposed
no usable parent repository. It has no configured remote.
