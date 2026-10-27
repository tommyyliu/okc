# Muse BF16 projections

Thirty scored cases: five projections at M = 1, 8, 16, 32, 128, 256. Start with
[TASK.md](TASK.md) for the computation, precision requirements, and scoring rules.

The default is the **PDL candidate measured at 1.2310594×**. This refactor retains
its GPU math, preparation, and launch settings; the rebuilt file has not yet had
a new GPU check. Combined and TMA are experimental presets, not measured upgrades.

The [October 26 autonomous experiments](results/round7_experiment.md) have a
fresh measured banked control at **3830.40 us / 1.26117x**. Command networking
works, and a local Triton 3.8 compiler is installed in `build/compiler-env`.
Cluster-router and deeper WGMMA prototypes passed correctness but lost on
performance. Small TMA and scheduling trials are recorded with their exact
results. The selected `round7` preset passed at **3764.88 us / 1.28312x**;
the 1.71% improvement over the fresh control still has overlapping intervals.
Its exact checked source is `results/round7_selected_source.py`. PDL remains
the default configuration.

The [five follow-up checks](results/round8_experiment.md) all passed correctness
but found no improvement over that selected candidate. Three TMA stages,
router M1 at sixteen warps and four-warp B32 reducers remain the best tested
choices. The daily check quota is exhausted; an unmeasured QKV S4 candidate is
compiled and staged at `build/next-qkv-split4/candidate.py` for the next reset.

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
python -m tools.build --preset pooler_tma --output build/pooler_tma.py
python -m tools.build --preset round3 --output build/multi-category-round3/candidate.py
python -m tools.build --preset round4 --output build/multi-category-round4/candidate.py
python -m tools.build --preset round5 --output build/multi-category-round5/candidate.py
```

The [round-three candidate](results/multi_category_round3_experiment.md) tests
prepared BF16 weight transposes, narrower router tiles, and chunked single-token
reductions, retaining the repeated O/router gains and small weight-major base.
It passed at 3935.59 us; selected layout/O/router gains are staged for consolidation.
The [round-four candidate](results/multi_category_round4_experiment.md) tests
tiled weights, reversed library operands, natural O stores and interleaved router
components. It passed GPU compilation/correctness at 3884.83 us; the new groups
gave no clear broad gain. [Round-five planning](results/multi_category_round5_plan.md)
targets loop scheduling, GEMV warps and a single-pass router. Its
[candidate passed](results/multi_category_round5_experiment.md) at 4321.59 us.
The four eight-warp GEMV cases improved, while unrolling and direct FP32 router
regressed. The clean selected source is `build/multi-category-round5/isolated/gemv8.py`;
its aggregate has not been measured.

The [round-six prototype](results/multi_category_round6_experiment.md) uses explicit
Hopper producer/consumer warp specialization for large QKV/pooler. It preserves
selected settings elsewhere, including GEMV8. It passed GPU compilation/correctness at 4168.94 us, but all four new paths
regressed by 31–76% against matching selected settings. Keep the banked control;
the producer needs deeper overlap before another check.

The [small-router reduction check](results/router_reduction_experiment.md) passed
at 3848.12 us. M8/M16/M32 normalized router latency improved 11.9%/9.4%/6.3% versus
round six. The measured keepers control was 3838.82 us, so this aggregate is
0.24% slower, within uncertainty; router gains versus keepers are 4–6%.
The fresh banked control confirms 5–9% normalized gains in a newer B32 run. It changes
only reduction blocks from 256 to 32 outputs. All 44 CPU tests pass. The
[October 25 handoff](results/handoff_2026-10-25.md) records the next steps.

The [cluster-router experiment](results/router_cluster_experiment.md) is staged
at `build/router-cluster/candidate.py`. It replaces the three small-router paths
with one tensor-core kernel using cluster-local K reduction and no global
scratch buffer. Its exact checked predecessor is preserved as `control.py`.
The first remote checks failed during frontend compilation. The corrected
kernel now passes GPU checks, but both BK256 and BK64 configurations regress.
All 54 CPU tests pass. [Agent-run OKC connectivity](results/okc_network_diagnosis.md)
is restored, and remote checks can run directly from this workspace.

Both the [pooler TMA check](results/pooler_tma_experiment.md) and
[split-K follow-up](results/pooler_splitk_experiment.md) passed correctness but
lost to cuBLAS at the changed sizes. PDL remains the default. The fresh control
and [optimization priorities](results/optimization_priorities.md) point to
small-batch QKV/pooler and single-token GEMV as the next areas to investigate.

```bash
python -m tools.build --preset pooler_splitk --output build/pooler_splitk.py
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

The tuner searches GEMV and streaming GEMM settings, preserving the
selected source's configuration as a control. Streaming GEMMs compare both
operand orientations, K tiles, pipeline stages, and independent K partitions;
cuBLAS competes for every non-router case. The new weight-major streaming
kernel retains row-major BF16 weights, FP32 accumulation, and PDL reduction.
It passed an initial H200 check at 1.2324506x; confirmation and nearby
validation of the new kernel are pending. PDL remains the default.
The tuner does **not yet sweep TMA or router GEMM settings**.
Use `--source build/tma.py` to retain TMA as
the control. Partial sweeps preserve all unswept choices and export exact-shape
overrides into a new standalone file. Promote measured winners into
`projection/configs.py`, then rebuild. Generated tuning files are not the source
of truth.

Timing uses cold-L2 CUDA graphs with paired comparisons and a fresh-input
confirmation. Recheck finalists with the judge and validate nearby sizes before
promotion; a small historical timing difference is not an established gain.

The [next optimization experiment](results/small_gemm_experiment.md) targets
QKV and pooler at M=8/16/32. Inspect or run a bounded sweep on an H200:

```bash
python -m tools.tune --plan --layers qkv_gate pooler --tokens 8 16 32
python -m tools.tune --layers qkv_gate pooler --tokens 8 16 32 \
  --stages 2 3 4 --k-tiles 64 128 256 --splits 2 4 8 \
  --output build/small-gemm-tuned.py --report build/small-gemm-tuning.json
```

Without a local GPU, export a PDL control and eight standalone experiments:

```bash
python -m tools.tune --ablations build/small-gemm --layers qkv_gate pooler --tokens 8 16 32
okc check build/small-gemm/control.py --json > build/small-gemm/control-check.json
okc check build/small-gemm/streaming.py --json > build/small-gemm/streaming-check.json
okc check build/small-gemm/weight_major.py --json > build/small-gemm/weight-major-check.json
```

`manifest.json` records each experiment's changes and comparison file. Start
with the control and streaming reference, then test variants within the check
budget. The weight-major candidate changes operand orientation alone relative
to `streaming.py`. The initial checks show lower normalized latency across all
six target cases, with overlapping aggregate confidence intervals. The
[experiment record](results/small_gemm_experiment.md) includes the results and
the next nearby-validation candidate.

## Inspect and analyze

```bash
python -m tools.codegen --preset tma --layers pooler --tokens 128 256 --plan
python -m tools.codegen --preset tma --layers pooler --tokens 128 256
python -m tools.codegen --source build/small-gemm/weight_major.py \
  --layers qkv_gate pooler --tokens 8 16 32 --output build/small-gemm-codegen
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
