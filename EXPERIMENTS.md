# Guild-inspired projection experiments

Both candidates are standalone files based on selective's measured 1.2048418x
check (`027b6db0ef044348`, result `submission-4`). The control's source hash is
`6c3bf8a7554abcb372647b1192b5babd279d7fd0d48e8a51b223cc2c2af0286b`.

Run from this directory, sequentially:

```bash
okc check submission_triton_orientation.py --json | tee submission-7
okc check submission_triton_pdl.py --json | tee submission-8
```

Both checks are complete: each passed all 9 correctness tests and 30 timed
shapes, with no flags or rejected integrity checks.

| Result | Candidate | Speedup | 95% interval | Score (us) |
|---|---|---:|---|---:|
| 4 | Selective control | 1.2048418x | [1.18677, 1.22291] | 4009.49 |
| 7 | O orientation | 1.2150730x | [1.19685, 1.23330] | 3975.73 |
| 8 | Dependent launches | 1.2310594x | [1.21259, 1.24953] | 3924.10 |

Relative to the historical control, the aggregate ratios improve 0.85% and
2.18%. Their intervals overlap the control, so the aggregate gains remain
uncertain. The affected per-case ratios are more promising: O orientation
improves 5.1%, 11.6%, and 10.5% at M=8,16,32; PDL improves router ratios
13.6% and 10.9% at M=16,32. These compare paired candidate/SGLang ratios across
runs, not same-run direct candidate-to-candidate measurements.

PDL's QKV M=32 and full unpooler paths are unchanged, despite changed measured
ratios; do not attribute every favorable case to the code changes.
The complete comparison is in `profiles/experiments_comparison.json`.

The next check combines both ideas:

```bash
okc check submission_triton_combined.py --json | tee submission-9
```

`submission_triton_combined.py` retains PDL's checked kernel bodies and adds
orientation's checked O kernel and dispatch. Its combination needs its own
GPU correctness and timing check.

| Candidate | Experiment | Shapes to inspect first |
|---|---|---|
| `submission_triton_orientation.py` | Compute O projection as W @ X.T; store directly into contiguous [M,N] output, no intermediate transpose | O at M=8,16,32 |
| `submission_triton_pdl.py` | Enable PDL only for the reducer; producers signal after K loop, consumer waits before reading any partial | Router at M=8,16,32,128,256; small QKV/pooler and sliced unpooler |

The orientation candidate covers O at 2..32 tokens. Weight operand is [64,128],
activation operand [128,16 or 32], accumulator [64,16 or 32], with 4 warps and
4 stages. All loads and output stores mask irregular dimensions. It retains
row-major BF16 weights and stores accumulator[n,m] at output[m*N+n].

The PDL candidate retains the control's math, precision, dispatch, tile sizes,
split counts, and temporary allocation shapes. It uses Triton 3.8.0's
`tl.extra.cuda.gdc_launch_dependents()` in split producers and `gdc_wait()` at
reducer entry, with `launch_pdl=True` only on reducer launches. It requires
Hopper or newer. It uses no counters, persistent workspace, or input caching.

NVIDIA documents PDL and stream capture in the
[CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).
The ideas were motivated by [BF16 discussion](https://guild.1nterface.ai/t/5#p14),
with our implementation written locally. Other teams' reported gains are not
predictions for these candidates.

Local verification:

```bash
python -m unittest test_projection_experiments test_projection_diagnostics test_lossless_codec -v
python inspect_codegen.py --cases orientation pdl --output profiles/codegen-experiments
```

The original 22 CPU tests passed; the added combination test also passes. Dispatch checks cover 100 shape/layer combinations for
PDL and 50 for orientation, and verify the compiler manifest against host
launch settings. Removing PDL signaling/wait calls recovers the control's
original GPU math exactly at the AST level. These are control-flow checks,
not numerical GPU validation.

All 23 representative AOT jobs pass with Triton 3.8.0 and sm_90a. The audit
checks reducer wait-before-load and producer signal-after-MMA/before-store
placement in emitted PTX, and the launch flags in compiler metadata. All 18
PDL ordering checks pass. Original cubins and PTX are saved alongside
`profiles/codegen-experiments/report.json`. The O kernels use 40 registers for
M<=16 and 48 above, with zero spill bytes in verbose ptxas reassembly. Compiler
resource counts do not establish runtime performance.

Compare results with selective by affected case and aggregate speedup. The
judge measures each candidate against SGLang on the same GPU, whereas our
historical candidate-to-candidate comparison spans runs. Small aggregate
differences within the reported intervals remain uncertain. The combination
test verifies identical checked JIT bodies and correct dispatch composition
across 100 shape/layer combinations. This validates the merge, not its GPU
correctness or performance.

## Next broader experiment: TMA dense projections

The next candidate is `submission_triton_tma.py`, based on the combination.
It replaces QKV and pooler at 2..256 tokens and full unpooler at 9..256 tokens.
The O, router, one-token and larger-batch fallback paths retain the combined
implementation. No result for the combined candidate was present when this
experiment was prepared.

```bash
okc check submission_triton_tma.py --json | tee submission-10
```

This tests a broader implementation change:

- Weight-major BF16 tensor-core math, with FP32 accumulation.
- Host tensor descriptors for TMA loads directly into shared memory. Construct
  both descriptors in the timed call from the current tensors, including any
  cloned prepared-state weights. `prepare()` stores no descriptors or addresses.
- Adjacent programs span token tiles for the same output-feature tile, to
  encourage weight reuse within a replay. The graph still starts with cold L2.
- 64 output features per tile, 16/32/64 token slots, K tile 128, 4 warps,
  3 stages. A fixed shape heuristic splits K only until the output grid has
  at least 128 CTAs. Full unpooler uses one kernel, as do larger pooler cases.
- Split paths retain producer signaling and the checked PDL reduction.

This is a single broader candidate, so a gain or regression does not isolate
TMA from the changed orientation, scheduling and split heuristic. Per-layer
results will determine what to retain and which ablation is worth running.

The descriptor API and TMA loading pattern were verified against the installed
Triton 3.8.0 source and the official
[persistent-matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html).
This candidate does not implement a persistent scheduler or manual warp
specialization. It uses Triton's generated software pipeline and barriers.

All 19 representative Hopper AOT jobs pass, including 7-token pooler,
130-token unpooler and other nearby shapes. Generated PTX contains TMA loads
and BF16 WGMMA. Verbose ptxas reassembly reports 34..60 registers/thread,
zero spill loads/stores, and 61464..98328 shared bytes. The report is
`profiles/codegen-tma/report.json`. These are compile-only checks, with no
new GPU timing or numerical correctness claims.

All 24 CPU tests pass. The new tests check 80 shape/layer dispatch combinations,
current-call descriptor binding under independent state copies, K partition
coverage, output shapes/dtypes, compiler signatures and launch settings.
GPU validation remains the next step.

The score sensitivity report `profiles/tma_target_budget.json` uses result 8.
The target still requires 23.54% less total latency. Hypothetically reducing
latency on the TMA-targeted shapes by 20% gives about 1.414x aggregate, and 30%
gives 1.530x. Putting only those shapes at the task's optimistic reported
floors gives 1.590x. These are mathematical scenarios, not expected timings.
We will also need gains on the remaining paths to reach 1.61x. One-token
bandwidth is the next broad target after this experiment; further router
polishing has less aggregate value.
