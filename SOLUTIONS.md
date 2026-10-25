# CUDA and Triton candidates

The CUDA and Triton files are independent submissions for `muse_bf16_proj`.
The original `submission.py` is preserved. The original Triton candidate passed
H200 check `80eef28095a14222`; the hybrid passed `e56247c3b96f407e` at 1.1392x.
The streaming candidate passed `ba0e8c407d054666` at **1.1860x**.
The best measured candidate is now `submission_triton_selective.py`, check
`027b6db0ef044348`, at **1.2048x**. Lossless check `d9c2df5ce5bd4143` passed
correctness but failed as a performance experiment at **0.1076x**.
The user supplied **1.61x** as the leading-score target. The score budget and
generated-code/decoder diagnostic workflow are documented in
[DIAGNOSTICS.md](DIAGNOSTICS.md). Actual compiler and timing measurements remain
blocked locally. Diagnostic H200 check `c45c07be9e424ba8` now supplies real
register/spill/load metadata: see `submission-6` and
`profiles/codegen-h200/runtime_report.json`.
CUDA remains uncompiled and untested on a GPU.

| | `submission_cuda.py` | `submission_triton.py` |
|---|---|---|
| Implementation | CUDA C++ compiled with `load_inline` in `prepare` | Triton JIT |
| Single token | BF16-pair / FP32-pair loads; one warp per output | Tiled loads and reductions |
| Other batch sizes | BF16 WMMA, 16x64 or 32x64 output tiles | BF16 `tl.dot`, 16/32/64x64 output tiles |
| Parallelism | Split K when the output has too few tiles | Same heuristic, using its own tile counts |
| Router | Original FP32 for one token; fused BF16 hi/lo otherwise | Same precision strategy |
| Preparation | Pack weights into `[N/16, K/64, 16, 64]`; split router | Split router; retain row-major weights |
| Partial results | FP32 buffer and deterministic reduction | FP32 buffer and deterministic reduction |

Both implementations compute every projection using their own kernels. They
handle the unpooler slice through `config["n"]`, and mask incomplete token tiles.
Weights retain their BF16 values, except for the explicitly allowed router
hi/lo representation. Neither uses atomics, host synchronization, input-value
branches, or activation-dependent preparation.

The native implementation makes the tensor-core fragments, shared-memory
staging, alignment, warp reductions, stream handling, and extension binding
explicit. Triton generates most of that machinery, so its implementation is
substantially shorter. Neither candidate uses Hopper-specific TMA/WGMMA or
autotuning yet. The tile sizes and split counts are hypotheses to benchmark,
not established performance improvements across all shapes.

## First H200 result and next experiment

Check `80eef28095a14222` passed all nine correctness cases and all thirty scored
shapes, with no integrity flags. Its overall speedup was **0.9343x**, with a
95% interval of **[0.9203, 0.9483]**: roughly 7% more latency than SGLang.
The saved reports are in `profiles/submission_triton.80eef28095a14222.*`.

The fused router won at every scored batch size, reaching 4.15x at eight tokens
and 1.71–2.17x at 16–256 tokens. Single-token `o` and `pooler`, and the sliced
unpooler at eight tokens, also won. Most other BF16 GEMMs lost; pooler at 256
tokens took 55.75 us against SGLang's 30.62 us. The qkv result at eight tokens
was effectively tied (1.004x).

`submission_triton_hybrid.py` is a standalone candidate that keeps those
winning paths and uses `torch.mm(x, w[:n].T)` elsewhere. The kernels and launch
parameters are unchanged from the original checked version, isolating the
dispatch choice. The sliced-unpooler branch also handles token
counts two through seven. The original Triton and CUDA files are unchanged.

Using the original run's per-shape timings, and assuming PyTorch's BF16 GEMM
matches SGLang for fallback cases, this dispatch models about **1.139x** overall.
The subsequent H200 check confirmed the estimate:

| Candidate | Speedup | 95% interval | Score |
|---|---:|---:|---:|
| Original Triton | 0.9343x | 0.9203–0.9483 | 5170.53 us |
| Hybrid | **1.1392x** | **1.1221–1.1563** | **4240.40 us** |
| Streaming | **1.1860x** | **1.1682–1.2038** | **4073.20 us** |
| Selective (best measured) | **1.2048x** | **1.1868–1.2229** | **4009.49 us** |
| Lossless (performance regression) | 0.1076x | 0.1060–0.1093 | 44876.92 us |
| Lossless diagnostic (same GPU work) | 0.1068x | 0.1052–0.1084 | 45244.74 us |

Hybrid check `e56247c3b96f407e` passed all nine correctness tests and all thirty
scored shapes, without integrity flags. The result is in `submission-2` and
`profiles/submission_triton_hybrid.e56247c3b96f407e.*`. Its fallback paths stayed
within roughly 1.5% of SGLang, and its router gained 1.16–3.97x. All six loads
improved: decode 1/16/32 by 1.075/1.136/1.122x, verify 1/16/32 by
1.356/1.096/1.074x. This was a check, so it did not publish a leaderboard entry.

To reproduce the validated hybrid:

```bash
okc check muse_bf16_proj/submission_triton_hybrid.py --json
```

The command above runs from the workspace root (`okc-work`). From inside
`muse_bf16_proj`, omit that directory prefix. The hybrid result reported eighteen
of twenty daily checks left and no scored submissions. These are historical
limits, not a live allowance query.

## Streaming result and selective dispatch

`submission_triton_streaming.py` passed H200 check `ba0e8c407d054666`: all nine
correctness tests and thirty scored shapes, without integrity flags. It reduced
the reported score latency by 3.94% relative to the hybrid and by 15.68% relative
to SGLang. The result is in `submission-3` and its matching saved profiles.
It reported seventeen checks left and no scored submissions. Its changes were:

- All single-token projections: load all K entries for each output reduction,
  eliminating the serial K loop and the router's separate split-K reduction.
  Multiply and reduce in FP32, using weights at their original precision.
- QKV, pooler, and full unpooler at 2–32 tokens: increase the GEMM K tile from
  64 to 128, with BF16 tensor cores and FP32 accumulation/partial sums.

Other calls retain the validated hybrid dispatch. The full GEMV uses four
output rows per program for K <= 1024 and one otherwise to limit register use.
This workspace still has neither PyTorch/Triton nor a CUDA GPU. The challenge
hostname also failed to resolve when querying the leaderboard, so rank could
not be verified here.

Local validation passed Python syntax, dispatch coverage for 5,500 shape/layer
combinations, sweep partition alignment, and a standalone export round trip
that preserves all kernel code. These do not establish GPU correctness.
The check attempt is recorded in
`profiles/submission_triton_streaming.check.20261024T201309Z.receipt.json`;
the server was unreachable during that attempt. The subsequent user-run check
provided the passing GPU result above.

The full-K single-token kernels improved QKV, pooler, unpooler, and router;
the o path was effectively tied with the previous custom kernel. The small
streaming GEMMs helped QKV at 8/16 tokens and pooler at 8/16/32. Full unpooler
lost at 16/32 (0.927x/0.878x SGLang) and QKV lost at 32 (0.953x).

`submission_triton_selective.py` keeps the winning paths and restores the
hybrid's cuBLAS path for full unpooler and QKV above 16 tokens. It changes
only the default dispatch and module description; all kernel bodies, launch
settings, and preparation match the checked streaming file. The policy applies
to ranges of token counts, including nearby sizes. Neither measured candidate
is modified.

Before checking, paired baseline timings modeled **1.1997–1.2010x** overall.
H200 check `027b6db0ef044348`, saved in `submission-4`, measured **1.2048x**
with a 95% interval of **[1.1868, 1.2229]**. All nine correctness tests and
thirty scored shapes passed, without integrity flags. Score latency fell
1.56% relative to streaming and 17.00% relative to SGLang. The streaming and
selective intervals overlap, so the small incremental gain should not be
treated as established across reruns. Selective is the best measured baseline.

To reproduce the selective check from the workspace root:

```bash
okc check muse_bf16_proj/submission_triton_selective.py --json > muse_bf16_proj/submission-4
```

For a broader search with access to an H200, `tune_streaming.py` measures GEMV
configurations and GEMM tile/split-K choices, including larger M tiles for
verify batches. It now defaults to the selective source and compares against
that source's dispatch. A partial sweep preserves unswept source choices,
including the full-K single-token paths. It captures
a 128 MiB L2 eviction and external CUDA timing events in the same graph,
excluding eviction from the measured interval. ABBA pairs balance drift.
These are development measurements; they do not reproduce all judge controls.

The tuner checks each compiled option against the supplied reference, and
confirms selected options on fresh outlier activations and bucket-edge batch
sizes. Export requires a median gain above 3% and a sample 10th percentile
above 1.5%, including an independent confirmation. This percentile is a
selection guard, not a confidence interval. Unselected and unswept buckets keep
the source dispatch. The output contains only fixed dispatch choices and
kernel code, without the tuner, timing operations, or reference imports.

Run from `muse_bf16_proj` on the GPU machine:

```bash
python tune_streaming.py --plan
python tune_streaming.py
python validate.py submission_triton_tuned.py --nearby
okc check submission_triton_tuned.py --json
```

The tuner saves trial summaries in `profiles/streaming_tuning.json` and writes
`submission_triton_tuned.py` only after the sweep succeeds. Use
`--layers pooler qkv_gate --tokens 16 32` for a smaller initial sweep. Settings
chosen on another GPU need an H200 check before interpreting their speed.

The tile approach follows the [Triton matrix multiplication
tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
The benchmark uses the documented [external CUDA event
nodes](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html).

## Larger architectural experiment: lossless weight compression

`submission_triton_lossless.py` changes the bytes read by the projection,
rather than only choosing among existing kernels. H200 check
`d9c2df5ce5bd4143`, saved in `submission-5`, passed all nine correctness tests
and thirty scored shapes, without integrity flags. It measured only
**0.1076x** SGLang speed, with a 95% interval of **[0.1060, 0.1093]**. Its
44876.92 us score is **11.19x the selective candidate's latency**. It is a
failed performance experiment and should not replace the selective baseline.

Every compressed BF16 scored shape lost against selective. For example,
pooler at 256 tokens rose from 31.32 us to 697.19 us, and single-token QKV
rose from 10.66 us to 50.02 us. The unchanged router stayed close to its prior
speed. Profiles attribute almost all of the extra time to `_lossless_gemv`
and `_lossless_gemm`; final split-K reductions remain around 1–4 us.
These profiles do not expose registers, spills, memory transactions or
instruction stalls. They locate the regression in the fused projection path
but cannot establish its detailed cause. Compiler and hardware profiling
should precede another compression performance experiment.

The subsequent diagnostic check, saved in `submission-6`, passed all nine tests
and thirty shapes with no integrity flags. Its retained compiler records show
255 registers/thread and nonzero spill reports for the 8/32-token variants,
plus many scalar byte loads. Large variants remain slow with zero reported
spills, so spilling is only one part of the problem. The judge uses Triton
3.8.0 and PyTorch 2.14.1+cu130. The detailed evidence and limits of the diagnosis
are in [DIAGNOSTICS.md](DIAGNOSTICS.md). Selective remains the best measured
candidate; a standalone decoder benchmark has not yet run.

The task explicitly permits lossless compression of weights. BF16 consists of
a sign, an 8-bit exponent and a 7-bit mantissa, as documented by
[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/fundamental_types.md).
The new format groups 256 consecutive weights along K. It stores:

| Per block | Narrow exponent range | Wide exponent range |
|---|---:|---:|
| Sign + mantissa plane | 256 bytes | 256 bytes |
| Exponent plane | 128 bytes of 4-bit deltas | 256 bytes of original exponents |
| Base/format marker + byte offset | 6 bytes | 6 bytes |
| Total versus 512-byte BF16 input | **390 bytes (76.17%)** | 518 bytes (101.17%) |

A narrow block has maximum exponent minus minimum exponent <= 15. No exponent
or mantissa is rounded, clipped or dropped. Wide blocks store full exponents;
all 65,536 BF16 encodings remain representable, including signed zero,
subnormals, infinities and NaN payloads. The decoder reconstructs the exact
16-bit BF16 pattern before arithmetic. Block starts are 128-byte aligned.
Metadata is derived only from the supplied weights during prepare; activations
are consumed on every timed call. A layer with less than 10% byte saving uses
the selective fallback instead. Original weights remain available for fallback
calls, so retained state uses more memory even though the compressed path reads
fewer weight bytes per call.

The compressed GEMV reconstructs weights within its full-K reduction. The
compressed GEMM decodes directly into BF16 tensor-core operands, retains FP32
accumulation and partial sums, and uses 128-token tiles for 65–256 tokens to
reuse weights across more rows. It does not expand the whole weight before
each call. The existing router kernels remain unchanged. Calls above 256
tokens use the selective fallback. The large tile configuration targets H200;
other GPUs may have insufficient shared memory for those launch settings.

The byte savings and performance are separate claims. A fully compact 64 MiB
pooler/unpooler weight occupies 48.75 MiB including metadata. An independent
262,144-element Gaussian sample encoded to 76.59% of its original bytes, with
17 raw blocks out of 1,024; this was **not** the judge's weights. The actual
ratio depends on supplied weights. Decoding uses integer instructions, and
the GEMM schedule changes too. The measured result above demonstrates that
byte savings alone did not make this implementation faster.

Local checks passed all six CPU format tests, including exhaustive compact
and raw round trips of all 65,536 bit patterns, and mixed-block offsets. Static
checks covered 1,020 GEMM schedules, verified exact K coverage without crossing
a compression-block boundary, and checked the unchanged fallback code. The
judge subsequently compiled the GPU implementation and passed its matrix
correctness checks. The separate exhaustive GPU codec test has not been
recorded as run. For deeper correctness validation on an H200:

```bash
python test_lossless_codec.py --gpu
python validate.py submission_triton_lossless.py --nearby
okc check submission_triton_lossless.py --json
```

The GPU codec test compares the actual Triton packer's payload and metadata
against an independent host encoder, then checks the actual GPU decoder bit
for bit. The matrix validator separately checks reference accuracy and fresh
activations under CUDA graph replay. Neither test helper is imported by the
submission. To reproduce the historical judge check from the workspace root:

```bash
okc check muse_bf16_proj/submission_triton_lossless.py --json > muse_bf16_proj/submission-5
```

## Public Guild research, 2026-10-24

Reviewed indexed public copies of [Lurrobert's team
posts](https://guild.1nterface.ai/teams/Lurrobert) and the
[meta feed](https://guild.1nterface.ai/r/meta). Direct access to the requested
[thread 36](https://guild.1nterface.ai/t/36) failed through the web tool and
the shell's DNS resolver. Its exact text and comments remain unread. The
team's power-cap and prepared-state posts were marked as requiring registered
agents; their titles alone are not evidence of their detailed conclusions.
Cached profile snapshots disagree on ranks, so no current rank is inferred.

Lurrobert's public post "Donation from Lurrobert: 11 dead ends and judge
gotchas from day 1" reports these useful findings from their H200 clone:

- Prepared tensors are cloned; pointer-bearing descriptors must follow the
  state supplied per call. Our current candidates pass tensors directly.
- Alternate A/B measurements on one GPU and apply the workload's score formula.
- Their cross-kernel prefetch and MoE copy-ring experiments lost. The latter
  does not establish BF16 behavior.
- A separate projection post excludes a slower BF16 split-K experiment.

These findings make descriptor lifetime and controlled measurement priorities
for our next implementation, rather than evidence that a particular kernel
design will win here.

An adjacent QuantIntellect post in the meta feed advertises a CPU-only
`sm_90a` compilation workflow with resource reports. That is particularly
relevant to our lossless regression, but it is their claim, not a toolchain
we have installed or run here. Triton's own [compiler
source](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py)
supports explicit compilation targets and assembly artifacts; runtime
`n_regs`/`n_spills` attributes are populated when a binary is loaded, so a
compile-only audit must obtain resource usage from compiler/assembler
reports rather than assuming those runtime attributes exist without a GPU.

Application to this project: retain selective as the measured baseline, audit
the lossless kernel's generated PTX/SASS and assembler resources, then measure
decode throughput separately from GEMM before combining them again. Verify
that any future TMA descriptor follows cloned state. Use same-GPU paired graph
measurements and the full score formula to choose a candidate. These are our
next-step conclusions, not performance results attributed to the forum.

## Validate in Colab

Use a GPU runtime with compute capability **8.0 or newer** (for example A100 or
L4). A T4 cannot run these BF16 tensor-core paths. CUDA builds for the local GPU,
so it can also be checked on supported GPUs other than H200. Performance results
on a different GPU will not determine the challenge score.

Copy the candidate files, `validate.py`, `reference.py`, and `checks.py` into the
same Colab directory. The CUDA candidate needs PyTorch, a CUDA toolkit with
`nvcc`, and the extension build tools. The Triton candidate needs PyTorch and
Triton. Those dependencies are provided by the challenge image; check your
Colab runtime separately.

Run from that directory in notebook cells:

```python
!nvidia-smi
!python validate.py submission_triton.py
!MAX_JOBS=1 python validate.py submission_cuda.py
```

`MAX_JOBS=1` limits extension compiler memory usage. To check additional shapes
and another set of weights:

```python
!python validate.py submission_triton.py --nearby --seed 27641
!MAX_JOBS=1 python validate.py submission_cuda.py --nearby --seed 27641
```

The validator covers the nine specified correctness cases and all thirty scored
shapes, deduplicating overlaps. It compares against the supplied FP64 reference,
including row and probe checks. Every case also runs under CUDA graph capture
and replay, then replays again with fresh activations at the captured addresses.
It does not attempt to reproduce the judge's timing or score.

## Check on H200

After the rules have been accepted through the normal `okc login` flow:

```bash
okc problems --json
okc check submission_triton.py --json
okc check submission_cuda.py --json
```

Each file keeps the required problem marker on its first line. Check runs spend
the check allowance; no scored submissions have been made. Inspect each passing
run's profile before changing layouts, split counts, or tile sizes. Use the
judge's cold-L2 CUDA graph timings to decide which changes help.

Implementation references: [CUDA WMMA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html),
[Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html),
and [PyTorch C++ extensions](https://docs.pytorch.org/docs/stable/cpp_extension.html).
