# Projection diagnostics and the 1.61x target

The target is the user's reported leading speedup, **1.61x**, not a live rank
query. Our best measured candidate is selective, **1.2048418x**, with a score
of **4009.49 us**. Under the judge's fixed 4830.8 us baseline calibration,
1.61x corresponds to **3000.50 us**: **25.17% less latency**, or **33.63% more
throughput**, relative to selective.

`projection_budget.py` reconstructs the task's weighted times and geometric
mean from `submission-4`. Its report is `profiles/target_budget.json`.
Removing all router time is an impossible oracle that reaches only 1.3552x.
Putting any one layer at its reported floor also falls short of 1.61x; the
largest such gain is pooler at 1.3571x. Putting every shape at its reported
floor gives 2.0114x. These are mathematical scenarios, not kernel performance
claims; lossless compression can change the weight-traffic component of a
reported uncompressed floor.

## What is known

Submission 5 passed matrix correctness but took 11.19x selective's score
latency. The measured regression is inside the compressed projection kernels.
Submission 6, diagnostic check `c45c07be9e424ba8`, now supplies compiler-resource
metadata for the same GPU work. It passed all nine correctness tests and
thirty scored shapes, without integrity flags, at **0.1067704x** and
**45244.74 us**. Its 95% interval, **[0.105169, 0.108372]**, overlaps the
original lossless result as expected.

The retained log contains 31 complete records representing 16 unique GEMM
variants. Earlier single-token and 16-token metadata was truncated. The
extracted evidence is in `profiles/codegen-h200/runtime_report.json`.

| Layer / tokens | Registers per thread | Raw spill report | Shared bytes | Measured time |
|---|---:|---:|---:|---:|
| Pooler / 32 | 255 | 6 | 32768 | 325.08 us |
| Sliced unpooler / 8 | 255 | 42 | 24576 | 26.24 us |
| Pooler / 128 | 193 | 0 | 114688 | 204.36 us |
| Pooler / 256 | 255 | 0 | 114688 | 702.06 us |

All retained 8/32-token variants report 255 registers/thread, nonzero spills,
and 128 static scalar `ld.global.b8` instructions, plus 64 each of the scalar
16/32-bit global load instructions. Retained 128/256-token variants report
64 byte loads plus 32 each of the 16/32-bit loads. No vectorized global load
opcodes are present in those recorded counts. These are static instructions
in emitted PTX, not hardware transaction counts or executed instructions.

The 255-register cases reach Hopper's per-thread register limit. Using the
reported registers and warp counts with Hopper's 64K-register SM budget gives
an upper bound of eight resident warps, or 12.5% of its 64-warp capacity, for
these variants. This is a resource-derived bound, not measured occupancy;
allocation rounding and other limits can lower it. The hardware limits are
documented in [NVIDIA's Hopper tuning
guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy).

The evidence confirms scalar load code and heavy register use. It does not
show that spills alone cause the regression: the slow 256-token pooler has
zero reported spills. The next decoder design should use word-packed loads,
explicit alignment and fewer live decoding temporaries. Decoder-only and
matching-raw-tile comparisons remain necessary to attribute the full cost.

The actual judge environment is now known: **Triton 3.8.0**,
**PyTorch 2.14.1+cu130**. Compile-only comparisons should use that Triton version.

Source inspection shows two stages of weight reads: offset/base metadata,
then payload bytes whose addresses depend on that metadata. The exponent
address also depends on the block's format. The original decoder does not
declare its guaranteed 128-byte block alignment to the compiler. These are
facts about the source. Submission 6 confirms scalar load code and some
spilling; pipeline serialization still requires further evidence.

This workspace has no CUDA GPU and no installed Triton. Cached Torch CPU
packages exist, but the compiler is not cached. DNS failed for both package
hosts and the challenge host. `profiles/codegen-h200/report.json` therefore
records a blocked audit with zero compiled jobs;
`profiles/lossless_benchmark_attempt.json` records a blocked benchmark with
zero measured cases. The local diagnostic check attempt failed before reaching
the server; its receipt is under
`profiles/submission_triton_lossless_diagnostic.check.*`. The subsequent user-run
check supplied the real compiler metadata above. No full PTX/SASS dump or
standalone decoder timings have been returned yet.

## Get resources from the real H200 check

`submission_triton_lossless_diagnostic.py` retains submission 5's preparation,
GPU kernel bodies and launch settings. It adds host-side diagnostic output
after the compressed projection launch. It prints compiler metadata and static
PTX memory-opcode counts; it prints no tensor values and adds no GPU work.
The task explicitly permits diagnostic printing. This is a diagnostic check,
not a proposed speed improvement.

From the workspace root:

```bash
okc check muse_bf16_proj/submission_triton_lossless_diagnostic.py --json > muse_bf16_proj/submission-6
```

Then, from `muse_bf16_proj`:

```bash
python inspect_codegen.py --read-result submission-6
```

This extraction needs only Python and writes
`profiles/codegen-h200/runtime_report.json`. It deduplicates retained metadata
records and counts occurrences. The judge retains only the last part of a run
log, so earlier shapes may be absent. Runtime `n_spills` is recorded under
`n_spills_raw`; do not infer byte counts from its name.

## Compile-only audit, without a GPU

On a CPU machine with Triton and its bundled NVIDIA assembler installed:

```bash
python inspect_codegen.py --plan
python inspect_codegen.py --output profiles/codegen-h200
```

The fourteen planned jobs compare raw and compressed GEMV, raw and compressed
GEMM with matching tiles, and four decoder/copy probes. Extraction includes
only local JIT definitions; it does not execute candidate preparation or import
Torch. The target is explicit Hopper `cuda:90:32`, which the NVIDIA backend
assembles for `sm_90a`. Use the judge's compiler version when available; version
and source hashes are recorded rather than assuming different versions agree.

Artifacts include TTIR, TTGIR, LLVM IR, PTX and cubin. The report counts static
global/local/shared memory operations, MMA, asynchronous copies, shuffles and
IR layout conversions. These counts describe code size, not executed work;
loop bodies are counted once, and assembler-introduced spills need not appear
as PTX local loads/stores.

If `cuobjdump` is available, the tool saves SASS and resource usage from the
original cubin; `--cuobjdump /path/to/cuobjdump` can select it. `nvdisasm` is an
optional SASS fallback. A separate verbose ptxas reassembly reports stack and
spill bytes, with its exact command logged. Those resource numbers are labeled
as reassembly diagnostics because backend flags can differ. Original cubin
resources are preferred. Failed stages leave an incomplete report, not success.

This workflow follows Triton's [AOT compilation
tool](https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py)
and [NVIDIA compiler
backend](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py).
The normal JIT launch returns its compiled-kernel handle, as shown in
[Triton's runtime](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/runtime/jit.py).

## Separate decoder and GEMM costs on H200

```bash
python bench_lossless.py --plan
python bench_lossless.py --layers pooler --tokens 16 256
```

This writes `profiles/lossless_benchmark.json`. It checks decoded weights bit
for bit, matrix outputs against the reference, and graph replay with fresh
outlier activations before timing. Every measurement uses a cold-L2 CUDA graph
with the cache eviction excluded from timing. ABBA order compares paths on the
same GPU. These are development measurements, not full judge reproduction.

The paths are:

- Raw matrix copy, to measure read/write throughput without decoding.
- Original full-matrix decoder, including its output write.
- Aligned decoder: the original with an explicit block-alignment hint.
- Per-block branch decoder: reads each packed exponent byte once, and uses a
  single format branch per program. It also changes program granularity; use
  `--groups-per-cta 1` to match the original decoder's number of programs.
- Selective projection, the measured baseline's implementation.
- Raw projection with exactly the compressed kernel's tile and split settings.
- Original fused compressed projection.
- Original decoder followed by cuBLAS, both inside the measured call.

Interpret the comparisons together. A slower matching raw GEMM implicates
scheduling independently of compression. A slow decoder isolates a problem
before tensor-core work. A much faster staged path would motivate inspection
of fusion layout/pipelining. A decoder writes an entire BF16 weight matrix,
while the fused projection does not, so their absolute times are different
workloads. The alignment and branch variants remain GPU-untested probes.

## Local verification

```bash
python -m unittest test_projection_diagnostics -v
python projection_budget.py --target 1.61
```

The twelve diagnostic tests passed: weighted-score reconstruction, invalid
input handling, assembly/resource parsers, truncated-log parsing, isolated JIT
extraction, complete AOT signatures, matching raw/compressed launch settings,
and preservation of all GPU work in the diagnostic variant. This validates the
diagnostic plumbing. The later compile-only audit below compiles the GPU probes;
it does not validate their results or timing on a GPU.

## Actual local compilation with the judge compiler

Network access was restored during the short research session. A temporary
Python environment with Triton **3.8.0** compiled all fourteen jobs for
`sm_90a`; `profiles/codegen-triton380/report.json` contains the source hashes,
compiler metadata and generated-code summaries. No GPU was used.

| Matching projection | Raw registers | Compressed registers | Compressed spill stores / loads |
|---|---:|---:|---:|
| QKV or pooler / 1 token | 52 | 188 | 0 / 0 bytes |
| Pooler / 16 tokens | 40 | 255 | 24 / 24 bytes |
| Pooler / 256 tokens | 94 | 255 | 0 / 0 bytes |
| Unpooler / 128 tokens | 78 | 193 | 0 / 0 bytes |

These are verbose **ptxas reassembly** resource counts, with the exact command
recorded; original backend flags can differ. They independently support the
judge's resource evidence, without establishing runtime occupancy or speed.
The raw matching GEMM is an experimental control, not necessarily selective's
dispatch choice at that shape.

Compilation exposed an invalid rank-two alignment hint in the aligned decoder
probe. Changing it to `(128, 1)` fixes compilation. Its PTX now contains one
`ld.global.v2.b32`, but eight scalar byte-load instructions remain. The
original decoder has sixteen scalar byte-load instructions. The per-group
branch probe uses 18 registers versus 27 for the original, but changes program
granularity and load paths. Neither result establishes GPU performance or
bit-exactness of these new probes; run `bench_lossless.py` before promotion.

## Short Guild research session

The user authorized a brief research session and invoked the
[Guild skill](https://guild.1nterface.ai/skill.md). After network access was
restored, `bf16-research` was registered and claimed under `tommyyliu`.
The session used foreground heartbeats only; no background heartbeat or cron
was installed. Following the invoked skill, we posted a short
[introduction](https://guild.1nterface.ai/t/2#p43) containing our measured
negative compression result, and sent idea-trade requests to QuantIntellect
and jakozaur. Credentials and raw API responses are outside the repository.
Requests were pending at the final mailbox read.

Useful leads, with other teams' measurements kept separate from ours:

- [BF16 discussion, bra1nDump reply](https://guild.1nterface.ai/t/5#p14):
  reports 7–13% per-case gains for O projection at 8–32 tokens by computing
  `W @ X.T` with row-major weights, small token tiles and no split-K.
  Exact operand/output strides are still an unanswered clarification in the
  thread. The same reply reports 4–9% router gains from programmatic dependent
  launch (PDL), and a separate failed exact BF16 compression experiment.
  Treat layout and PDL as experiments to reproduce, not promised aggregate gains.
- [QuantIntellect compiler report](https://guild.1nterface.ai/t/18#p20):
  reports CUDA 13.0 `ptxas` warning **C7512** when a bounded barrier wait traps
  in a WGMMA kernel, serializing the entire function. Their SASS signature was
  `WARPGROUP.DEPBAR.LE ... 0x0` after each HGMMA and a 168-register allocation;
  replacing the trap with an error path restored pipelining in their kernel.
  This is a specific compiler pitfall, not an explanation established for ours.
- [Streaming experiments](https://guild.1nterface.ai/t/18#p30): early-release
  PDL and larger, fewer bulk copies improved that team's streaming probes.
  Its 18 KB/four-copy stage beat a 9 KB/sixteen-copy stage despite equal bytes
  in flight. These MoE experiments motivate copy-issue and scheduling probes,
  rather than assuming every decode loss is integer arithmetic.
- [Replay race notes](https://guild.1nterface.ai/t/42): barrier lifetime,
  shared-memory proxy ordering and self-resetting counters caused rare errors
  visible only under long sequences of queued graph replays. Any future
  asynchronous implementation needs repeated graph validation with fresh
  inputs, including sequences without a host sync after each replay.
- [Power-cap discussion](https://guild.1nterface.ai/t/12): teams report
  sustained H200 clocks around 1.7–1.8 GHz. Our before/after clock snapshots
  do not measure sustained in-kernel frequency.

We also read the later corrections in thread 18: an inferred double-decode
penalty was retracted after a direct no-gain experiment, and resident-loop
TB/s-equivalent includes repeated processed bytes rather than unique HBM
traffic. Do not use either as a transferable bandwidth claim.

Primary-source check: NVIDIA documents that
[PDL supports CUDA graph capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).
A consumer must wait for producer completion before reading dependent data;
concurrent execution is opportunistic. The installed Triton 3.8.0 provides
`tl.extra.cuda.gdc_launch_dependents`, `gdc_wait` and the `launch_pdl` option.
A candidate must preserve that ordering, and its captured graph and GPU
results still need checking.

Recommended next comparison: preserve selective as the control, test the
small-token O-projection orientation, then isolate PDL on existing split-K
paths. Pause fused compression until decoder benchmarks demonstrate a gain.
The advertised donated-compute queue was empty for our agent; available
workers were not verified, so no GPU job or availability claim follows.
