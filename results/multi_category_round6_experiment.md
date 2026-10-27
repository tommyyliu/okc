# Round six: explicit Hopper producer/consumer GEMM

Status: user-run check **34425ce43f3e4f6c** passed GPU compilation and all nine
correctness tests, including QKV512 outliers. Measured **4168.94 us**, **1.15876x**
(95% interval 1.14138–1.17614x). Reject all four replacements at these settings.
PDL remains the measured default. The check was run by the user.

The completed combined check was run from `muse_bf16_proj/`:

```bash
okc check build/multi-category-round6/candidate.py --json \
  | tee build/multi-category-round6/check.json
```

The candidate preserves selected settings on 26 scored cases, including all four
successful GEMV8 changes. `build/multi-category-round6/control.py` reproduces the
selected round-five isolated `gemv8.py` dispatch. Its estimated 3830.27 us aggregate
is assembled from per-case historical measurements, not a measured clean run.
No separate control check is requested for this round.

| Layer | M | Logical BM/BN/BK | Split K | CTAs | Buffers |
|---|---:|---|---:|---:|---:|
| qkv_gate | 128 | 64/64/128 | 2 | 144 | 2 |
| qkv_gate | 256 | 64/64/128 | 1 | 144 | 2 |
| pooler | 128 | 64/64/128 | 1 | 128 | 2 |
| pooler | 256 | 64/64/128 | 1 | 256 | 2 |

One producer warp issues two TMA transfers per stage. Four compute warps consume
BF16 shared-memory operands with FP32 WGMMA accumulation. Hopper rounds the CTA
to eight warps. `maxnreg=96` bounds the CTA's register allocation; the emitted resource record reports 96 registers, zero spills and 73764 bytes
of shared memory. The two operand buffers require 65536 bytes plus
32 bytes of barriers, before compiler scratch. The compiler stage setting is one;
the explicit ring, rather than TL software pipelining, schedules the work.

Each slot has empty/ready phase barriers. Ready requires both TMA transfers to
complete. The consumer waits for WGMMA completion before releasing a slot, then
converts its final accumulator layout for contiguous output stores. QKV128 uses
the existing PDL reducer on FP32 partials; other scored changes store BF16 directly.
The normal producer signals its dependent after the final MMA wait and before
partial stores. The reducer retains its dependency wait and PDL launch.

This tests an engine redesign: previous TMA experiments already used TMA and
WGMMA, but did not explicitly assign a separate producer warp. Waiting on each
MMA limits compute overlap while making buffer reuse explicit. If it wins,
remaining options include additional buffers, deeper MMA overlap and wider tiles.
If it loses, resource counts and separate QKV/pooler results help distinguish
occupancy, scheduling and split overhead. This four-case probe alone is not
expected to close the full gap to 2.95 ms: a 20% reduction in this group would save
about 141 us in the historical aggregate sensitivity estimate.

The same family covers nearby QKV/pooler M>32 with masked stores and zero-padded
activation TMA loads, including the QKV512 outlier correctness case. K partitions
remain disjoint and exhaustive. Weight preparation uses original BF16 weights;
only unpooler retains a prepared transpose. Descriptors bind to the current
activation and prepared weight tensors at each call.

The implementation follows pinned Triton 3.8 APIs:
[warp specialization](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/08-warp-specialization.py),
[WGMMA](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/05-wgmma.py),
[host descriptors](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/nvidia/hopper.py).

CPU validation covers all preset launch manifests, unchanged old GPU bodies and
launches, exact control dispatch, current descriptor identities, K partitions,
odd-size output addresses, and 560 randomized producer/DMA/MMA schedules. These
checks cannot establish GPU correctness. Local ahead-of-time compilation was attempted before the remote run and
reported `No module named 'triton'`; no Torch, Triton or GPU is installed locally.

`OKC_RESOURCE` lines report scalar compiler metadata once per compiled metadata
signature for custom kernels, including unchanged small GEMMs and reducers.
No tensor values or timing probes are printed. Preserve the check's adjacent
`profiles/submission.<job>.log` as well as JSON: it contains resource diagnostics.
The normal presets keep this logging disabled. The remote compiled metadata reports eight total warps for the new kernel;
the configured default compute partition remains four warps.

The compile auditor now extracts Gluon imports, constructs layout-qualified TMA
types and uses GluonASTSource. For partitioned PTX, it checks the dependency
signal's presence rather than interpreting linear instruction text as execution
order; the source protocol needs review alongside the generated control flow.

Rebuild:

```bash
python -m tools.build --preset hopper_ws --output build/multi-category-round6/candidate.py
python -m tools.build --preset banked --output build/multi-category-round6/control.py
python -m tools.codegen --source build/multi-category-round6/candidate.py \
  --layers qkv_gate pooler --tokens 128 256 \
  --output build/multi-category-round6/codegen
```

## Completed result and interpretation

| Case | Previous selected latency, normalized to this run | New latency | Normalized regression |
|---|---:|---:|---:|
| pooler M128 | 22.98 us | 36.89 us | +60.6% |
| pooler M256 | 29.91 us | 40.39 us | +35.0% |
| qkv_gate M128 | 17.73 us | 23.14 us | +30.5% |
| qkv_gate M256 | 20.16 us | 35.43 us | +75.7% |

Comparisons use the round-five official paired ratios against each run's own
SGLang baseline; predecessor configurations match these four selected settings.
These are cross-run comparisons, not paired A/B measurements within one run.
Seventeen unchanged configurations versus round five move by a median +0.53%,
range -4.36% to +5.52%, much smaller than the four new regressions.

The new engine also trails the old unsplit pooler TMA prototype by 12.1% at M128
and 7.3% at M256 on normalized ratios. Explicit producer specialization by itself
has not improved the previous TMA engine. The successful barrier/numerical check
is useful infrastructure validation, not a performance win.

All emitted resource variants report zero spills. The new 73764-byte shared
allocation is below the old large TMA audit's roughly 98 KiB; those facts do not
establish achieved occupancy or memory bandwidth. The resource logger deduplicates
by kernel name and resource tuple, so the new kernel appears once, first at the
QKV512 correctness test, rather than once for every shape.

QKV128 profiling attributes about 21.0 us (91%) to the producer and 2.1 us (9%)
to its reducer. Pooler128, QKV256 and pooler256 have no reducer and still lose.
The primary problem is therefore the producer path, not just split-K overhead.
The consumer drains all outstanding MMA after every K tile. This is a concrete
restriction on overlap; how much latency it causes remains unmeasured. Warpgroup
allocation and shallow grids are additional hypotheses, not proven bottlenecks.

Next priority: a consumer with multiple MMA groups in flight and safe delayed
buffer release, followed by a separately identifiable split-K scheduling ablation.
Do not widen this version to the small GEMMs or unpooler. Retain the banked
control and avoid another check until the next candidate changes those mechanisms.

Replacing just these four timings with their predecessor ratios on the current
run's baseline projects **3854.64 us**. This is a counterfactual calculation, not
a measured rollback. The earlier clean estimate was 3830.27 us; neither should
be presented as an achieved aggregate score.

Raw evidence: `multi_category_round6.json`, `multi_category_round6.txt`,
`multi_category_round6.log`. Calculations and resource records:
`multi_category_round6_analysis.json`. Candidate/control sources in the round-six
build directory remain unchanged from the checked candidate.
