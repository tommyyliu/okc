# Architectural redesigns after round five

The clean selected combination projects **3830.27 us**, not yet measured as a
whole. Reaching the user-reported 2950 us target needs about **23.0%** less score
latency. The latest combined experiment measured 4321.59 us because its failed
unroll and direct-router paths outweighed the four GEMV improvements.

Prioritize a reusable dense-GEMM architecture. QKV/pooler M8..256 and full
unpooler M16..256 account for **63.4%** of projected latency sensitivity. A
20% reduction on those fourteen cases, capped at reported floors, saves about
**492 us**, projecting 3338 us. Even taking all fourteen to their reported
uncompressed floors gives about **3021 us**. Another family must contribute to
the 2950 us target; the floor scenario is an optimistic bound, not a forecast.

## 1. Explicit asynchronous Hopper GEMM pipeline — main project

Design independent producer and consumer warp groups. Producers transfer the
next BF16 weight/activation tiles into a shared-memory ring; consumers issue
WGMMA on ready tiles while later transfers progress. Explicit buffer ownership,
barrier phases, asynchronous MMA waits and accumulator lifetimes determine the
schedule. Use separate shape configurations for small bandwidth-dominated M and
larger M, while sharing the implementation.

The earlier TMA prototype already emitted TMA and WGMMA and used compiler
software pipelining. Therefore the hypothesis is improved overlap/resource
allocation from explicit producer/consumer control, not merely adding those
instructions. Archived large-case compiles used about 98 KiB shared memory and
58–60 registers/thread without recorded spills. That does not identify the
runtime bottleneck, and it does not justify attributing every loss to spills.

First prototype scope: **large QKV and pooler at M128/M256**, four cases. Their
20% conditional score saving is about 141 us. Keep the existing selected
implementations for the other twenty-six cases, including eight-warp GEMV.
Cover nearby/irregular shapes using the same masking/descriptor conventions and
exercise the new QKV path on its harder outlier correctness shape if feasible.

After an end-to-end win, extend to six small QKV/pooler cases, then full unpooler.
Those groups add approximately 217 us and 140 us of individually calculated,
floor-capped 20% payoff; savings are not additive. The large target is the shared
fourteen-case engine, not an isolated microsecond on one shape.

Implementation should follow verified Hopper-capable Gluon/WGMMA primitives in
the judge's Triton 3.8 release. The generic `tl.range(warp_specialize=True)`
documentation is not a blanket guarantee of Hopper support. Maintain BF16
operands and FP32 accumulation, binding descriptors to current tensors rather
than saving stale addresses in prepared state.

References: [NVIDIA Hopper warp-specialized GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html),
[pinned Triton 3.8 WGMMA tutorial](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/05-wgmma.py).

## 2. Keep weight panels reusable across token tiles — extension of the engine

At M128/M256, organize work around a weight panel and multiple token tiles.
Candidate mechanisms include larger useful output tiles, persistent tile
scheduling, or TMA multicast to cooperating CTAs that need the same weight tile.
Retain enough independent work to occupy the GPU; a weight-stationary schedule
that launches too few blocks can lose despite less traffic.

The current tile order already places token tiles together, and L2 can satisfy
reuse within a replay. Thus do not assume all repeated logical loads reach HBM.
The useful hypothesis is reducing duplicated operand supply or transition cost,
to be tested by traffic/resource evidence. Every timed replay starts cold, so
the design must create reuse inside the current call.

Cluster multicast is an advanced extension after a competitive single-CTA
pipeline. Hardware support does not by itself establish support for the exact
layout in the available compiler. Reference:
[PyTorch's TMA/multicast explanation](https://pytorch.org/blog/hopper-tma-unit/).

## 3. Tensor-core router with cooperative K reduction — secondary project

Preserve BF16 hi/lo tensor-core products and independent FP32 accumulators.
Distribute K work across cooperating warp groups or clustered CTAs and combine
their FP32 results through shared/distributed shared memory. The intended saving
is global scratch traffic and the second launch while retaining the throughput
lost in the direct FP32 experiment.

The hard constraint is parallelism: router output width is only 256, so simply
giving each output tile a full K loop produces too few independent blocks.
Cluster-based split-K is a possible architecture because Hopper permits shared
memory exchange and synchronization across CTAs, but this needs a proven compiler
path and careful barrier/reduction design. A `num_ctas` setting alone is not a
complete implementation. Coordination overhead may consume the benefit.

All router cases are only **11.0%** of projected score sensitivity. A 20%
reduction saves about **84 us**; even the reported-floor scenario gives only
about 3534 us overall. This cannot be the main route to 2950 us.
Reference: [NVIDIA Hopper clusters and distributed shared memory](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

## 4. Decode an exact compressed representation in the producer — research

Reduce the required bytes per weight while reconstructing every BF16 bit before
matmul. A successful 12-bit representation would move 25% fewer payload bytes
than BF16 before metadata/fallback overhead. The key architectural opportunity
is a dedicated decode producer feeding shared memory while tensor cores consume
previous tiles. Fixed-stride bitplanes or tile-local metadata could remove
variable-offset dependencies from the old decoder's hot path.

The archived lossless implementation already tried sign/mantissa bytes with
packed exponent deltas and measured roughly 0.1076x. Its inline decoder produced
hundreds-of-microseconds kernels. This is strong negative evidence for repeating
that implementation. A new format/decoder needs bit-exact round-trip proof,
fewer actual transferred bytes, and a measured decode cost that can be hidden
or repaid by the saved traffic. Integer decoding can become the bottleneck.

Build the uncompressed pipeline first. Only then investigate this extension;
its upside comes with substantially greater implementation risk.

## Make the next architectural check diagnostic

Keep the banked per-case settings as control and request one full check for the
first four large-GEMM replacements. Record available compiled-kernel register,
spill and shared-memory metadata for new and unchanged custom paths, using
scalar diagnostics outside GPU work. Do not print tensors or add timing-based
dispatch. Actual kernel timing and correctness still come from OKC.

Resource reports and kernel-stage profiles should distinguish a failure to feed
tensor cores from excessive shared memory, register growth or reduction overhead.
Current CPU-only access cannot produce GPU bottleneck counters; compiler
metadata is useful but does not prove memory throughput or tensor-core occupancy.

Use matching predecessor settings and official paired ratios versus same-run
SGLang. Evaluate producer plus reducer together. Preserve a precise source/hash
record and inspect unchanged cases for run variation. Do not promote a pipeline
solely because it uses the desired instruction names.

The calculations are retained in `redesign_budget.json`, using the same load-sum
and geometric-mean scoring model as earlier analyses. All scenarios are
conditional and the reference aggregate remains hypothetical.
