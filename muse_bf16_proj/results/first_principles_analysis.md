# First-principles performance analysis

The next focused investigation should be the **router, starting with M8/M16/M32**.
Its selected kernels take about 3.5–3.8 times the ideal uncompressed hardware
bound across multi-token sizes. The evidence points toward the cost of exposing
parallelism and combining partial results, rather than a large amount of useful
arithmetic. This is a diagnostic priority; it is not a claim that the whole gap
is recoverable or that router tuning can close the overall target by itself.

No GPU check or new kernel was run for this analysis. The calculations are in
[first_principles_analysis.json](first_principles_analysis.json).

## What the hardware must do

For a BF16 projection `X[M,K] @ W[N,K].T`, the optimistic traffic is

```
bytes = 2*N*K + 2*M*K + 2*M*N
FLOPs = 2*M*N*K
t_ideal = max(bytes / 4.8e12, FLOPs / 989e12)
```

These are seconds, using decimal TB and TFLOPs. The judge uses the same model.
NVIDIA specifies [4.8 TB/s HBM bandwidth for H200](https://www.nvidia.com/en-us/data-center/h200/).
Its advertised 1,979 TFLOP/s BF16 figure includes sparsity; the challenge uses
989 TFLOP/s dense. Inputs here do not qualify for the sparse throughput figure.

The roofline crossover is about **206 FLOPs per byte**. When weights dominate
traffic, BF16 arithmetic intensity is approximately M, so M1/M8/M16/M32 are
strongly bandwidth dominated. All M128 cases remain bandwidth dominated in this
model. At M256, QKV, pooler and full unpooler become compute dominated; O remains
just on the bandwidth side because activation/output traffic matters.

Cold L2 means we must fetch the weights on every invocation. Untimed weight
preparation can improve their layout, but cannot remove this read. Reuse between
blocks *within* the invocation can still come from L2. The bound assumes perfect
reuse, perfect overlap, and no launch, pipeline, synchronization or tail cost.
It is an ideal uncompressed model, not an achievable promise or an absolute
bound on every possible legal lossless representation.

For the router, original FP32 weights or BF16 hi/lo together take four bytes per
weight, and output takes four bytes per value. Hi/lo requires **two** BF16 matrix
products, so its arithmetic is `4*M*N*K`, not `2*M*N*K`. M1 uses original FP32
weights and FP32 scalar arithmetic. Correcting these compute models does not
change any router roofline value: traffic still dominates at all scored sizes.

## How far the selected implementations are from that model

Each cell is **selected latency / ideal latency in microseconds**.

| Projection | M1 | M8 | M16 | M32 | M128 | M256 |
|---|---:|---:|---:|---:|---:|---:|
| QKV | 10.64 / 7.87 | 12.05 / 7.90 | 13.06 / 7.93 | 13.47 / 8.00 | 17.73 / 8.42 | 20.16 / 9.77 |
| O | 5.84 / 3.50 | 6.08 / 3.53 | 6.20 / 3.56 | 6.46 / 3.62 | 7.52 / 3.99 | 9.55 / 4.48 |
| Pooler | 17.27 / 13.99 | 18.76 / 14.02 | 19.31 / 14.06 | 19.78 / 14.14 | 22.98 / 14.64 | 29.91 / 17.37 |
| Unpooler | 4.08 / 1.75 | 5.90 / 1.76 | 17.81 / 14.06 | 18.37 / 14.14 | 19.22 / 14.64 | 24.07 / 17.37 |
| Router | 4.90 / 1.75 | 6.36 / 1.78 | 6.59 / 1.81 | 6.56 / 1.86 | 8.16 / 2.21 | 10.07 / 2.68 |

Provenance: 26 selected cases were measured in the passing
[round-six check](multi_category_round6.txt). The four rejected large QKV/pooler
replacements are excluded: their selected times above apply the matching
round-five paired ratios to round six's SGLang baseline. Those four entries
are counterfactual normalized comparisons, not measurements in that check.
The corresponding selected aggregate is **3854.64 us, hypothetical**. The actual
round-six experiment measured 4168.94 us. No clean aggregate for the selected
combination has been measured.

Small pooler cases are already within about 1.3–1.4 times the roofline; asking for
a dramatic improvement there conflicts with their required weight traffic.
Large QKV remains a substantial opportunity at about twice its floor. Router
is further away proportionally, with a repeated structural reason to investigate.
Small unpooler M8 is also far away, and uses the same split/reduce structure.

Applying the task's weighted load sums and geometric-mean score, rather than
adding shape times, gives an ideal score of **2371.15 us** on the round-six
baseline. Against the composite selected estimate, the 2950 us user-reported
target requires **23.5% less score latency**. The target has not been refreshed
from a live leaderboard.

A useful planning model is `t = t_ideal / efficiency + residual_overhead`.
Efficiency and overhead below are assumptions, uniform across all thirty calls:

| Assumption | Hypothetical aggregate score |
|---|---:|
| Perfect roofline, zero residual overhead | 2371 us |
| 80% of peak, plus 1 us per projection | 3278 us |
| 85% of peak, plus 1 us per projection | 3104 us |
| 90% of peak, plus 1 us per projection | 2949 us |

This does not predict those efficiencies. It shows how aggressive 2950 us is.
One extra microsecond on each of five projections costs 310 us in each weighted
load (five times 62), before translating back to the score. The judge already
subtracts its fixed graph replay cost; do not subtract another four microseconds
from the reported times. Remaining dispatch/gap and kernel costs still matter.

## Why the small router is struggling

Take **M16, N256, K8192**. Its hi/lo weights total 8 MiB; minimum traffic including
activations and output is 8.67 MB. The ideal memory time is **1.81 us**, versus
only **0.136 us** of globally parallel dense BF16 compute for the two products.
Its selected latency is **6.59 us**. At M8, padding the 16-token tile doubles the
useful arithmetic, but even that padded arithmetic remains well below the ideal
memory time. Useful arithmetic alone does not explain the measured latency.

The launch is `BM16/BN64/BK64`, split K 32 ways:

1. Without splitting, there are only `ceil(16/16) * ceil(256/64) = 4`
   independent output blocks. That cannot fill an H200.
2. Splitting creates **128 producer blocks**. Each handles K256 in four K64
   iterations, with independent FP32 high/low accumulation. This exposes work,
   but makes each block's useful main loop very short. Setup and pipeline
   fill/drain become plausible significant costs.
3. Those blocks write **512 KiB of FP32 partials**. A second kernel reads them,
   sums 32 partials per output, and stores the result. It launches just **16
   blocks** at M16; at M8 it launches eight. This is another small GPU operation
   with a dependency and completion boundary.

The scratch read plus write at M16 is only **1 MiB**. At ideal HBM bandwidth that
is about 0.22 us, and scratch may be served through cache. Scratch byte volume
alone is not a persuasive explanation for the reducer's approximately 1.7 us
profile time; launch, dependency, block scheduling and reduction execution are
plausible contributors. Their individual costs have not been measured.

The round-six profile reports about **2.6 us producer + 1.7 us reducer** at M16,
with 37–40% of profiled kernel time in the reducer across M8/M16/M32. Compiler
records show zero spills for the logged router variants. These support examining
coordination before blaming spills. They do not establish achieved bandwidth,
occupancy, or the exact reason the producer stalls.

Crucially, the profile sum is about **4.3 us**, while the official paired score
time is **6.59 us**. They come from different measurement procedures. We cannot
attribute the remaining 2.3 us to one cause, or subtract the profile's reducer
time directly from the official latency to predict a fused implementation.
Likewise, minimum bytes divided by reported latency is an effective bandwidth
proxy, not a measured DRAM throughput counter.

NVIDIA's [CUTLASS explanation of split K](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
describes precisely this tradeoff: split K exposes parallelism for small output
matrices, then requires partial-result reduction. More sophisticated loading
instructions do not by themselves resolve the tradeoff.

## What the previous failures establish

- **BN32/split16 kept 128 producer blocks and halved partial storage, yet lost
  8–16% at M8/M16/M32.** Each block had a longer K loop and different tensor-core
  geometry. Equal block count and fewer scratch bytes did not guarantee a win.
  This was already tried in [round three](multi_category_round3_experiment.md).
- **Direct FP32 eliminated the reducer but lost 68–330%.** That design replaced
  tensor-core reuse with one expert and two tokens per block. It requested the
  weight matrix once per token pair and repeatedly loaded activations across
  experts. At M16 the logical weight loads alone were 64 MiB, versus 8 MiB in
  the current producer. Cache can absorb some of that; these are not measured
  HBM bytes. It also used a much lower arithmetic-throughput path. The
  [round-five result](multi_category_round5_experiment.md) rules out that
  implementation, not every possible single-kernel router.
- **Interleaving hi/lo into a wider dot did not give a broad win**, and lost about
  11% at M8 in [round four](multi_category_round4_experiment.md). Merely changing
  dot packaging has already been tested.

The working hypothesis is therefore **a small-matrix parallelism and coordination
problem**. Its individual components remain to be separated experimentally.

## The next bounded investigation

Keep the current hi/lo tensor-core path as control and study M8/M16/M32 only.
Before constructing another combined candidate, collect three local diagnostic
measurements on H200: a checksum/read kernel with the producer's operand tiling,
the existing producer writing genuine partials, and the complete producer plus
reducer. The read kernel is diagnostic tooling only, never a substitute submitted
as the projection. Match cold L2, graph replay, and fresh-input timing conditions.

Collect DRAM and L2 traffic, active blocks/warps, and main-loop/barrier stalls
where profiling access permits. Measure with instrumentation separately from
timing. Repeat complete-call comparisons in the same session; validate final
candidate timing through OKC's paired procedure.

If reading operands already consumes most of the producer's time, investigate
operand delivery and traffic. If the producer adds substantial time beyond that,
investigate tile setup, layout conversion and the short pipeline. If the second
kernel/dependency dominates the complete-call gap, investigate a tensor-core
design with cooperative local K reduction. That design must explicitly retain
enough independent work; simply moving all K work into four blocks repeats the
parallelism problem. Cluster/shared-memory approaches require proving compiler
support, synchronization correctness and scheduling cost before a full check.

A **4–5 us complete-call target** for these three cases is a useful first
experiment threshold, not a forecast. Reaching 4 us would save about **84 score
us** from the selected composite estimate. Router across all sizes is 11.1% of
score sensitivity; even idealizing the entire router saves only **304 score us**.
This category can teach us how much of the short-kernel overhead is recoverable,
but larger QKV/pooler work must still contribute to the overall target.

There is no local Torch, Triton or GPU runtime in this workspace, so the proposed
diagnostic measurements remain outstanding. The table, bounds and score scenarios
above use retained GPU evidence and CPU arithmetic only.
