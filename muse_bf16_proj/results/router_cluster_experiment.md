# Single-kernel router with cluster-local reduction

Status: **compiled and GPU-correct, but rejected on performance**. The corrected
BK256 candidate, job `5e18ccab93f940f7`, passed all nine tests and thirty scored
cases at **3980.10 us / 1.21374x**. Small-router normalized latency regressed
70–110% against the exact B32 control. BK64, job `e40c0396e29b49a0`, also passed
but was slower still, at **4187.91 us / 1.15351x**. Keep the two-kernel B32 path.

The initial membership failure was fixed, then check `c0c78fb3324a4b2e` exposed
a Gluon layout equality error at the store. The working kernel retains the
reduction's `SliceLayout` for output addressing and uses the rank-three MMA
instruction shape `[1, 16, 8]`. A two-dimensional instruction shape instead
aborted the local compiler. Triton 3.8.0 is now installed in
`build/compiler-env`, allowing CPU compilation before remote checks.

The local audits confirm tensor-core MMA, cluster barriers, distributed shared
memory accesses and no PDL dependency. BK256 used 150/192/254 registers at
M8/M16/M32; BK64 reduced these to 78/78/100, with no reported spills, but lost
more time. Lower register counts alone did not improve this schedule.
Retained results are `router_cluster_fixed.*` and `router_cluster_bk64.*`;
see [the complete new comparison](round7_experiment.md).

Historical staging and original design notes follow.

The original [failure result](router_cluster_failed.json),
[compiler log](router_cluster_failed.log), and
[duplicate retry response](router_cluster_duplicate.json) are retained.
The exact failed source is `build/router-cluster/failed-candidate.py`; the
historical `candidate.py` includes the syntax fix. The staging **51 CPU tests passed**, with
a regression check against the unsupported JIT membership expression.
The checked working sources are `build/router-cluster/layout-fixed.py` and
`build/router-cluster/bk64.py`. Command networking is now available.

Run from `muse_bf16_proj/`:

```bash
okc check build/router-cluster/candidate.py --json \
  | tee build/router-cluster/check.json
```

The exact previously checked source is preserved at
`build/router-cluster/control.py`: byte-identical to job `3f90adf02bb14808`,
which passed at 3848.12 us. That is the primary control for this experiment.
The older keepers result was 3838.82 us; the newer control is not an established
overall improvement over it. See [the measured comparison](router_reduction_vs_keepers.json).
No previous checked candidate has been overwritten.

## Mechanism

The last change improved reducer geometry while retaining both launches. This
candidate replaces the complete small-router path with **one cluster kernel**:

1. Sixteen logical programs each own sixteen experts and all tokens in one
   token tile. Each program is a cluster of eight CTAs, giving **128 physical
   CTAs** on every targeted scored shape.
2. Each CTA owns a different contiguous K1024 slice. It loads BF16 high/low
   weights and BF16 activations in K256 tiles, with **four loop iterations**.
3. Each CTA uses tensor-core MMA v2 for the two products, with independent FP32
   high/low accumulators. It adds the two FP32 components after the K loop.
4. A Gluon sum reduces the logical K-slice axis, explicitly sharded across the
   cluster. The intended lowering exchanges partials through distributed shared
   memory with cluster synchronization, then stores final FP32 outputs.

This removes the global partial buffer, its producer/reducer traffic, PDL
dependency and separate reducer launch. It retains disjoint K work and avoids
the four-block unsplit grid. It also retains tensor-core arithmetic, unlike the
failed direct-FP32 experiment.

The intended compiler mechanisms follow the
[Gluon multi-CTA documentation](https://triton-lang.org/main/getting-started/tutorials/gluon/multicta.html),
with [Triton 3.8 layout APIs](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/_layouts.py)
and [MMA v2 API](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/experimental/gluon/language/nvidia/ampere/__init__.py).
[NVIDIA's Hopper guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
describes thread-block clusters and distributed shared memory. These support
the design primitives; they do not establish that this exact rank-three batched
MMA/cluster-reduction composition compiles or runs correctly. That remains a
required validation gate in the judge's pinned Triton 3.8 environment.

## Scope and unchanged behavior

Only scored router **M8/M16/M32** changes relative to the checked B32-reducer
candidate. Nearby `1 < M <= 32`, including M9 outliers, uses the same kernel.
M1, M>32, and all other projections retain the checked control's dispatch.
All 27 other scored calls match. Existing GPU kernel bodies and preparation
are unchanged. PDL remains the default source configuration.

| M | Token tile | Expert tile | K tile | CTAs per cluster | Clusters | Global scratch |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 16 | 256 | 8 | 16 | 0 |
| 16 | 16 | 16 | 256 | 8 | 16 | 0 |
| 32 | 32 | 16 | 256 | 8 | 16 | 0 |

CTA layouts explicitly place the K-slice dimension on cluster-ID bits. The
reduction drops that dimension, leaving replicated output bases; compiler
store ownership must prevent redundant physical stores. This, cross-CTA
barriers, and distributed-shared-memory accesses need GPU/compiler validation.
There are no global spin loops, persistent counters or cached activation data.

## Upside and risks

The primary control's small-router times are 5.51/5.97/6.29 us. A useful local
goal is **4–4.5 us**, not a forecast. Making all three 4.5 us would conditionally
save about **47 score us**; making them 4 us would save **64 score us** (1.7%).
This three-case probe still cannot produce a large overall speedup alone.
The purpose is to establish a reusable architecture before extending scope.
Conditional calculations are in [router_cluster_plan.json](router_cluster_plan.json).

The reduction is not free: cluster scheduling, barriers and remote shared-memory
access may cost more than the removed launch. Narrower expert tiles create more
logical activation loads, although cache can satisfy reuse. Explicit Gluon MMA
v2 and the short non-pipelined K loop may also lose to the existing producer.
Register/shared-memory resource counts do not by themselves identify the
bottleneck. A failure would reject this configuration, not every cluster design.

## Validation and comparison

All **51 CPU tests passed**. Tests cover the 27 unchanged calls, one-launch /
one-output-allocation behavior, current cloned pointers, exact disjoint K
coverage, actual address/mask expressions on odd tiles, partitioned FP32 hi/lo
arithmetic on probe and outlier inputs, and compiler-manifest/launch agreement.
They cannot establish physical CTA store ownership or GPU synchronization.

Local compiler auditing was attempted and returned `No module named 'triton'`.
No compatible compiler could be installed in the local Python environment.
`build/router-cluster/codegen/report.json` records the planned jobs and failure.
The new audit propagates `num_ctas`, checks tensor-core instructions, cluster
barriers and remote shared accesses, and rejects an unexpected PDL dependency.

```bash
python -m tools.build --preset router_cluster --output build/router-cluster/candidate.py
python -m unittest discover -s tests -v
python -m tools.build --check

# With Triton available; no GPU required:
python -m tools.codegen --source build/router-cluster/candidate.py \
  --layers router --tokens 8 9 16 32 --output build/router-cluster/codegen
```

First establish compilation and the judge's nine correctness tests, including
M9 outliers. If the candidate passes, compare official per-case ratios against
the exact checked control, inspect the 27 unchanged cases for variation, and
read complete kernel profiles/resources. The intended single-kernel path must
actually appear in the profile. Confidence intervals and repeated normalized
gains still matter; do not subtract a prior reducer profile from scored time
and present the difference as an achieved gain.

If it wins, the next extension is a separately controlled larger-router probe.
If it loses, inspect resource counts and the combined kernel time before
changing cluster size or tile geometry. Keep the previous passing sources.
Candidate/control hashes and launch plans are in both
`build/router-cluster/manifest.json` and `results/router_cluster_plan.json`.
