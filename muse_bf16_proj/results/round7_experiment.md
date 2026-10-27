# October 26: autonomous BF16 projection experiments

The selected candidate passed at **3764.88 us / 1.2831221x**, versus the fresh
banked control's **3830.40 us / 1.2611733x**: **65.52 us / 1.71% less score
latency**. This is a promising point estimate. Their 95% speedup intervals
overlap: selected [1.2638753, 1.3023690], control [1.2422557, 1.2800909].
The selected run retains five changes relative to the banked control; twenty-five
scored configurations are identical. No scored submission was made, and PDL
remains the historical default.

Eight remote checks were launched: one frontend failure and seven complete
passing runs, including the fresh control and selected confirmation. The final
quota query reported **5 checks and 10 scored submissions remaining**.

## Measured runs

| Check | Job | Score, us | Speedup | Outcome |
|---|---|---:|---:|---|
| Cluster frontend retry | `c0c78fb3324a4b2e` | — | — | Gluon store-layout equality failure; fixed locally |
| Fresh banked control | `6ddee3c328e34835` | 3830.40 | 1.2611733x | Passing measured reference |
| Corrected cluster BK256 | `5e18ccab93f940f7` | 3980.10 | 1.2137396x | Correctness passed; reject |
| Cluster BK64 | `e40c0396e29b49a0` | 4187.91 | 1.1535114x | Correctness passed; reject |
| Six small TMA replacements | `0f311fd4ab4342af` | 3824.02 | 1.2632767x | Correctness passed; small mixed changes |
| Overlapping large WGMMA | `c02532ce190544f5` | 4582.38 | 1.0542127x | Correctness passed; reject |
| More splits / GEMV16 | `ed6f332cdf94424b` | 3865.96 | 1.2495731x | Correctness passed; retain only router M1 signal |
| Selected confirmation | `f5ab0410cf1e4449` | **3764.88** | **1.2831221x** | All 9 correctness tests and 30 timed cases pass |

Raw JSON, readable timings/profiles and resource logs are retained alongside
this report. [The manifest](round7_manifest.json) records local source paths,
local SHA256 values and the service's separate content hashes.
[Per-case comparisons](round7_analysis.json) contain every unchanged and changed
case, rather than only favorable observations.

## Selected behavior

| Case | Change from banked | Control, us | Selected, us | Normalized latency change |
|---|---|---:|---:|---:|
| Router M1 | GEMV 8 to 16 warps | 4.649 | 3.805 | -17.9% |
| Router M8 | Reducer B256 to B32 | 6.293 | 5.569 | -10.5% |
| Router M16 | Reducer B256 to B32 | 6.590 | 5.674 | -13.8% |
| Router M32 | Reducer B256 to B32 | 6.665 | 6.327 | -6.7% |
| QKV M32 | Descriptor loads / WGMMA, same tile and splits | 13.308 | 12.231 | -8.9% |

Normalized change is `control.paired_ratio / selected.paired_ratio - 1`, using
each run's official median paired speedup against its own SGLang baseline.
These are cross-run comparisons, not paired A/B measurements between candidates.

Router M1 first improved 12.1% in the occupancy trial, then 17.9% in confirmation.
QKV M32 first improved 3.8% in the six-case TMA trial, then 8.9% in confirmation.
The direction repeated; the magnitude varied. B32 improved the three small
router cases by 5.3–8.8% in the small-TMA run and again in confirmation.

The selected run's twenty-five unchanged configurations shift by median -0.53%,
range -8.94% to +3.77%. Aggregate movement therefore includes run variation.
A diagnostic replacing only the five changed cases by prior control ratios,
while retaining selected-run baselines and unchanged timings, gives 3812.02 us.
That order-dependent cross-run counterfactual is not an independently measured
control or a causal attribution. It is retained in the analysis JSON.

The exact checked source is [round7_selected_source.py](round7_selected_source.py).
It is an immutable measurement artifact; edit `projection/` for further work.
Rebuild the current implementation with:

```bash
python -m tools.build --preset round7 --output build/agent-round7/selected-current.py
```

The selected preset preserves BF16 weights, FP32 accumulation and router hi/lo
arithmetic. Router M1 still reads the original FP32 weights. The QKV TMA nearby
policy covers 16 < M <= 32; other nearby dispatch retains `router_reduce`.
No state derived from activations is cached.

## Rejected probes and what they establish

**Cluster-local router reduction.** The final store uses the reduction's
`SliceLayout`, avoiding the frontend equality failure. Batched MMA requires the
rank-three instruction shape `[1, 16, 8]`; the old `[16, 8]` aborted compilation.
Both corrected kernels compile locally and pass remote correctness, including
router M9 outliers. Generated PTX contains tensor-core instructions, cluster
barriers and remote shared-memory accesses, with no grid dependency.

BK256 router M8/M16/M32 was 110.2%/69.9%/96.7% slower than the exact checked B32
predecessor. BK64 lowered register counts from 150/192/254 to 78/78/100 with no
reported spills, but ran still slower, at 16.37/17.19/19.43 us. This rejects both
tested schedules. Lower register counts did not rescue the cluster design.

**Small TMA.** The six-case trial holds BM/BN/BK, K splits, stages, FP32 scratch,
reducer geometry and output layout fixed relative to the selected small
weight-major GEMMs. It changes operand supply and the resulting tensor-core
path. QKV M8/M16 moved +0.8%/+2.4%; pooler M8/M16/M32 moved -0.3%/-0.1%/-1.2%.
Only QKV M32 had a sufficiently strong initial signal to enter confirmation.
The nearby pooler M7 correctness test also exercised TMA successfully.

**Deeper WGMMA pipeline.** `hopper_overlap` retains one outstanding MMA group,
releases the preceding operand slot only after its group retires, and drains the
last group before output stores. The PTX contains a loop `wait_group 1` and final
`wait_group 0`, with no per-iteration `wait_group 0`. Shared memory remains
73764 bytes, 96 registers and zero reported spills, matching the earlier
warp-specialized prototype's resource allocation. The four large paths were
43–62% slower even than that earlier failed prototype. The increased overlap
alone is not a viable improvement at this geometry. The implementation follows
the async accumulator primitives in the
[pinned Triton WGMMA tutorial](https://github.com/triton-lang/triton/blob/v3.8.0/python/tutorials/gluon/05-wgmma.py).

**Scheduling.** Doubling K splits from QKV 8 to 16 and pooler 4 to 8 slowed all
six target calls by 6.9–12.8%. Sixteen-warps GEMV changed QKV/pooler M1 by only
-1.2%/-1.8%, but slowed unpooler M1 by 8.1%. Retain eight warps on those three
cases. The router M1 result is the distinct repeatable signal.

## Validation and next useful experiment

All **53 CPU tests pass**, including randomized interleavings of operand loads,
DMA completion, two outstanding MMA groups and delayed buffer retirement.
Compiler manifests match runtime launches for every preset. The generated PDL
file is current, and `git diff --check` passes. CPU tests do not establish GPU
correctness; the selected remote check establishes the nine required tests and
all thirty timed cases. Additional local nearby GPU validation remains unavailable.

Python 3.13 and Triton **3.8.0** are installed in the ignored
`build/compiler-env` environment. This supports Hopper compilation and assembly
audits without a GPU. Torch and a local CUDA runtime are absent.

Preserve the selected candidate as the current experimental reference. A useful
next probe is a bounded tile/pipeline sweep for the QKV M32 TMA path, using this
measured configuration as control. Keep pooler on weight-major and avoid another
cluster or large warp-specialized check without a different, audited schedule.
