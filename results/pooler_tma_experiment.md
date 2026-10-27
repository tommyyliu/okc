# Pooler TMA experiment — correctness passed, performance regressed

Check `dc36c8ef363340e3` passed all nine preliminary tests and all thirty scored
cases. Aggregate speedup was 1.1739579x, with a 95% interval of
[1.1563485, 1.1915672]. Retained evidence: `pooler_tma.json` and `pooler_tma.txt`.

| Changed case | Candidate | Same-run SGLang | Candidate latency overhead |
|---|---:|---:|---:|
| Pooler, M=128 | 33.46 us | 25.28 us | 32.4% |
| Pooler, M=256 | 36.42 us | 29.86 us | 22.0% |

The profile places the extra time inside `_tma_weight_major` (32.5/36.3 us in
the separate profiling pass). No reducer is launched. The result does not
identify whether memory throughput, pipeline scheduling, or another resource
is limiting. Several unchanged cases also moved across runs, so the aggregate
change cannot all be attributed to this experiment. The PDL default is retained.
The next experiment is documented in [pooler_splitk_experiment.md](pooler_splitk_experiment.md).

## Original experiment design

Replace the measured PDL candidate's cuBLAS pooler path for 33–256 tokens with
the existing weight-major TMA kernel. Only two scored cases change: M=128 and
M=256. Other projections and token counts retain PDL's configuration. The
nearby-size policy uses the same kernel across the entire interval.

| Case | Historical PDL latency | PDL / SGLang speedup | Task floor |
|---|---:|---:|---:|
| Pooler, M=128 | 25.65 us | 1.013x | 14.64 us |
| Pooler, M=256 | 29.64 us | 0.991x | 17.37 us |

Hypothesis: the explicit TMA pipeline and weight-major tiles can improve these
long-K GEMMs. This is a comparison against cuBLAS, not an isolated measurement
of TMA alone. This hypothesis was not supported by the measured result above.

Launch: 64 token slots, 64 output features, K tile 128, four warps, three stages,
no split-K or reduction launch. The grids have 128 and 256 programs respectively.
Weights and activations stay BF16 and accumulation stays FP32. Descriptors bind
to the current tensors on every timed call.

The unchanged kernel and these two configurations were compiled with Triton
3.8.0 for sm_90a in `pre-reorganization:profiles/codegen-tma/report.json`.
That archived audit reports TMA and WGMMA instructions, 58 registers/thread,
98,328 shared-memory bytes, and zero spill bytes in ptxas reassembly. This is
historical compilation evidence, not a new GPU correctness or timing result.

From the project directory:

```bash
python -m tools.build --preset pooler_tma --output build/pooler_tma.py
okc check build/pooler_tma.py --json | tee build/pooler_tma-check.json
```

If a local H200 is available, also run:

```bash
python -m tools.validate build/pooler_tma.py --nearby
```

The judge checks every timed case, so the two changed scored shapes receive
correctness checks even though the nine preliminary tests do not exercise this
pooler range. Local nearby validation includes M=34 and M=127 for the new path.

Inspect pooler M=128 and M=256 first. Compare their candidate/SGLang ratios with
the historical ratios above, since absolute timings vary across runs. Keep a
win per shape; a regression at one does not invalidate a gain at the other.
If both improve, next test tile size and pipeline depth around this configuration.
If neither improves, use the profile and bandwidth measurements to decide
between changing scheduling and abandoning this layout. Aggregate score alone
can hide the outcome because only two of thirty scored cases change.

For scale, a hypothetical 20% reduction in both changed cases gives the score
scenario recorded in `build/pooler_tma-budget.json`. It is not a prediction.
