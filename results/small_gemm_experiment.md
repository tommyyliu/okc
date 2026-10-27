# Small-batch QKV/pooler: weight-major promising, confirmation pending

**Follow-up:** `67a1b145d83b4f89` passed all tests and scored cases at
3934.4243 us / 1.2278289x, speedup interval [1.2094115, 1.2462464]. It exercised
the new weight-major kernel at nearby pooler M=7. The raw result and profile
are retained as `small_gemm_weight_major_nearby.json/txt`.
[The opportunity analysis](opportunity_analysis.md) puts this incremental gain
in the context of the approximate 3000 us target. Earlier pending-check notes
below describe the original experiment sequence; the nearby check is now done.

The first [optimization priority](optimization_priorities.md) is QKV and pooler
at M=8/16/32: six scored cases with 27.57% score-weighted latency sensitivity.
The measurement reference remains PDL recheck `706c118665f14447` at 1.2197188x.
Both initial experiments passed nine correctness tests and all thirty scored
shapes on H200. Weight-major improved the normalized latency of all six target
cases versus streaming in these checks. PDL remains the default pending
confirmation and nearby validation of the new kernel.

## Initial GPU results

| Candidate | Job | Aggregate speedup | 95% interval |
|---|---|---:|---|
| PDL control | `706c118665f14447` | 1.2197188x | [1.2014230, 1.2380146] |
| Streaming | `829a3acc7d344f38` | 1.2132923x | [1.1950929, 1.2314916] |
| Weight-major | `5aa7a0e662a84bfd` | 1.2324506x | [1.2139639, 1.2509374] |

The streaming and weight-major jobs ran on different H200 nodes in sfo3 and
nyc3, respectively. Compare each candidate to its same-run SGLang timing before
comparing runs. The last column below is
`1 - (weight_major_time / its_SGLang_time) / (streaming_time / its_SGLang_time)`.

| Case | Streaming time (us) | Weight-major time (us) | Lower normalized latency |
|---|---:|---:|---:|
| QKV M=8 | 13.132 | 12.374 | 5.24% |
| QKV M=16 | 13.202 | 12.513 | 4.57% |
| QKV M=32 | 14.220 | 13.272 | 5.51% |
| Pooler M=8 | 19.402 | 18.556 | 2.04% |
| Pooler M=16 | 19.536 | 18.679 | 3.17% |
| Pooler M=32 | 20.602 | 19.514 | 4.60% |

Against the PDL control's normalized ratios, weight-major's reductions are
1.56%/4.97%/6.43% for QKV and 2.79%/3.01%/5.35% for pooler, in M=8/16/32
order. Streaming QKV M=32 alone improves its normalized ratio by only 0.98%
versus PDL/cuBLAS, which is insufficient to select it independently.

These are cross-run observations, not paired candidate-to-candidate confidence
intervals. The aggregate intervals overlap. Even the 24 unchanged cases shift
between the two new runs: normalized latency reductions range from -2.10% to
4.62%, with a median of 0.62%. The coherent six-case result supports another
check but does not establish a reproducible gain.

The separate profiling pass shows weight-major producers at 7.2/7.7/8.7 us for
QKV and 17.3/17.8/18.4 us for pooler at M=8/16/32. Reducers remain about
1.2–1.5 us. Pooler still spends over 92% of profiled time in the producer, so
K tile and pipeline experiments remain useful after confirmation. These profile
times must not be subtracted from the score timings above.

Raw responses and text profiles are retained as `small_gemm_streaming.json/txt`
and `small_gemm_weight_major.json/txt`. `small_gemm_comparison.json` records all
thirty normalized comparisons, source fingerprints, run identities, and a
clearly labeled hypothetical score calculation; it is not another measured run.

## Next check

`build/small-gemm/weight_major_nearby.py` extends the measured weight-major
configuration to every QKV/pooler token count from 2 through 32. It preserves
all thirty scored launches, every GPU kernel, and dispatch outside that range.
This gives the existing pooler M=7 correctness test the new kernel and obtains
another timing observation with the same scored configurations. The initial
weight-major file used the old fallback at M=7, so its correctness result did
not test the new kernel on that nearby shape.

CPU dispatch checks confirm the intended scope and unchanged GPU kernel ASTs.
This extended candidate has not yet had a GPU check. Run from the project root:

```bash
okc check build/small-gemm/weight_major_nearby.py --json \
  | tee build/small-gemm/weight-major-nearby-check.json
```

Keep PDL as the default until nearby correctness passes and the per-case gains
repeat. The associated `weight_major_nearby-manifest.json` describes the change.

## Implementation

The tuner now compares both token-major and weight-major streaming GEMM, with
independent choices of K tile, pipeline stages, and split count for each case.
Default K tiles are 64/128/256 and stages are 2/3/4; flags can bound the search
further. The source dispatch is retained as the control, and cuBLAS competes
for every non-router case. Tiles use BM=16 for M=8/16 and BM=32 for M=32;
BN/warp pairs are 32/4, 64/4, and 128/8.

The new `_weight_major_stream_gemm` computes `W @ X.T`, storing the result with
the required `[M, N]` addressing. It uses the streaming kernel's flattened tile
order and K partition boundaries. Weights and activations remain BF16,
accumulators and split partials remain FP32, and split launches use the existing
PDL reducer. Existing kernels and PDL configuration choices are unchanged.

The local tuner still requires a 3% median gain with a 1.5% lower-decile gain
against source dispatch, then confirms with fresh outlier activations and at
least 40 timing pairs. A selected multi-token kernel is checked on nearby
shapes below and above its scored size before export. These are selection
thresholds, not measurements or guarantees.

## Standalone experiments

Generate all files and their comparison manifest without a GPU:

```bash
python -m tools.tune --ablations build/small-gemm --layers qkv_gate pooler --tokens 8 16 32
```

| File | Change | Compare against |
|---|---|---|
| `control.py` | Exact generated PDL source | Fresh control check |
| `streaming.py` | QKV M=32 uses BM32/BN64/BK128, S8, four warps, three stages | `control.py` |
| `stages2.py` | Two pipeline stages for all six cases | `streaming.py` |
| `stages4.py` | Four pipeline stages for all six cases | `streaming.py` |
| `bk64.py` | K tile 64 for all six cases | `streaming.py` |
| `bk256.py` | K tile 256 for all six cases | `streaming.py` |
| `splits_half.py` | QKV S4, pooler S2 | `streaming.py` |
| `splits_double.py` | QKV S16, pooler S8 | `streaming.py` |
| `weight_major.py` | Reverse dot operands for all six cases | `streaming.py` |

Every variant changes one setting relative to its listed comparison. No other
scored cases or nearby-size dispatch policies change. Nearby correctness for
the experimental kernel itself must also be checked explicitly; exact-shape
overrides preserve the default nearby dispatch.

The control and streaming comparison is now recorded above. Follow with
nearby confirmation before further ablations. Compare candidate/SGLang
ratios per case within each run, and compare these normalized ratios across
runs. Retain only case-specific winners, recheck small gains, and validate
nearby shapes before promotion. Further pipeline/K-tile variants should use
weight-major as their comparison if the initial gains repeat; the originally
exported `stages*` and `bk*` files still use token-major streaming.

For an offline Hopper compile audit, including PDL instruction ordering:

```bash
python -m tools.codegen --source build/small-gemm/weight_major.py \
  --layers qkv_gate pooler --tokens 8 16 32 --output build/small-gemm-codegen
```

For a local H200 search and validation:

```bash
python -m tools.tune --layers qkv_gate pooler --tokens 8 16 32 \
  --stages 2 3 4 --k-tiles 64 128 256 --splits 2 4 8 \
  --output build/small-gemm-tuned.py --report build/small-gemm-tuning.json
python -m tools.validate build/small-gemm-tuned.py --nearby
okc check build/small-gemm-tuned.py --json > build/small-gemm-tuned-check.json
```

## Checks in this environment

- 21 CPU tests pass, including checkpoint kernel/launch preservation, scoped
  ablations, search-axis coverage, and weight-major launch/compile-manifest
  agreement with FP32 partials and PDL reduction.
- `python -m tools.build --check` passes.
- All nine standalone sources are generated under ignored `build/small-gemm/`.
- Torch and Triton are unavailable, and no `nvidia-smi` executable is present.
- `okc problems --json` failed with `can't reach openkernelchallenge.dev: no such host`.

The user supplied the two initial GPU results after these local checks.
Confirmation and broader nearby validation of the new kernel remain pending.
The priority stays open.
