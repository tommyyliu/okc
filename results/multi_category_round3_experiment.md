# Round three: prepared layout, router scheduling, and chunked GEMV

User-reported competitor target: **2950 us**. Base job `67a1b145d83b4f89`
measured 3934.4243 us; reaching 2950 needs about 25% less score latency.
The previous two broad experiments found repeatable O orientation and router
M=128 gains but no substantial aggregate improvement. Avoid repeating their
failed BK256 or large non-TMA configurations.

## Completed result

Job `327bb5b2fdc6423a` passed all nine correctness tests and thirty timed cases
at **3935.59 us / 1.2274639x**, with speedup interval [1.2090520x, 1.2458759x].
The aggregate is effectively unchanged from the 3934.42 us base, but gains and
regressions cancel. Raw response/profile are retained as
`multi_category_round3.json/txt`; `multi_category_round3_analysis.json` records
all official paired-ratio comparisons and the hypothetical consolidation.

| Experiment | Normalized latency change vs base | Decision |
|---|---:|---|
| Pooler prepared transpose M128/M256 | 9.3% / 4.0% faster | Confirm both; M128 is the clearest new gain. |
| Unpooler prepared transpose M16/M32/M128/M256 | 3.3% / 2.3% / 5.3% / 2.4% faster | Coherent direction; small changes need confirmation. |
| QKV prepared transpose M128/M256 | 0.8% faster / 5.3% slower | Return to original layout. |
| Router narrower BN32/S16, M8/M16/M32 | 15.9% / 8.2% / 14.2% slower | Reject this configuration despite preserved CTA count. |
| Chunked GEMV, QKV/pooler/unpooler/router | 4.5% slower / 0.4% faster / 0.2% slower / 41.6% slower | Return all four to original full-K reduction. |
| O orientation M8/M16/M32 | 8.8% / 8.4% / 9.2% faster | Repeated winner; retain. |
| Router M128 split16 | 5.4% faster | Repeated promising gain; retain. |

The eleven unchanged cases shifted between 3.8% faster and 3.2% slower,
with median 0.45% slower. Layout results on small unpooler cases therefore
remain uncertain. The profile shows the packed layout changes the cuBLAS
kernel family to NNT for pooler/unpooler; QKV selects NNN and does not benefit.
Equal CTA count alone did not make the narrower router configuration faster.

The user-reported 2950 us competitor still requires about **984 us / 25%**
less latency from base. No measured broad improvement closes that gap yet.
Applying only the ten selected per-case ratios to the base load sums projects
**3858.27 us**. This is a cross-run estimate, not a measured score or confidence
interval.

## Next check: consolidate the promising cases

`build/multi-category-round3/keepers.py` retains O orientation, router M128
split16, and the six pooler/unpooler transpose cases. It restores original
QKV layout, all full-K GEMVs, and small-router BN64/S32 settings. Its thirty
host dispatches match either the passing base or the passing round-three
candidate; only pooler/unpooler transpose copies are prepared. GPU math is
unchanged. `keepers-manifest.json` records the selection and source hash.
Nearby policies are inherited from base, and the initial round-three result
already exercised the transpose path on nearby correctness shapes.

The consolidated candidate is **GPU unmeasured**. Run one check:

```bash
okc check build/multi-category-round3/keepers.py --json \
  | tee build/multi-category-round3/keepers-check.json
```

Rebuild the original exploratory candidate from source with:

```bash
python -m tools.build --preset round3 --output build/multi-category-round3/candidate.py
```

## Three independent experiments

| Group | Scored cases | Change | Hypothesis and limit |
|---|---|---|---|
| Prepared weight layout | QKV/pooler M=128/256; unpooler M=16/32/128/256 | Create contiguous BF16 `[K,N]` copies in `prepare()`, then call `torch.mm(x, packed)` | Test a new cuBLAS operand layout without another custom GEMM guess. These eight cases have about 36.1% of base score sensitivity. A loss would not rule out custom kernels on packed layouts. |
| Small router | M=8/16/32 | BN64/S32 to BN32/S16, retaining BM, BK64, four warps, three stages | The grid remains 128 CTAs and FP32 partial storage halves. Each CTA does a longer K loop, so reduced scratch/reduction cost may still fail to repay producer cost. Retain independent BF16 hi/lo FP32 accumulators and PDL. |
| Single-token GEMV | QKV, pooler, unpooler, router at M=1 | Replace full-K vectors by a 1024-element FP32 vector accumulated over K chunks, with one final reduction | Reduce live vector size while keeping one output row per CTA and four warps. Loop overhead could outweigh the resource benefit. Router consumes original FP32 weights; other weights remain BF16. |

The groups do not overlap. O M=8/16/32 orientation and router M=128 split16
are retained gains from previous checks. Six small QKV/pooler cases keep base
weight-major BK128 settings. Eleven scored dispatches match base unchanged;
nineteen differ, comprising fifteen new experiments and four retained gains.

Nearby dispatch uses the new prepared layout for QKV/pooler M>32 and unpooler
M>8, so correctness cases QKV M512 and unpooler M130/M1000 exercise it. Router
M2..32 uses the narrower configuration, including the M9 outlier test. Small
pooler M7 retains the already checked weight-major kernel. No input-dependent
work runs in preparation.

Packing adds **164 MiB** of BF16 weight copies to each prepared state. The
original weights remain available to unchanged paths. Prepared state contains
tensors rather than bound pointers; timed calls use the current state, including
when the evaluator clones it.

## Initial local validation and comparison

All **25 CPU tests pass**, including checkpoint kernel preservation, compile
manifest/host launch agreement, preparation scope/precision, current cloned-state
binding, router grid/scratch geometry, and single-token weight/output dtypes.
The default generated build is current. The candidate has a 32-job Triton
compile plan; all original JIT function ASTs match the passing base and the only
new JIT is `_gemv_chunked`. Torch, Triton, and a GPU are unavailable locally,
so this is not a GPU compilation, correctness, or timing result.

`build/multi-category-round3/manifest.json` records the source hash, all case
changes, nearby policy, and exact base. `control.py` in that directory combines
base settings with the retained O/router gains; it is available for a later
confirmation, not an additional requested check now.

Use each case's official median paired speedup against same-run SGLang when
comparing runs. Retain gains by case and inspect unchanged cases for variance.
Profile-pass kernel times are separate from scored timings. Choose the second
check only after seeing which of these experiments has evidence of a gain;
the aggregate of a mixed candidate is not the promotion decision.
