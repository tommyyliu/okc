# Round four: tiled weights, native outputs, and wider router dots

**Completed:** `0fe4f6662e1d458a` passed at **3884.83 us / 1.2435046x**,
with interval [1.2248520x, 1.2621571x]. All correctness and timed cases passed.
Raw response/profile and per-case comparisons are retained as
`multi_category_round4.json/txt` and `multi_category_round4_analysis.json`.
The four new groups show no clear broad improvement over their measured
predecessors; carried earlier gains and run variation contribute to the aggregate.

Tiled weights were flat/slower; reversed cuBLAS was flat except a roughly 10%
QKV M256 regression; native O stores were inconclusive; router interleaving
regressed M8 about 11% with smaller, uncertain changes elsewhere. Unchanged
cases shifted between 3.7% faster and 4.9% slower. The
[round-five plan](multi_category_round5_plan.md) prioritizes K-loop scheduling,
full-K GEMV warp count and a single-pass router.

The following describes the submitted round-four candidate.
The consolidation response is not yet present as
`build/multi-category-round3/keepers-check.json`. Round four uses that candidate's
scored settings as a provisional control, with measured per-case evidence from
`67a1b145d83b4f89` and `327bb5b2fdc6423a`. Its 3858.27 us projected score is not
a measurement. The expanded [21-idea map](optimization_idea_map.md) records
the alternatives and result-dependent follow-ups.

Run from the project root:

```bash
okc check build/multi-category-round4/candidate.py --json \
  | tee build/multi-category-round4/check.json
```

Rebuild reproducibly with:

```bash
python -m tools.build --preset round4 --output build/multi-category-round4/candidate.py
```

## Four independent groups

| Group | Cases | Change | What stays fixed for the comparison |
|---|---|---|---|
| Tiled BF16 weights | QKV/pooler M8/M16/M32: 6 | Prepare contiguous `[N/64,K/128,64,128]` tiles and load them in `_weight_major_tiled_gemm` | Logical dot orientation, FP32 accumulation, BM/BN/BK, splits, warps, stages, PDL, output and scratch layouts |
| Reversed cuBLAS | QKV M128/M256; unpooler M16/M32/M128/M256: 6 | Compute `[N,M]` using `W @ X.T` and return a transpose view | Original QKV storage and the selected transposed unpooler storage; arithmetic precision |
| Natural O output | O M8/M16/M32: 3 | Store the accumulator directly as `[N,M]` and return a transpose view | Original O loads/dot loop, tiles, warps and stages |
| Router interleaving | Router M8/M16/M32/M128/M256: 5 | Prepare `[expert,hi/lo,K]` BF16 storage; one dot produces adjacent independent FP32 component accumulators, then adds each pair | Logical expert tiles, grids, K splits, stages, warps, FP32 scratch and PDL reducer |

The groups occupy twenty distinct scored cases. Ten scored cases retain
control dispatches, including all M1 cases, large pooler, large O and sliced
unpooler. Existing GEMV, dot, streaming, weight-major and TMA JIT bodies are
unchanged. Default PDL dispatch remains the historical configuration.

Transposed outputs are complete `[M,N]` results with different strides. The
task's local checker requires shape/dtype and numerical accuracy, and does not
require contiguous outputs. There is no deferred numerical computation or output
copy. All activation-dependent computation remains in the timed call.

The router's single wider dot has the same total arithmetic as the original
two dots. Its hypothesis concerns load/layout/instruction scheduling. High and
low components remain BF16 and are accumulated separately in FP32 before their
pairwise addition; the original FP32 router weights remain available to M1.

## Preparation, nearby shapes and resources

The candidate prepares lossless BF16 copies for weight layouts and the allowed
router hi/lo decomposition. Added storage is **236 MiB per prepared state**:
128 MiB of transposed pooler/unpooler copies, 100 MiB of QKV/pooler tiles,
and 8 MiB of interleaved router components. All descriptors/pointers used in
timed calls come from the current prepared state, including evaluator clones.

Nearby dispatch applies the new tiled kernel throughout QKV/pooler M2..32,
including pooler M7, the reversed cuBLAS path to larger QKV and full unpooler,
including M512/M130/M1000, and the interleaved router kernel to M>1, including
the M9 outlier test. O M2..32 uses native stores; M33 retains library fallback.

Preparation retains the preset's required layouts when scored tuning overrides
change families, since nearby fallback can still select those layouts.

## Local validation and artifacts

All **31 CPU tests pass**, including:

- Numerical preparation checks on exact BF16 patterns, matching actual packed
  GPU address expressions across output/K tiles and K splits.
- Numerical hi/lo interleaving and reconstruction against toy FP32 router weights.
- Logical router grid, scratch shape/precision and PDL agreement.
- Native output shapes/view returns, current cloned-state pointer binding,
  nearby dispatch and checkpoint launch/kernel preservation.
- Compile-manifest signatures/constants matching the host launches.

The generated default build is current. The planned GPU compilation consists
of **32 kernel/reducer jobs**. Torch/Triton and a GPU are unavailable locally;
the numerical CPU preparation checks did not establish GPU compiler success,
GPU correctness or performance. The completed check above subsequently passed
GPU compilation and correctness; its timings provide the performance evidence.

`build/multi-category-round4/manifest.json` records all case changes, source
hashes, pending consolidation status and preparation memory. `control.py`
has scored dispatches matching the in-flight consolidation candidate. The four
`isolated/*.py` files are diagnostic exports in case a combined failure needs
localization; they are not four additional requested checks. Each export's
host dispatch was compared on scored and sampled nearby shapes.

Use official per-case paired speedups versus same-run SGLang. Compare each group
with consolidation when that result is available, and inspect unchanged cases
for run variation. Retain winners by case; a mixed aggregate can hide gains.
