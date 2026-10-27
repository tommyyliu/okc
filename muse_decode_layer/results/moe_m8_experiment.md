# Marlin token block cap at 8

Status: **passed correctness, rejected for performance**, job `7844a7a3dbef43b6`.
All 13 correctness tests and 12 timed shapes passed at **20511.35 us /
1.1028188x**, interval [1.0862766, 1.1193611]. This is 7.69% slower than
submission 5. Exact source, original manifest, response, profile, and analysis
are retained in `results/moe-m8/`.

Large verify normalized latency regresses 13.11–30.56%; decode 32 sliding
regresses 8.96%. The four unchanged T=1/8 shapes vary from -1.44% to +0.27%.
Keep block cap 16. The next candidate is
[direct single-token NVFP4](single_nvfp4_experiment.md).

This candidate changes only `BLOCK_SIZE_M_CANDIDATES = (8, 16)` to `(8,)`.
All GPU definitions, preparation, routing, local expert evaluation, and
precision boundaries are unchanged. Both GEMMs and their alignment use
the existing selector's result. The passing fused finalize remains enabled
at T <= 16. The exported AST differs only in this one constant.

Scored T=16/32/128/256, both sliding and full, change from block size 16 to
8. Scored T=1/8 keep size 8, providing four unchanged comparison shapes.
Nearby sizes use the same capped policy; tests cover odd T and dispatch
boundaries. Smaller blocks may reduce padding/scheduling costs but may
repeat weight reads or underuse tensor cores. No gain is assumed.

From `muse_decode_layer/`:

```bash
okc check build/moe-m8/candidate.py --json > submission6.json
python -m tools.analyze submission6.json --compare submission5.json \
  --output build/moe-m8/analysis.json
```

To reproduce the frozen candidate:

```bash
python -m tools.build --preset full_fa3_m8_finalize \
  --output build/moe-m8/candidate.py --manifest build/moe-m8/manifest.json
```

Use the complete GPU correctness/timed output checks before promotion. Judge
end-to-end graph timing and normalized per-shape results, not the raw expert
profile alone. Submission 5's unchanged larger verify cases varied by several
percent, so interpret small differences cautiously. If this loses, retain the
default and prioritize a different expert implementation over more block caps.
