# Small-token fused expert finalize

Status: **passed** as `418dc106c212475a`, all 13 correctness tests and 12 timed
shapes, at **19046.05 us / 1.1876636x**, interval [1.1698486, 1.2054785]. The
exact source, original manifest, response, profile, and analysis are retained
in `results/small-finalize/`. This is now the default; `submission.py` matches
the checked source byte for byte. The prior m16 control was 20264.20 us /
1.1162692x, so the reported score improves by 6.01%.

| Changed small-token shape | Normalized latency change vs m16 control |
|---|---:|
| Decode 1 sliding | -16.87% |
| Decode 1 full | -14.92% |
| Decode 16 sliding | -13.10% |
| Decode 16 full | -8.82% |
| Verify 1 sliding | -15.95% |
| Verify 1 full | -13.34% |

The kernel itself takes about 3.0–3.3 us in the profiles. Larger unchanged
verify cases vary 3.24–7.58% worse; normalized ratios do not remove all workload
and timing variability. Keep the fused path, but do not attribute those
unaffected changes to it. The next candidate is
[block cap 8](moe_m8_experiment.md).

## Why this experiment

Small-token profiles show multiple PyTorch reductions and elementwise
operations around expert output normalization; larger shapes use a dedicated
library kernel. The new path fuses the small-token work into one Triton
program per token, processing its eight expert slots and 4096 channels.

It loads each slot's local expert id first, and masks remote-slot loads
before any arithmetic. Marlin may leave those slots uninitialized, so
multiplying a later result by zero would not be sufficient to exclude NaNs.
For local slots, multiply the BF16 expert output by the existing effective
norm weight, round that product to BF16, compute the RMS in FP32 with epsilon
1e-8, multiply by the FP32 routing weight, and sum slots in FP32. Store the
result as BF16. Remote slots contribute zero.

The candidate enables this only at T <= 16: scored decode 1/16 and verify 1,
both sliding/full. T=32/128/256 retains the library finalize. Both Marlin
GEMMs, alignment/block-cap policy, SiLU, routing, attention, projections,
boundaries, and preparation are unchanged. After normalizing the enable flag,
the candidate's exported AST equals the regenerated control's AST.

One large reduction tile may have high register use or insufficient
parallelism at one token. Avoiding several launches is the expected benefit;
GPU timing must establish whether that outweighs those costs.

## Run

From `muse_decode_layer/`:

```bash
python -m tools.build --preset full_fa3_m16_finalize \
  --output build/small-finalize/candidate.py --manifest build/small-finalize/manifest.json
okc check build/small-finalize/candidate.py --json > submission5.json
python -m tools.analyze submission5.json --compare submission4.json \
  --output build/small-finalize/analysis.json
```

The default build now enables the passing fused path, and is byte-identical
to `results/small-finalize/candidate.py`. The earlier unfused control remains
at `results/moe-m16/candidate.py`. The check commands above are historical;
use the next experiment record for the current check command.

CPU tests execute the real expert host path with recorded tensor/kernel
calls, check the T=16/17 boundary and larger fallback, and verify that both
GEMMs, block alignment, routing weights, local ids, and output shape/dtype
are preserved. They do not execute GPU numerical reductions or establish
compiler/resource behavior. Require the complete GPU correctness suite and
all timed output checks before considering this path a measured improvement.
