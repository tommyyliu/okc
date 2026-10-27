# Full-attention FA3 candidate

Status: **passed** as `2524b1c4986a49b4`, all 13 correctness tests and 12 timed
shapes. Measured **21184.86 us / 1.0677577x**, with a 95% speedup interval of
[1.0517413, 1.0837740]. `submission.py` retains this measured default.
The prior split-attention control is the 22056.34 us / 1.0255689x FP8-fix run
`71b479393ecd4532`, retained in `results/fp8-fix/`. This experiment's exact
source, original manifest, response, profile, and analysis are retained in
`results/full-fa3/`.

## Result

| Full-attention shape | Normalized latency change vs split control |
|---|---:|
| Decode 1 | +3.04% |
| Decode 16 | -8.69% |
| Decode 32 | -11.20% |
| Verify 1 | -2.05% |
| Verify 16 | -19.17% |
| Verify 32 | -22.09% |

Negative means lower candidate/SGLang latency. Unchanged sliding shapes vary
between -1.04% and +1.89%; the aggregate speedup intervals do not overlap.
The reported score improves by 3.95%. Keep FA3 as the next control. Leave
small-batch backend selection alone for now: decode 1's regression is small
in aggregate score terms and needs confirmation before introducing another
shape policy. Experts now dominate the larger sliding verify profiles, so
the next experiment is [a Marlin block-size cap](moe_m16_experiment.md).

## Change and hypothesis

Use the baseline's `flash_attn_with_kvcache` entrypoint on every full-attention
layer. Retain custom split attention on every sliding layer. Full layers at
batches 16/32 currently run at 0.79–0.93x same-run SGLang, with custom attention
using 227.5–663.8 us in the profile. This experiment asks whether FA3 removes
that regression while keeping our other whole-layer work. It is not a claim
that FA3 will improve all six full shapes, especially the smallest batches.

The query is normalized, rounded directly to FP8 E4M3, and passed to FA3 with
`softmax_scale = sm_scale * qk_scale_factor`. The factor is not multiplied
into the query before FP8 rounding. This uses the query precision allowed by
the task and follows the baseline's scaling convention. The custom sliding
path retains its BF16 queries with the factor folded in before attention.
The backend comparison therefore includes this permitted query-precision
change on full layers.

Both paths use the existing fused QK norm/RoPE/cache-store kernel. Cache
locations and scale tensors still come from the current inputs. FA3 receives
the supplied page table, cache lengths, cumulative query lengths, scheduler
metadata, and conditional per-KV-head descales. Full attention is causal and
has no window. A new output kernel applies the BF16 attention-output boundary,
per-head RMSNorm, and sigmoid gate using the same math as the control merge.

The six scored full shapes change. All six scored sliding shapes, projection
dispatch, routing, expert alignment/GEMMs, boundaries, preparation, and stream
scheduling remain at control. Existing GPU definitions and the retained split
implementation are checked against the passing source checkpoint.

## Frozen exports and checking

Run from `muse_decode_layer/`:

```bash
python -m tools.build --preset full_fa3 --output build/full-fa3/candidate.py \
  --manifest build/full-fa3/manifest.json
python -m tools.build --preset control --output build/control/candidate.py \
  --manifest build/control/manifest.json
python -m tools.build --check
python -m unittest discover -s tests -v

okc check build/full-fa3/candidate.py --json > submission3.json
python -m tools.analyze submission3.json --compare submission2.json \
  --output build/full-fa3/analysis.json
```

The local environment has no Torch/Triton or GPU, so CPU tests execute the real
Python dispatch against recorded fake tensor/kernel calls. They check backend
selection, FP8 query scaling, causal verification lengths, scheduler forwarding,
unit/nonunit KV descales, output shape/dtype, and tuple-return handling. They
do not establish GPU compilation, numerical accuracy, or speed.

Require all correctness cases and timed shapes to pass. Inspect full-layer
normalized latency ratios against the prior control and use unchanged sliding
shapes to assess run-to-run variation. Do not promote a speedup based solely
on a small aggregate difference within the intervals. If correctness fails,
retain the frozen candidate and diagnose its first failing case before tuning.

The check is complete; no leaderboard submission has been made for this
candidate. The measured file is retained at `results/full-fa3/candidate.py`.
