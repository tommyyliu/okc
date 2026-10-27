# Marlin token block cap at 16

Status: **passed** as `f2293cc26af04ec1`, all 13 correctness tests and 12 timed
shapes. Measured **20264.20 us / 1.1162692x**, interval [1.0995252, 1.1330132].
This is now the default policy. Exact source, original manifest, response,
analysis, and profile are retained in `results/moe-m16/`.

| Changed large verify shape | Normalized latency change vs FA3 control |
|---|---:|
| Batch 16 sliding | -16.15% |
| Batch 32 sliding | -14.84% |
| Batch 16 full | -12.16% |
| Batch 32 full | -9.75% |

Unchanged cases vary from -0.74% to +2.67%. The score improves by 4.35%,
and aggregate speedup intervals do not overlap. The next experiment targets
small-token expert finalization; see [the record](small_finalize_experiment.md).

## Evidence and scope

Submission 3 passed and retained the larger full-attention gains. The two
Marlin GEMMs still take 336.2/534.0 us on sliding verify 16/32, or 69.2%/74.3%
of the summed profile time. These profiles justify a bounded expert policy
comparison before writing new expert math.

The entire exported source differs from the measured FA3 candidate in one
constant: `BLOCK_SIZE_M_CANDIDATES` changes from `(8, 16, 32, 48, 64)` to
`(8, 16)`. The existing selector, alignment, GEMMs, SiLU, norm/weighted sum,
NVFP4 repacking, all local routes, and every other layer stage are unchanged.
The same selected block size is passed to alignment and both GEMMs, as before.

The policy reduces the tile from 48 to 16 at T=128 and from 64 to 16 at
T=256. T=1/8/16/32, including all decode and verify-1 scored shapes, retain
their current selection. Nearby sizes use the same capped range policy;
selection starts changing at T=58. It branches only on tensor shape.

Smaller blocks may improve scheduling and reduce padding on skewed routing.
They may also repeat weight reads and lose tensor-core reuse. This is a
performance hypothesis, not a predicted gain. No precision or expert skipping
is introduced.

## Check

From `muse_decode_layer/`:

```bash
python -m tools.build --preset full_fa3_m16 --output build/moe-m16/candidate.py \
  --manifest build/moe-m16/manifest.json
okc check build/moe-m16/candidate.py --json > submission4.json
python -m tools.analyze submission4.json --compare submission3.json \
  --output build/moe-m16/analysis.json
```

Thirteen CPU checks pass, including exact AST equality after normalizing the
one changed constant, executing the real exported selector on scored/nearby
sizes, and all previous workflow checks. They do not establish GPU results.

Compare the four large-verify shapes to submission 3 and inspect both Marlin
times and end-to-end graph timing. Eight scored shapes should be unaffected
and provide a noise comparison. The cap won in this run and is retained as
the next control. The check commands above are historical; do not rerun them
without a specific confirmation need.
