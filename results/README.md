# Retained results

| Preset | Original job | Measured speedup | Status |
|---|---|---:|---|
| Selective | `027b6db0ef044348` | 1.2048418× | Historical control |
| Orientation | `8fbfde82af1a4761` | 1.2150730× | Historical O-projection experiment |
| PDL | `e6f6e1af0bd446a1` | 1.2310594× | Default source configuration |
| Combined | — | — | PDL plus O orientation; GPU check pending |
| TMA | — | — | Combined plus dense TMA projections; GPU check pending |

PDL's 95% interval is [1.21259, 1.24953]. These are historical check results,
not leaderboard submissions or measurements of the refactored file. All three
measured candidates passed nine correctness cases and thirty timed shapes.

`pdl.json` and `selective.json` contain complete check responses. Their `.txt`
files contain readable per-shape timings and profiles. Orientation is retained
as `orientation.profile.json` plus `orientation.txt`; the original complete
response was not available. Full earlier evidence is in `pre-reorganization`.

The recorded target is 1.61×, not a refreshed live leaderboard. From PDL this
requires 23.54% less aggregate latency. Pooler, QKV, and unpooler account for
approximately 78% of per-load latency, averaged over the six loads. Use
`python -m tools.budget` to reconstruct the score and floor scenarios.

The failed lossless implementation measured 0.1076×. Its evidence and decoder
diagnostics are archived in Git rather than kept in the active implementation.
