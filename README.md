# muse_attention_decode — fused gated attention decode (Triton, H200)

Open Kernel Challenge problem `muse_attention_decode`: one decode step (q_len 1) or 8-token
speculative-verify block of GQA attention (16 q heads / 2 KV heads, head dim 64) over a paged
FP8 E4M3 KV cache, with QK RMSNorm, RoPE on 2,048-token sliding-window layers, and an
RMSNorm × sigmoid(gate) output. Baseline is SGLang's FA3 path (5 kernels).

## Status

| run | what | result |
|---|---|---|
| `3819260aba274207` (scored) | `results/submitted_v1.py` | **1.559×** on the board, 16/16 tests |
| `c1e07a264d3e459a` (check) | same code | 1.574× [1.550, 1.601] |
| bench12 (check) | current `submission.py` defaults | 1.717×, 16/16 tests — not yet submitted |

Leader at the time: 3.03×. Per-shape judged times for v1 are in
`results/runs/submission.c1e07a264d3e459a.txt`.

## Design (`submission.py`)

One Triton kernel per call, no other launches:

- **Grid** `(KV-head pack, split, request)`. Each program normalizes and rotates its queries
  straight from the fused `qkv` row and attends a fixed slice of the cached context with an
  online softmax.
- **Head packing (decode)**: two KV heads per program with a block-diagonal query
  (`[16 rows, 128 cols]`), so each gathered cache row is one contiguous 128-byte read and the
  16-row MMA fills exactly. Random 64-byte reads top out at ~2 TB/s on H200; 128-byte reads reach ~3.0–3.3 TB/s.
- **New tokens on an extra split**: one extra program per (request, pack) builds the block's
  keys/values, writes them to the cache and attends them causally, in parallel with the rest.
- **Two-level tree merge**: partials go to a workspace; the last arriver in each group of splits
  merges the group, and the last group merger finishes. It applies the BF16 attention-output
  boundary, the per-head RMSNorm and the sigmoid gate. Arrival counters reset themselves, so
  CUDA-graph replays work.
- **Latency details**: fixed per-split slices (on full layers the page-table loads don't wait on
  `seq_lens`), unmasked interior tiles with masked edge tiles only, host-cached `inv_freq`,
  range-reduced `sin/cos.approx` for RoPE, gate prefetch to L2 or registers.
- **Precision**: queries are rounded to FP8 E4M3 for QK (as FA3 does). P·V runs in FP16 with
  FP32 accumulation; P is *not* FP8. Relative L2 error is ~0.25% against a 3–15% bound.
- `_plan()` holds the per-shape launch table (pack, splits, tile, warps, stages, maxnreg…),
  tuned under CUDA-graph replay with a cold L2. `TUNE` overrides it for development only.

## Where the time goes (kernel times, H200)

- **Full decode** (d16f/d32f ≈ 109/199 µs) is at the measured random-gather ceiling.
- **Full verify** dropped from 410 to ~325 µs (v32f). That came from unmasked tiles and
  `maxnreg=120` to fit 4 programs per SM; the memory floor is ~200 µs. The loop is now
  compute/latency-bound: `wgmma`, but V needs fp8→fp16 through shared memory.
- **Sliding** shapes are latency chains: `seq_lens` → page table → KV is 3 DRAM round
  trips, then the merge tail (atomic + merge loads + epilogue) is ~2 µs. d1s ≈ 5.4 µs.

## Next ideas (untested)

1. Full verify: a "dual-head" path. One program loads both heads' halves of each 128-byte row
   back to back and runs two separate 64-row dots, which avoids the block-diagonal waste.
2. Small-batch merges (v1s/v1f): fewer serial merge round trips, e.g. `gs` 8–12. This helped
   v1f 36.6 → 34.1 µs in bench12.
3. The stamps show the PROF phase "p5" fires before the merge loads finish (it only depends on
   `l`). Re-check merge vs. epilogue cost with a dependency on `o`.

## Development workflow (no local GPU)

All measurement happens inside `okc check` runs (20/day):

```bash
tools/make_bench.sh bench        # submission.py + tools/bench_tail.py -> build/bench.py
okc check build/bench.py --json > build/bench.json
```

`tools/bench_tail.py` runs at import in the judge's process. It generates inputs, checks
correctness against an FP32 torch reference, times SGLang and each `SWEEP()` config as CUDA-graph
replays after an L2 flush, and can record per-phase `%globaltimer` stamps (`_prof`). Edit `SWEEP()`
for the experiment. `tools/ablate_tail.py` is the stripped verify-loop ablation.
The tails are dev-only: never submit a file with one appended.

`results/sweeps/*.log` are the raw outputs of every sweep (bench1–12, ablate1), in order.
`results/runs/` keeps the JSON and profiles of the reference runs above.

Challenge files (`TASK.md`, `AGENTS.md`, `baseline.py`, `reference.py`, `checks.py`) come from
`okc init` and are git-ignored, so they aren't redistributed.
