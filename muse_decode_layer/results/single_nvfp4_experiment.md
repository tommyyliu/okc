# Direct single-token NVFP4 experts

Status: implemented, twenty CPU checks pass; GPU compilation, correctness,
and timing pending. The measured control remains submission 5 at 19046.05 us /
1.1876636x. Cap 8 failed its performance experiment and is not retained.

## Design and scope

For T=1, bypass expert alignment, Marlin's two GEMMs, and its separate SiLU
launch. One Triton kernel reads gate/up weights in their original NVFP4
packing, expands them to BF16 in registers, accumulates FP32 vector products,
rounds gate/up outputs to BF16, and fuses SwiGLU into a BF16 intermediate.
The next kernel projects that intermediate through the original packed w2,
again expanding only in registers and accumulating in FP32. The passing
masked expert norm/weighted-sum kernel finalizes the result.

Each kernel assigns eight output rows to a program and one routing slot to
the second grid axis. It uses 512-element K chunks and attends to every
4096 input channel of each selected local expert. The grid is 512 x 8;
nonlocal expert ids skip their GPU work, and the finalizer masks their
uninitialized slots before loading. All input-dependent work is timed.

Preparation clones the original w13/w2 packed bytes, FP8 block scales, and
FP32 global scales before Marlin's loader runs. The extra storage is 864 MiB
plus 256 bytes per prepared copy. Dtypes and bits are preserved; there are no
expanded BF16 expert weights in state. Marlin remains available for T>=2.
Only the two scored decode-1 shapes change.

The kernel retains BF16 projection/intermediate/output boundaries and FP32
accumulation. E2M1 decoding handles all sixteen signed codes, 16-element block
scales, and each expert's global scale. It neither approximates the routing
nor skips any selected local expert.

Empty GPU programs for remote routes, register pressure, extra decoding
instructions, and repeated scale loads may outweigh the removed launches.
The design is a bounded first raw-NVFP4 implementation, not a predicted win
or a solution to the entire remaining score gap.

## Check

From `muse_decode_layer/`:

```bash
okc check build/single-nvfp4/candidate.py --json > submission7.json
python -m tools.analyze submission7.json --compare submission5.json \
  --output build/single-nvfp4/analysis.json
```

Reproduce the frozen export:

```bash
python -m tools.build --preset single_nvfp4 \
  --output build/single-nvfp4/candidate.py --manifest build/single-nvfp4/manifest.json
```

CPU checks execute the actual nibble decoder with scalar FP32/BF16 semantics
for all sixteen E2M1 codes and 21 scale combinations, verify actual packed
weight/scale addresses at expert/row/nibble/block boundaries, and execute the
actual single-token host dispatch using recorded GPU calls. Existing checks
preserve other GPU functions and the fallback paths. They do not compile or
run these GEMVs on a GPU, test GPU reduction order, or measure resources.

Require the full GPU correctness suite and timed output checks. Compare the
two decode-1 shapes to submission 5; the other ten shapes are controls. Retain
the original measured file at `results/small-finalize/candidate.py`: the
regenerated default includes new disabled direct-kernel code and guards.
