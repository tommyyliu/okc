"""Execute actual expert host dispatch with recorded CUDA calls; no GPU math."""

import ast
import math
from types import SimpleNamespace
import unittest

from tests.test_attention import Kernel
from tools.build import build_source, source_manifest


class Tensor:
    def __init__(self, shape, dtype="bf16", device="cuda"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device

    def to(self, dtype):
        return self

    def contiguous(self):
        return self

    def __getitem__(self, index):
        return Tensor(index.shape, "int32") if isinstance(index, Tensor) else self

    def view(self, *shape):
        count = math.prod(self.shape)
        size = -math.prod(shape)
        return Tensor(tuple(count // size if n == -1 else n for n in shape), self.dtype, self.device)


class FinalizeTests(unittest.TestCase):
    def test_candidate_only_enables_finalize_flag(self):
        trees = [ast.parse(build_source(p)) for p in ("full_fa3_m16", "full_fa3_m16_finalize")]
        for tree in trees:
            for n in tree.body:
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SMALL_EXPERT_FINALIZE" for t in n.targets):
                    n.value = ast.Constant("normalized-flag")
        self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))
        manifest = source_manifest(build_source("full_fa3_m16_finalize"))
        self.assertEqual(manifest["preset"], "full_fa3_m16_finalize")
        self.assertIs(manifest["small_expert_finalize"], True)

    def test_real_experts_host_flow_changes_only_small_finalize(self):
        for enabled in (False, True):
            for tokens in (1, 4, 8, 16, 17, 32, 128, 256):
                with self.subTest(enabled=enabled, tokens=tokens):
                    tree = ast.parse(build_source())
                    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_experts")
                    fused, legacy_calls, gemm_calls, align_calls = Kernel(), [], [], []
                    local_ids = Tensor((tokens, 8), "int32")

                    class Map:
                        def __getitem__(self, index):
                            return local_ids

                    def legacy(y, weights, **kwargs):
                        legacy_calls.append((y, weights, kwargs))
                        return Tensor((tokens, 4096))

                    def gemm(*args, **kwargs):
                        gemm_calls.append((args, kwargs))
                        return args[1]

                    def align(ids, block, experts):
                        align_calls.append((ids, block, experts))
                        return object(), object(), object()

                    namespace = dict(
                        DIRECT_SINGLE_EXPERTS=False,
                        SMALL_EXPERT_FINALIZE=enabled, BLOCK_SIZE_M_CANDIDATES=(8, 16),
                        TOP_K=8, LOCAL_EXPERTS=32, N_EXPERTS=256, POST_NORM_EPS=1e-8,
                        torch=SimpleNamespace(float32="fp32", empty=Tensor,
                                              empty_like=lambda x: Tensor(x.shape, x.dtype, x.device)),
                        scalar_types=SimpleNamespace(float4_e2m1f="nvfp4"),
                        moe_align_block_size=align, moe_wna16_marlin_gemm=gemm,
                        silu_and_mul=lambda *args, **kwargs: None,
                        fused_post_expert_norm_weighted_sum=legacy, _small_expert_finalize=fused,
                    )
                    exec(compile(ast.Module(body=[function], type_ignores=[]), "<expert host>", "exec"), namespace)
                    bank = {key: object() for key in ("w13", "w13_s", "w13_gs", "w2", "w2_s", "w2_gs", "workspace")}
                    bank.update(size_n=4096, expert_map=Map())
                    norm_w = object()
                    x, routes = Tensor((tokens, 4096)), Tensor((tokens, 8), "fp32")
                    out = namespace["_experts"](x, routes, Tensor((tokens, 8), "int32"), dict(bank=bank, post_expert_w=norm_w))
                    self.assertEqual((out.shape, out.dtype), ((tokens, 4096), "bf16"))
                    self.assertEqual(len(gemm_calls), 2)
                    self.assertEqual(gemm_calls[0][1]["moe_block_size"], align_calls[0][1])
                    self.assertEqual(gemm_calls[1][1]["moe_block_size"], align_calls[0][1])
                    if enabled and tokens <= 16:
                        self.assertFalse(legacy_calls)
                        grid, args, options = fused.calls[0]
                        self.assertEqual(grid, (tokens,))
                        self.assertEqual(args[0].shape, (tokens, 8, 4096))
                        self.assertIs(args[1], routes)
                        self.assertIs(args[2], local_ids)
                        self.assertIs(args[3], norm_w)
                        self.assertIs(args[4], out)
                        self.assertEqual(options["num_warps"], 8)
                    else:
                        self.assertFalse(fused.calls)
                        self.assertEqual(len(legacy_calls), 1)
                        self.assertIs(legacy_calls[0][2]["slot_expert_ids"], local_ids)


if __name__ == "__main__":
    unittest.main()
