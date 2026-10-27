"""Check actual Python dispatch/FA3 arguments without importing GPU libraries."""

import ast
from types import SimpleNamespace
import unittest

from tools.build import build_source


def load_host_functions(namespace):
    tree = ast.parse(build_source())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_attention", "_fa3_attention")]
    exec(compile(tree, "<attention host functions>", "exec"), namespace)


class Tensor:
    def __init__(self, shape, dtype="bf16", device="cuda", origin=None):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.origin = origin if origin is not None else self

    def reshape(self, *shape):
        return Tensor(shape, self.dtype, self.device, self.origin)

    def expand(self, *shape):
        return Tensor(shape, self.dtype, self.device, self.origin)


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **options):
            self.calls.append((grid, args, options))
        return launch


class AttentionTests(unittest.TestCase):
    def test_only_full_layers_select_fa3_and_control_keeps_all_split(self):
        namespace = {"_split_attention": lambda data, state: "split"}
        load_host_functions(namespace)
        namespace["_fa3_attention"] = lambda data, state: "fa3"
        for backend in ("control", "full_fa3"):
            namespace["ATTENTION_BACKEND"] = backend
            for window in (0, 2048):
                actual = namespace["_attention"]({"config": {"window": window}}, {})
                self.assertEqual(actual, "fa3" if backend == "full_fa3" and window == 0 else "split")

    def test_fa3_preserves_causal_lengths_scales_metadata_and_output_boundary(self):
        for batch, length, scale in ((1, 1, 1.), (16, 1, 1.), (3, 8, 1.), (1, 8, 1. / 64)):
            with self.subTest(batch=batch, length=length, scale=scale):
                tokens = batch * length
                norm, gate, fa_calls = Kernel(), Kernel(), []
                attention_result = Tensor((tokens, 16, 64))

                def flash(**kwargs):
                    fa_calls.append(kwargs)
                    return attention_result, object()

                def project(x, state, name):
                    self.assertEqual(name, "qkv_gate")
                    return Tensor((tokens, 2304))

                namespace = {
                    "torch": SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8", empty=Tensor),
                    "_project": project, "_qkv_norm_store": norm,
                    "_attention_gate": gate, "flash_attn_with_kvcache": flash,
                }
                load_host_functions(namespace)
                cfg = dict(batch=batch, q_len=length, tokens=tokens, kv_scale=scale,
                           rope_theta=0., sm_scale=.125, qk_scale_factor=5.473)
                data = {name: object() for name in ("attn_input", "k_cache", "v_cache", "positions", "cache_loc",
                        "page_table", "cache_seqlens", "cu_seqlens_q")}
                data.update(config=cfg, scheduler_metadata=object() if length == 1 else None,
                            k_scale=Tensor((), "fp32"), v_scale=Tensor((), "fp32"))
                state = {"rope_freq": object()}
                out = namespace["_fa3_attention"](data, state)
                self.assertEqual((out.shape, out.dtype), ((tokens, 1024), "bf16"))
                grid, args, options = norm.calls[0]
                self.assertEqual(grid, (tokens, 18))
                self.assertEqual(args[1].dtype, "fp8")
                self.assertEqual(args[9], 1.)  # query factor is applied after FP8 rounding
                self.assertIs(args[2], data["k_cache"])
                self.assertIs(args[3], data["v_cache"])
                self.assertEqual(len(fa_calls), 1)
                call = fa_calls[0]
                self.assertIs(call["q"], args[1])
                self.assertEqual(call["softmax_scale"], cfg["sm_scale"] * cfg["qk_scale_factor"])
                self.assertTrue(call["causal"])
                self.assertEqual(call["window_size"], (-1, -1))
                self.assertEqual(call["max_seqlen_q"], length)
                for name in ("k_cache", "v_cache", "page_table", "cache_seqlens", "cu_seqlens_q", "scheduler_metadata"):
                    self.assertIs(call[name], data[name])
                if scale != 1.:
                    for prefix in ("k", "v"):
                        self.assertEqual(call[prefix + "_descale"].shape, (batch, 2))
                        self.assertIs(call[prefix + "_descale"].origin, data[prefix + "_scale"])
                else:
                    self.assertNotIn("k_descale", call)
                    self.assertNotIn("v_descale", call)
                self.assertEqual(gate.calls[0][0], (tokens, 16))
                self.assertIs(gate.calls[0][1][0], attention_result)
                self.assertIs(gate.calls[0][1][2], out)


if __name__ == "__main__":
    unittest.main()
