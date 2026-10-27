"""Validate the actual decode helper, raw-weight addresses, and host dispatch."""

import ast
import copy
import struct
from types import SimpleNamespace
import unittest

from tests.test_attention import Kernel
from tests.test_expert_finalize import Tensor
from tools.build import build_source, source_manifest


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16(x):
    bits = struct.unpack("I", struct.pack("f", x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("f", struct.pack("I", bits))[0]


class Scalar:
    """Only scalar operations needed to execute the exported nibble decoder."""
    def __init__(self, value, kind="fp32"):
        self.kind = kind
        self.value = f32(value) if kind == "fp32" else value

    def to(self, kind, bitcast=False):
        if bitcast:
            return Scalar(struct.unpack("f", struct.pack("I", int(self.value)))[0])
        return Scalar(bf16(self.value) if kind == "bf16" else self.value, kind)

    def _op(self, other, function):
        value = other.value if isinstance(other, Scalar) else other
        return Scalar(function(self.value, value), self.kind)

    def __and__(self, other): return self._op(other, lambda a, b: a & b)
    def __or__(self, other): return self._op(other, lambda a, b: a | b)
    def __rshift__(self, other): return self._op(other, lambda a, b: a >> b)
    def __lshift__(self, other): return self._op(other, lambda a, b: a << b)
    def __add__(self, other): return self._op(other, lambda a, b: a + b)
    def __mul__(self, other): return self._op(other, lambda a, b: a * b)
    def __neg__(self): return Scalar(-self.value, self.kind)
    def __lt__(self, other): return self.value < other
    def __ne__(self, other): return self.value != other


def exported_functions(*names):
    return [n for n in ast.parse(build_source("single_nvfp4")).body
            if isinstance(n, ast.FunctionDef) and n.name in names]


class NVFP4Tests(unittest.TestCase):
    def test_all_e2m1_codes_and_scale_combinations(self):
        function = exported_functions("_decode_nvfp4")[0]
        function.decorator_list = []
        namespace = {"tl": SimpleNamespace(uint32="uint32", float32="fp32", bfloat16="bf16",
                                           where=lambda condition, yes, no: yes if condition else no)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<actual nibble decoder>", "exec"), namespace)
        grid = (0., .5, 1., 1.5, 2., 3., 4., 6.)
        for code in range(16):
            for sf in (.015625, .125, .5, 1., 8., 128., 448.):
                for gs in (1e-5, .00314159, .1):
                    expected = grid[code & 7] * (-1 if code & 8 else 1)
                    expected = bf16(f32(expected * f32(sf * f32(gs))))
                    actual = namespace["_decode_nvfp4"](Scalar(code, "uint32"), Scalar(sf), Scalar(gs)).value
                    self.assertEqual(struct.pack("f", actual), struct.pack("f", expected), (code, sf, gs))

    def test_raw_weight_and_scale_addresses_use_correct_expert_rows(self):
        class ScalarIndex(ast.NodeTransformer):
            def visit_Subscript(self, node):
                if isinstance(node.value, ast.Name) and node.value.id in ("row", "k"):
                    return ast.copy_location(ast.Name(id=node.value.id, ctx=ast.Load()), node)
                return self.generic_visit(node)

        for kernel, rows in (("_nvfp4_gate_up", 8192), ("_nvfp4_down", 4096)):
            function = exported_functions(kernel)[0]
            expressions = {n.targets[0].id: n.value for n in ast.walk(function)
                           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                           and n.targets[0].id in ("byte_offset", "scale_offset")}
            codes = {key: compile(ast.fix_missing_locations(ast.Expression(ScalarIndex().visit(copy.deepcopy(value)))), "<actual address>", "eval")
                     for key, value in expressions.items()}
            for expert in (0, 7, 31):
                for row in (0, 7, 4095):
                    for k in (0, 1, 15, 16, 511, 512, 4095):
                        scope = dict(expert=expert, row=row, k=k)
                        self.assertEqual(eval(codes["byte_offset"], scope), (expert * rows + row) * 2048 + k // 2)
                        self.assertEqual(eval(codes["scale_offset"], scope), (expert * rows + row) * 256 + k // 16)

    def test_direct_host_path_launches_two_gemvs_then_masked_finalize(self):
        up, down, finalize = Kernel(), Kernel(), Kernel()
        local_ids, routes = Tensor((1, 8), "int32"), Tensor((1, 8), "fp32")

        class Map:
            def __getitem__(self, index): return local_ids

        bank = {key: object() for key in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
        state = dict(raw_bank=bank, bank=dict(size_n=4096, expert_map=Map()), post_expert_w=object())
        namespace = dict(DIRECT_SINGLE_EXPERTS=True, _nvfp4_gate_up=up, _nvfp4_down=down,
                         _small_expert_finalize=finalize,
                         torch=SimpleNamespace(float32="fp32", empty=Tensor,
                                               empty_like=lambda x: Tensor(x.shape, x.dtype, x.device)))
        functions = exported_functions("_single_token_experts", "_experts")
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<direct expert host>", "exec"), namespace)
        out = namespace["_experts"](Tensor((1, 4096)), routes, Tensor((1, 8), "int32"), state)
        self.assertEqual((out.shape, out.dtype), ((1, 4096), "bf16"))
        self.assertEqual(up.calls[0][0], (512, 8))
        self.assertEqual(down.calls[0][0], (512, 8))
        self.assertIs(up.calls[0][1][1], local_ids)
        self.assertIs(up.calls[0][1][2], bank["w13"])
        self.assertIs(down.calls[0][1][2], bank["w2"])
        self.assertIs(down.calls[0][1][0], up.calls[0][1][5])
        self.assertIs(finalize.calls[0][1][0], down.calls[0][1][5])
        self.assertIs(finalize.calls[0][1][1], routes)
        self.assertIs(finalize.calls[0][1][2], local_ids)
        self.assertIs(finalize.calls[0][1][4], out)

    def test_only_direct_flag_differs_from_regenerated_control(self):
        trees = [ast.parse(build_source(p)) for p in ("full_fa3_m16_finalize", "single_nvfp4")]
        for tree in trees:
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DIRECT_SINGLE_EXPERTS" for t in node.targets):
                    node.value = ast.Constant("normalized-flag")
        self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))
        manifest = source_manifest(build_source("single_nvfp4"))
        self.assertEqual(manifest["preset"], "single_nvfp4")
        self.assertEqual(manifest["expert_block_size_candidates"], (8, 16))


if __name__ == "__main__":
    unittest.main()
