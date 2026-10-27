"""Verify the exported Marlin policy experiment and its actual selector scope."""

import ast
import copy
import unittest

from tools.build import build_source, source_manifest


def exported_selector(preset):
    tree = ast.parse(build_source(preset))
    constants = {n.targets[0].id: ast.literal_eval(n.value)
                 for n in tree.body if isinstance(n, ast.Assign)}
    experts = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_experts")
    loop = next(n for n in experts.body if isinstance(n, ast.For))
    # Execute the selector from the exported function, without the tensor work.
    probe = ast.FunctionDef(name="select", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg="num_tokens")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[copy.deepcopy(loop), ast.Return(ast.Name(id="block_size_m", ctx=ast.Load()))],
        decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    exec(compile(module, "<actual Marlin selector>", "exec"), constants)
    return constants["select"]


class ExpertPolicyTests(unittest.TestCase):
    def test_m8_changes_only_tile_policy_and_retains_fused_finalize(self):
        trees = [ast.parse(build_source(p)) for p in ("full_fa3_m16_finalize", "full_fa3_m8_finalize")]
        for tree in trees:
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BLOCK_SIZE_M_CANDIDATES" for t in node.targets):
                    node.value = ast.Constant("normalized-policy")
        self.assertEqual(ast.dump(trees[0]), ast.dump(trees[1]))
        selector = exported_selector("full_fa3_m8_finalize")
        for tokens in (1, 4, 8, 9, 16, 17, 32, 33, 58, 64, 128, 129, 256, 257, 264):
            self.assertEqual(selector(tokens), 8, tokens)
        manifest = source_manifest(build_source("full_fa3_m8_finalize"))
        self.assertEqual(manifest["preset"], "full_fa3_m8_finalize")
        self.assertIs(manifest["small_expert_finalize"], True)
        self.assertEqual(manifest["expert_block_size_candidates"], (8,))

    def test_m16_candidate_differs_only_in_block_size_candidate_tuple(self):
        before, after = (ast.parse(build_source(p)) for p in ("full_fa3", "full_fa3_m16"))
        for tree in (before, after):
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BLOCK_SIZE_M_CANDIDATES" for t in node.targets):
                    node.value = ast.Constant("normalized-policy")
        self.assertEqual(ast.dump(before), ast.dump(after))
        manifest = source_manifest(build_source("full_fa3_m16"))
        self.assertEqual(manifest["preset"], "full_fa3_m16")
        self.assertEqual(manifest["expert_block_size_candidates"], (8, 16))

    def test_actual_selector_keeps_decode_and_small_verify_unchanged(self):
        control, candidate = exported_selector("full_fa3"), exported_selector("full_fa3_m16")
        for tokens in range(1, 58):
            self.assertEqual(candidate(tokens), control(tokens), tokens)
        for tokens in (58, 64, 127, 128, 129, 255, 256, 257, 264, 512):
            self.assertEqual(candidate(tokens), 16, tokens)
        self.assertEqual(control(128), 48)
        self.assertEqual(control(256), 64)
        for tokens in (1, 8, 16, 32):
            self.assertEqual(candidate(tokens), control(tokens))


if __name__ == "__main__":
    unittest.main()
