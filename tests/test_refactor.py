"""Check launch equivalence to the checkpoint without requiring a GPU."""

import ast
import copy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import unittest

from projection.configs import CONFIGS
from projection.shapes import SHAPES, SCORED_TOKENS, shape_config
from tools.build import ROOT, build_source
from tests.host import HostCandidate


def normalize(obj):
    if is_dataclass(obj):
        return {k: normalize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (tuple, list)):
        return [normalize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    return obj


class RefactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent / "fixtures/checkpoint.json").read_text())

    def test_all_presets_preserve_checkpoint_launches(self):
        for preset, cases in self.fixture["dispatch"].items():
            candidate = HostCandidate(build_source(preset))
            for case, expected in cases.items():
                layer, m = case.split('/')
                result = candidate.run(int(m), layer)
                digest = hashlib.sha256(json.dumps(normalize(result), sort_keys=True).encode()).hexdigest()
                self.assertEqual(digest, expected, (preset, case, result))

    def test_gpu_math_and_original_preparation_are_unchanged(self):
        tree = ast.parse(build_source())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name, expected in self.fixture['kernels'].items():
            original = copy.deepcopy(functions['_prepare_weights'] if name == 'prepare' else functions[name])
            original.name = name
            actual = hashlib.sha256(ast.dump(original).encode()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_generated_submission_is_current_and_standalone(self):
        source = build_source()
        self.assertEqual((ROOT / 'submission.py').read_text(), source)
        self.assertTrue(source.startswith('# okc: problem=muse_bf16_proj\n'))
        tree = ast.parse(source)
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual(len(names), len(set(names)))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertNotIn(node.module.split('.')[0], ('projection', 'tools', 'tests', 'reference'))

    def test_scored_configs_are_complete_and_partition_k(self):
        expected = {(layer, m) for layer in SHAPES for m in SCORED_TOKENS}
        for preset, configs in CONFIGS.items():
            self.assertEqual(set(configs), expected, preset)
            for (layer, m), option in configs.items():
                if len(option) == 7:
                    _, bm, bn, bk, splits, warps, stages = option
                    self.assertEqual(shape_config(layer, m)['k'] % (splits * bk), 0)

    def test_exact_override_does_not_change_neighboring_cases(self):
        candidate = HostCandidate(build_source())
        before = candidate.run(17, 'qkv_gate')
        candidate.ns['TUNED_CONFIGS'][('qkv_gate', 16)] = ('torch',)
        self.assertEqual(candidate.run(16, 'qkv_gate')[1][0][0], 'torch.mm')
        self.assertEqual(candidate.run(17, 'qkv_gate'), before)

    def test_tma_descriptors_bind_to_current_state(self):
        candidate = HostCandidate(build_source('tma'))
        candidate.run(130, 'unpooler')
        clone = copy.deepcopy(candidate.last_state)
        candidate.calls.clear()
        candidate.ns['kernel_fn'](candidate.last_data, clone)
        wd, xd = candidate.calls[0][2][:2]
        self.assertIs(wd.base, clone['weights']['unpooler'])
        self.assertIs(xd.base, candidate.last_data['x'])

    def test_pooler_experiment_changes_only_large_pooler(self):
        control = HostCandidate(build_source('pdl'))
        experiment = HostCandidate(build_source('pooler_tma'))
        tma = HostCandidate(build_source('tma'))
        for m in (1, 7, 8, 16, 31, 32, 33, 64, 65, 127, 128, 129, 130,
                  255, 256, 257, 512, 1000):
            for layer in SHAPES:
                expected = control.run(m, layer)
                actual = experiment.run(m, layer)
                if layer == 'pooler' and 32 < m <= 256:
                    if m in (128, 256):
                        self.assertEqual(actual, tma.run(m, layer), (layer, m))
                    self.assertEqual(actual[0].shape, expected[0].shape)
                    self.assertEqual(len(actual[1]), 1)
                    self.assertEqual(actual[1][0][0], '_tma_weight_major')
                    self.assertEqual(actual[1][0][2][7], 1)  # S: no split-K
                else:
                    self.assertEqual(actual, expected, (layer, m))

    def test_pooler_splitk_scope_partitions_and_dependency(self):
        control = HostCandidate(build_source('pdl'))
        candidate = HostCandidate(build_source('pooler_splitk'))
        for m in (1, 8, 16, 32, 33, 64, 65, 127, 128, 129, 130,
                  192, 193, 255, 256, 257, 1000):
            for layer in SHAPES:
                expected = control.run(m, layer)
                actual = candidate.run(m, layer)
                if layer != 'pooler' or not 32 < m <= 256:
                    self.assertEqual(actual, expected, (layer, m))
                    continue
                out, calls = actual
                self.assertEqual(out.shape, expected[0].shape)
                self.assertEqual(out.dtype, 'bf16')
                self.assertEqual([c[0] for c in calls], ['_tma_weight_major', '_reduce'])
                _, grid, args, launch = calls[0]
                splits = args[7]
                self.assertEqual(args[6] % (splits * launch['BK']), 0)
                self.assertEqual(args[3].shape, (splits, m, 4096))
                self.assertEqual(args[3].dtype, 'fp32')
                self.assertEqual(calls[1][2][2:4], (m * 4096, splits))
                self.assertTrue(calls[1][3]['launch_pdl'])
                if m in (128, 256):
                    self.assertEqual(grid[0] * grid[1], 512)
                    self.assertEqual(splits, 4 if m == 128 else 2)


if __name__ == '__main__':
    unittest.main()
