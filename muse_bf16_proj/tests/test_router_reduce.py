"""Scope and address coverage for the small-router reduction diagnostic."""

import ast
from pathlib import Path
import random
import unittest

from projection.shapes import SHAPES, SCORED_TOKENS
from tests.host import HostCandidate
from tools.build import build_source
from tools.codegen import compile_jobs
from tools.tune import read_source_choices


class RouterReductionTests(unittest.TestCase):
    def test_only_small_router_reducer_geometry_changes(self):
        control = HostCandidate(build_source('banked'))
        candidate = HostCandidate(build_source('router_reduce'))
        for layer in SHAPES:
            for m in (*SCORED_TOKENS, 2, 7, 9, 15, 17, 31, 33, 127, 130, 257):
                expected = control.run(m, layer)
                actual = candidate.run(m, layer)
                if layer != 'router' or not 1 < m <= 32:
                    self.assertEqual(actual, expected, (layer, m))
                    continue
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[1][0], expected[1][0])
                self.assertEqual(actual[1][1][0], '_reduce')
                self.assertEqual(actual[1][1][2], expected[1][1][2])
                self.assertEqual(actual[1][1][1], (m * 256 // 32,))
                self.assertEqual(actual[1][1][3], dict(B=32, num_warps=4, launch_pdl=True))
                self.assertEqual(expected[1][1][3], dict(B=256, num_warps=4, launch_pdl=True))

    def test_standalone_source_inspection_retains_reducer_choice(self):
        source = build_source('router_reduce')
        choices = read_source_choices(source, ('router',), (2, 8, 9, 16, 32, 33))
        # Compile manifests made from --source use the default preset and overrides.
        # The reducer choice must survive in the exported option itself.
        for m in (2, 8, 9, 16, 32, 33):
            jobs = compile_jobs(layers=('router',), tokens=(m,), overrides=choices)
            self.assertEqual(jobs[0]['kernel'], '_gemm')
            self.assertTrue(jobs[0]['constants']['ROUTER'])
            self.assertEqual(jobs[1]['constants']['B'], 32 if m <= 32 else 256)
            self.assertTrue(jobs[1]['launch_pdl'])

    def test_actual_reducer_addresses_cover_odd_tails_and_all_splits(self):
        tree = ast.parse(Path('projection/kernels/reduce.py').read_text())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        load = next(n for n in ast.walk(kernel) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'load')
        store = next(n for n in ast.walk(kernel) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'store')
        load_address = compile(ast.Expression(load.args[0]), '<partial address>', 'eval')
        load_mask = compile(ast.Expression(load.args[1]), '<partial mask>', 'eval')
        store_address = compile(ast.Expression(store.args[0]), '<output address>', 'eval')
        store_mask = compile(ast.Expression(store.args[2]), '<output mask>', 'eval')

        class Lane(int):
            def __getitem__(self, _):
                return int(self)

        rng = random.Random(9173)
        # Tiny and non-aligned totals exercise masking beyond scored N256 cases.
        for total in (1, 31, 32, 33, 257, 8 * 256, 9 * 256, 32 * 256):
            partials = [rng.randrange(-64, 65) for _ in range(32 * total)]
            output = [None] * total
            for block in range((total + 31) // 32):
                for lane in range(32):
                    env = dict(P=0, Y=0, TOTAL=total, offsets=Lane(block * 32 + lane))
                    values = []
                    for split in range(32):
                        env['splits'] = Lane(split)
                        if eval(load_mask, {}, env):
                            address = eval(load_address, {}, env)
                            self.assertTrue(0 <= address < len(partials))
                            values.append(partials[address])
                    if eval(store_mask, {}, env):
                        address = eval(store_address, {}, env)
                        self.assertIsNone(output[address], 'overlapping output blocks')
                        output[address] = sum(values)
            self.assertEqual(output, [sum(partials[s * total + i] for s in range(32))
                                      for i in range(total)])


if __name__ == '__main__':
    unittest.main()
