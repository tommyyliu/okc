"""Cluster-router host scope, source addresses and partitioned hi/lo arithmetic.

These CPU checks cannot validate compiler layout lowering or GPU synchronization.
"""

import ast
import copy
import math
from pathlib import Path
import random
import unittest

from projection.shapes import SHAPES, SCORED_TOKENS
from tests.host import HostCandidate
from tests.test_round4 import bf16, fp32
from tools.build import build_source
from tools.codegen import compile_jobs
from tools.tune import read_source_choices


class Lane(int):
    def __getitem__(self, _):
        return int(self)


class RouterClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path('projection/kernels/router_cluster.py').read_text())
        cls.kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        loop = next(n for n in cls.kernel.body if isinstance(n, ast.For))
        cls.loop_exprs = {n.targets[0].id: compile(ast.Expression(n.value), '<cluster address>', 'eval')
                          for n in loop.body if isinstance(n, ast.Assign)
                          and n.targets[0].id in ('k', 'woffsets', 'xoffsets')}
        cls.out_exprs = {n.targets[0].id: compile(ast.Expression(n.value), '<cluster epilogue>', 'eval')
                         for n in cls.kernel.body if isinstance(n, ast.Assign)
                         and n.targets[0].id in ('offsets', 'mask')}

    def test_only_small_router_changes_and_it_allocates_only_the_output(self):
        control = HostCandidate(build_source('router_reduce'))
        candidate = HostCandidate(build_source('router_cluster'))
        allocate = candidate.ns['torch'].empty
        allocations = []

        def record_empty(shape, device, dtype):
            allocations.append((shape, dtype))
            return allocate(shape, device=device, dtype=dtype)

        candidate.ns['torch'].empty = record_empty
        for layer in SHAPES:
            for m in (*SCORED_TOKENS, 2, 7, 9, 15, 17, 31, 33, 130, 512):
                allocations.clear()
                before = control.run(m, layer)
                out, calls = candidate.run(m, layer)
                if layer != 'router' or not 1 < m <= 32:
                    self.assertEqual((out, calls), before, (layer, m))
                    continue
                bm = 16 if m <= 16 else 32
                self.assertEqual(allocations, [((m, 256), 'fp32')])
                self.assertEqual(out, before[0])
                self.assertEqual(len(calls), 1)
                name, grid, args, opts = calls[0]
                self.assertEqual(name, '_router_cluster')
                self.assertEqual(grid, (16,))
                self.assertEqual(opts, dict(BM=bm, BN=16, BK=256, C=8,
                                           num_ctas=8, num_warps=4, num_stages=1))
                self.assertEqual(args[1].dtype, 'bf16')
                self.assertEqual(args[2].dtype, 'bf16')

    def test_jit_avoids_membership_operator_rejected_by_pinned_frontend(self):
        # Job 2d1ffa9571c8437d failed before code generation on `C in (...)`.
        # The same operator in the later BM assertion would fail next.
        unsupported = [op for node in ast.walk(self.kernel) if isinstance(node, ast.Compare)
                       for op in node.ops if isinstance(op, (ast.In, ast.NotIn))]
        self.assertEqual(unsupported, [])

    def test_prepared_pointers_and_activation_are_current_after_cloning(self):
        candidate = HostCandidate(build_source('router_cluster'))
        for m in (8, 9, 17, 32):
            candidate.run(m, 'router')
            state = copy.deepcopy(candidate.last_state)
            data = copy.deepcopy(candidate.last_data)
            candidate.calls.clear()
            candidate.ns['kernel_fn'](data, state)
            args = candidate.calls[0][2]
            self.assertIs(args[0], data['x'])
            self.assertIs(args[1], state['router_hi'])
            self.assertIs(args[2], state['router_lo'])

    def test_source_k_partitions_cover_all_weights_once_per_output_tile(self):
        k, c, bk = 8192, 8, 256
        visited = []
        for split in range(c):
            for step in range(k // (c * bk)):
                for lane in range(bk):
                    env = dict(K=k, C=c, BK=bk, slices=Lane(split), step=step, ks=Lane(lane))
                    value = eval(self.loop_exprs['k'], {}, env)
                    self.assertTrue(split * (k // c) <= value < (split + 1) * (k // c))
                    visited.append(value)
                    env.update(k=value, ns=Lane(7), ms=Lane(5))
                    self.assertEqual(eval(self.loop_exprs['woffsets'], {}, env), 7 * k + value)
                    self.assertEqual(eval(self.loop_exprs['xoffsets'], {}, env), 5 * k + value)
        self.assertEqual(sorted(visited), list(range(k)))

    def test_actual_epilogue_covers_odd_tiles_without_overlapping_programs(self):
        for m in (2, 7, 8, 9, 16, 17, 31, 32):
            bm = 16 if m <= 16 else 32
            for n in (17, 256):
                writes = [0] * (m * n)
                for tile in range(math.ceil(m / bm) * math.ceil(n / 16)):
                    mt = math.ceil(m / bm)
                    m0, n0 = (tile % mt) * bm, (tile // mt) * 16
                    for nn in range(n0, n0 + 16):
                        for mm in range(m0, m0 + bm):
                            env = dict(M=m, N=n, out_m=Lane(mm), out_n=Lane(nn))
                            if eval(self.out_exprs['mask'], {}, env):
                                writes[eval(self.out_exprs['offsets'], {}, env)] += 1
                self.assertEqual(writes, [1] * len(writes))

    def test_partitioned_hi_lo_math_preserves_probe_and_outlier_accuracy(self):
        rng = random.Random(2741)
        k, c = 1024, 8
        weights = [[fp32(rng.gauss(0, 1 / math.sqrt(k))) for _ in range(k)] for _ in range(3)]
        hi = [[bf16(v) for v in row] for row in weights]
        lo = [[bf16(fp32(v - h)) for v, h in zip(row, high)] for row, high in zip(weights, hi)]
        for row_index in range(9):
            if row_index % 3 == 0:
                x = [0.0] * k
                x[rng.randrange(k)] = 2.0 ** rng.randrange(-3, 4)
            else:
                x = [bf16(rng.gauss(0, 1) * (30 if j % 100 == 0 else 1)) for j in range(k)]
            actual, reference = [], []
            for original, high, low in zip(weights, hi, lo):
                partials = []
                for split in range(c):
                    ah = al = 0.0
                    for kk in range(split * (k // c), (split + 1) * (k // c)):
                        ah = fp32(ah + x[kk] * high[kk])
                        al = fp32(al + x[kk] * low[kk])
                    partials.append(fp32(ah + al))
                while len(partials) > 1:
                    partials = [fp32(partials[j] + partials[j + 1])
                                for j in range(0, len(partials), 2)]
                actual.append(partials[0])
                reference.append(math.fsum(v * w for v, w in zip(x, original)))
            relative = math.sqrt(sum((a - r) ** 2 for a, r in zip(actual, reference)) /
                                 sum(r * r for r in reference))
            self.assertLess(relative, 1e-4)

    def test_exported_compile_manifest_has_one_cluster_kernel(self):
        choices = read_source_choices(build_source('router_cluster'), ('router',), (8, 9, 16, 32))
        for m in (8, 9, 16, 32):
            jobs = compile_jobs(layers=('router',), tokens=(m,), overrides=choices)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['kernel'], '_router_cluster')
            self.assertEqual(jobs[0]['language'], 'gluon')
            self.assertEqual(jobs[0]['num_ctas'], 8)
            self.assertEqual(jobs[0]['constants']['C'], 8)
            self.assertNotIn('S', jobs[0]['constants'])
            self.assertNotIn('P', jobs[0]['signature'])
            self.assertNotIn('launch_pdl', jobs[0])


if __name__ == '__main__':
    unittest.main()
