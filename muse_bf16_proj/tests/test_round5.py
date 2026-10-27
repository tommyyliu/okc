"""Round-five scheduling contracts and CPU router address/math validation."""

import ast
import copy
import math
import random
import unittest

from projection.configs import KEEPER_CONFIGS, ROUND5_CONFIGS
from projection.shapes import SHAPES, SCORED_TOKENS
from tests.host import HostCandidate, Tensor
from tests.test_round4 import bf16, fp32
from tools.build import build_source


class ScalarIndex:
    """Evaluate a GPU vector's address expression at one logical element."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, _):
        return self.value

    def __mul__(self, other):
        return self.value * other

    def __radd__(self, other):
        return other + self.value

    def __lt__(self, other):
        return self.value < other


class Round5Tests(unittest.TestCase):
    def test_unroll_changes_only_loop_compiler_hint(self):
        tree = ast.parse(build_source('round5'))
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        old = functions['_weight_major_stream_gemm']
        new = copy.deepcopy(functions['_weight_major_unroll2_gemm'])
        old_loop = next(n for n in old.body if isinstance(n, ast.For))
        new_loop = next(n for n in new.body if isinstance(n, ast.For))
        self.assertEqual(ast.unparse(new_loop.iter.func), 'tl.range')
        self.assertEqual([(k.arg, k.value.value) for k in new_loop.iter.keywords],
                         [('loop_unroll_factor', 2)])
        self.assertEqual(ast.dump(new_loop.iter.args[0]), ast.dump(old_loop.iter.args[0]))
        new_loop.iter = copy.deepcopy(old_loop.iter)
        new.name = old.name
        self.assertEqual(ast.dump(new), ast.dump(old))

    def test_thirteen_scored_changes_are_disjoint_and_others_keep_control_calls(self):
        before, after = HostCandidate(build_source('keepers')), HostCandidate(build_source('round5'))
        changed = []
        for layer in SHAPES:
            for m in SCORED_TOKENS:
                key = layer, m
                old, new = KEEPER_CONFIGS[key], ROUND5_CONFIGS[key]
                if old == new:
                    self.assertEqual(before.run(m, layer), after.run(m, layer))
                    continue
                changed.append(key)
                if m == 1:
                    self.assertEqual(new, ('gemv', 1, 8))
                    self.assertEqual(old, ('gemv', 1, 4))
                    _, calls = after.run(m, layer)
                    self.assertEqual(calls[0][0], '_gemv_full')
                    self.assertEqual(calls[0][3]['num_warps'], 8)
                elif layer == 'router':
                    self.assertEqual(new, ('router_direct', 2, 8))
                else:
                    self.assertEqual(new[0], 'weight_major_unroll2')
                    self.assertEqual(new[1:], old[1:])
        self.assertEqual(len(changed), 13)

    def test_direct_router_has_one_allocation_one_launch_and_original_fp32_weights(self):
        candidate = HostCandidate(build_source('round5'))
        allocations = []
        empty = candidate.ns['torch'].empty

        def record_empty(*args, **kwargs):
            result = empty(*args, **kwargs)
            allocations.append(result)
            return result

        candidate.ns['torch'].empty = record_empty
        for m in (2, 8, 9, 16, 17, 32):
            allocations.clear()
            out, calls = candidate.run(m, 'router')
            self.assertEqual(len(allocations), 1)
            self.assertEqual([c[0] for c in calls], ['_router_direct'])
            self.assertEqual(calls[0][1], (256, (m + 1) // 2))
            self.assertEqual(calls[0][2][1].dtype, 'fp32')
            self.assertEqual(out.shape, (m, 256))
            self.assertEqual(out.dtype, 'fp32')
            self.assertEqual(calls[0][3]['BLOCK_K'], 8192)
            state = copy.deepcopy(candidate.last_state)
            state['weights']['router'].name = 'current_fp32_clone'
            candidate.calls.clear()
            candidate.ns['kernel_fn'](candidate.last_data, state)
            self.assertIs(candidate.calls[0][2][1], state['weights']['router'])
        candidate.ns['TUNED_CONFIGS']['pooler', 8] = ('router_direct', 2, 8)
        with self.assertRaisesRegex(ValueError, 'FP32 router weights'):
            candidate.run(8, 'pooler')

    def test_direct_router_actual_addresses_masks_and_fp32_reduction_on_odd_rows(self):
        tree = ast.parse(build_source('round5'))
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_router_direct')
        assignments = {n.targets[0].id: n.value for n in kernel.body if isinstance(n, ast.Assign)}
        store = next(n.value for n in kernel.body if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'tl.store')
        expressions = {name: compile(ast.Expression(assignments[name]), '<router GPU expression>', 'eval')
                       for name in ('woffsets', 'xoffsets', 'offsets', 'products')}
        masks = {}
        for name in ('w', 'x'):
            cast = assignments[name]
            self.assertEqual(ast.unparse(cast.args[0]), 'tl.float32')
            masks[name] = compile(ast.Expression(cast.func.value.args[1]), '<router load mask>', 'eval')
        masks['y'] = compile(ast.Expression(store.args[2]), '<router store mask>', 'eval')
        self.assertEqual(ast.unparse(assignments['sums']), 'tl.sum(products, axis=1)')
        rng = random.Random(9173)
        n, k, block_k = 5, 13, 16
        weights = [fp32(rng.uniform(-0.5, 0.5)) for _ in range(n * k)]
        for m in (3, 9, 17):
            inputs = [bf16(rng.uniform(-4, 4) * (1000 if row == 1 else 1))
                      for row in range(m) for _ in range(k)]
            inputs[:k] = [0.0] * k
            inputs[5] = 8.0
            inputs[(m - 2) * k:(m - 1) * k] = [0.0] * k
            output, writes = [None] * (m * n), [0] * (m * n)
            for expert in range(n):
                for tile in range((m + 1) // 2):
                    # One weight vector is shared by both token reductions.
                    wblock = []
                    for lane_k in range(block_k):
                        ns = dict(expert=expert, N=n, K=k, ks=ScalarIndex(lane_k))
                        valid = eval(masks['w'], {}, ns)
                        index = eval(expressions['woffsets'], {}, ns)
                        wblock.append(weights[index] if valid else 0.0)
                    for local_m in range(2):
                        row = tile * 2 + local_m
                        products = []
                        for lane_k in range(block_k):
                            ns = dict(expert=expert, M=m, N=n, K=k,
                                      ms=ScalarIndex(row), ks=ScalarIndex(lane_k))
                            valid = eval(masks['x'], {}, ns)
                            index = eval(expressions['xoffsets'], {}, ns)
                            value = fp32(inputs[index]) if valid else 0.0
                            products.append(fp32(eval(expressions['products'], {},
                                                      dict(x=value, w=ScalarIndex(wblock[lane_k])))))
                        while len(products) > 1:
                            products = [fp32(products[i] + products[i + 1]) for i in range(0, len(products), 2)]
                        ns = dict(M=m, N=n, expert=expert, ms=ScalarIndex(row))
                        if eval(masks['y'], {}, ns):
                            index = eval(expressions['offsets'], {}, ns)
                            writes[index] += 1
                            output[index] = products[0]
            self.assertEqual(writes, [1] * (m * n))
            reference = [math.fsum(inputs[row * k + col] * weights[expert * k + col] for col in range(k))
                         for row in range(m) for expert in range(n)]
            error = math.sqrt(sum((x - y)**2 for x, y in zip(output, reference)))
            norm = math.sqrt(sum(y**2 for y in reference))
            self.assertLess(error / norm, 1e-4)
            self.assertEqual(output[:n], [fp32(8 * weights[e * k + 5]) for e in range(n)])

    def test_round_five_preparation_uses_only_selected_weight_transposes(self):
        candidate = HostCandidate(build_source('round5'))
        weights = {layer: Tensor(layer, shape, 'fp32' if layer == 'router' else 'bf16')
                   for layer, shape in SHAPES.items()}
        state = candidate.ns['prepare'](weights)
        self.assertEqual(set(state), {'weights', 'router_hi', 'router_lo', 'weights_transposed'})
        self.assertEqual(set(state['weights_transposed']), {'pooler', 'unpooler'})


if __name__ == '__main__':
    unittest.main()
