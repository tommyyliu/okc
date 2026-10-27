"""Host contracts and asynchronous ring ownership; these are not GPU validation."""

import ast
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

from projection.configs import BANKED_CONFIGS, HOPPER_WS_CONFIGS
from projection.shapes import SHAPES, SCORED_TOKENS
from tests.host import HostCandidate, Tensor
from tests.test_round5 import ScalarIndex
from tools.build import build_source
from tools.codegen import compile_jobs
from tools.tune import read_source_choices

ROOT = Path(__file__).resolve().parents[1]


class HopperWSTests(unittest.TestCase):
    def setUp(self):
        self.source = build_source('hopper_ws')
        self.functions = {n.name: n for n in ast.parse(self.source).body
                          if isinstance(n, ast.FunctionDef)}

    def test_four_changes_preserve_selected_control(self):
        banked = HostCandidate(build_source('banked'))
        candidate = HostCandidate(self.source)
        # Historical export is an additional independent dispatch control when present.
        old = ROOT / 'build/multi-category-round5/isolated/gemv8.py'
        old_choices = read_source_choices(old.read_text()) if old.exists() else None
        changed = []
        for layer in SHAPES:
            for m in SCORED_TOKENS:
                key = layer, m
                if old_choices is not None:
                    self.assertEqual(BANKED_CONFIGS[key], old_choices[key])
                if BANKED_CONFIGS[key] == HOPPER_WS_CONFIGS[key]:
                    self.assertEqual(banked.run(m, layer), candidate.run(m, layer))
                else:
                    changed.append(key)
                    _, calls = candidate.run(m, layer)
                    self.assertEqual(calls[0][0], '_hopper_ws_gemm')
                    splits = HOPPER_WS_CONFIGS[key][4]
                    self.assertEqual([c[0] for c in calls], ['_hopper_ws_gemm'] +
                                     (['_reduce'] if splits > 1 else []))
        self.assertEqual(changed, [('qkv_gate', 128), ('qkv_gate', 256),
                                   ('pooler', 128), ('pooler', 256)])

    def test_current_descriptors_nearby_partitions_and_preparation(self):
        candidate = HostCandidate(self.source)
        weights = {layer: Tensor(layer, shape, 'fp32' if layer == 'router' else 'bf16')
                   for layer, shape in SHAPES.items()}
        prepared = candidate.ns['prepare'](weights)
        self.assertEqual(set(prepared['weights_transposed']), {'unpooler'})
        for layer in ('qkv_gate', 'pooler'):
            for m in (33, 63, 65, 127, 128, 129, 130, 256, 257, 512, 1000):
                out, calls = candidate.run(m, layer)
                wd, xd, y, partial, mm, n, k, splits = calls[0][2]
                opts = calls[0][3]
                self.assertEqual(out.shape, (m, n))
                self.assertEqual(out.dtype, 'bf16')
                self.assertEqual(opts, dict(BM=64, BN=64, BK=128, BUFFERS=2,
                                           num_warps=4, num_stages=1, maxnreg=96))
                self.assertEqual(wd.shape, [n, k])
                self.assertEqual(xd.shape, [m, k])
                self.assertEqual(wd.layout.swizzle_byte_width, 128)
                self.assertEqual(xd.layout, wd.layout)
                self.assertEqual(sorted(split * (k // splits) + step * 128 + lane
                                        for split in range(splits)
                                        for step in range(k // (splits * 128))
                                        for lane in range(128)), list(range(k)))
                self.assertEqual(partial.dtype, 'fp32' if splits > 1 else 'bf16')
                self.assertEqual(calls[0][1], (((m + 63) // 64) * (n // 64), splits))
                jobs = compile_jobs('hopper_ws', (layer,), (m,))
                self.assertEqual(jobs[0]['signature']['WD'], wd.__mangle__())
                self.assertEqual(jobs[0]['language'], 'gluon')
                if splits > 1:
                    self.assertTrue(calls[-1][3]['launch_pdl'])
                state = copy.deepcopy(candidate.last_state)
                data = copy.deepcopy(candidate.last_data)
                data['x'].name = 'fresh activation'
                state['weights'][layer].name = 'current cloned weight'
                candidate.calls.clear()
                candidate.ns['kernel_fn'](data, state)
                self.assertIs(candidate.calls[0][2][0].base, state['weights'][layer])
                self.assertIs(candidate.calls[0][2][1].base, data['x'])

    def test_actual_epilogue_addresses_mask_odd_sizes(self):
        kernel = self.functions['_hopper_ws_compute']
        assignments = {n.targets[0].id: n.value for n in kernel.body if isinstance(n, ast.Assign)}
        offsets = compile(ast.Expression(assignments['offsets']), '<GPU offsets>', 'eval')
        mask = compile(ast.Expression(assignments['mask']), '<GPU mask>', 'eval')
        # Walk every logical lane of full and partial tiles, using source expressions.
        for m in (33, 65, 129, 257):
            n = 70
            writes = [0] * (m * n)
            for m0 in range(0, m, 64):
                for n0 in range(0, n, 64):
                    for row in range(m0, m0 + 64):
                        for col in range(n0, n0 + 64):
                            ns = dict(ms=ScalarIndex(row), ns=ScalarIndex(col), M=m, N=n)
                            if eval(mask, {}, ns):
                                writes[eval(offsets, {}, ns)] += 1
            self.assertEqual(writes, [1] * len(writes))
        text = ast.unparse(kernel)
        self.assertIn('gl.zeros((BN, BM), gl.float32', text)
        self.assertIn('xbuf.index(slot).permute((1, 0))', text)
        self.assertIn('P + split * M * N + offsets', text)

    def test_barrier_protocol_under_random_interleavings(self):
        loader = self.functions['_hopper_ws_load']
        compute = self.functions['_hopper_ws_compute']
        loop = next(n for n in loader.body if isinstance(n, ast.For))
        assignments = {n.targets[0].id: n.value for n in loop.body if isinstance(n, ast.Assign)}
        slot_expr = compile(ast.Expression(assignments['slot']), '<GPU ring slot>', 'eval')
        phase_expr = compile(ast.Expression(assignments['phase']), '<GPU ring phase>', 'eval')
        compute_loop = next(n for n in compute.body if isinstance(n, ast.For))
        compute_assignments = {n.targets[0].id: n.value for n in compute_loop.body if isinstance(n, ast.Assign)}
        for key in ('slot', 'phase'):
            self.assertEqual(ast.dump(assignments[key]), ast.dump(compute_assignments[key]))
        body = ast.unparse(compute_loop)
        order = [body.index(s) for s in ('mbarrier.wait(', 'token = warpgroup_mma(',
                                         'acc = warpgroup_mma_wait(num_outstanding=0',
                                         'mbarrier.arrive(')]
        self.assertEqual(order, sorted(order))
        self.assertLess(ast.unparse(compute).index('griddepcontrol.launch_dependents;'),
                        ast.unparse(compute).index('gl.store('))
        self.assertIn('], [1], [24])', ast.unparse(self.functions['_hopper_ws_gemm']))
        self.assertIn('phase ^ 1', ast.unparse(loader))
        for buffers in (2, 3):
            for steps in (1, 2, 3, 9, 32, 64, 128):
                for seed in range(40):
                    rng = random.Random(seed)
                    # Initial phase 1 is complete. Ownership persists through MMA completion.
                    free, ready = [1] * buffers, [1] * buffers
                    owners = [None] * buffers
                    pending = []
                    transfers = {}
                    producer = consumer = 0
                    active = None
                    completed = []
                    def coords(step):
                        env = dict(step=step, BUFFERS=buffers)
                        return eval(slot_expr, {}, env), eval(phase_expr, {}, env)
                    while consumer < steps:
                        choices = []
                        if producer < steps:
                            slot, phase = coords(producer)
                            if free[slot] == (phase ^ 1):
                                choices.append(('produce', producer))
                        choices.extend(('dma', transfer) for transfer in pending)
                        if active is None:
                            slot, phase = coords(consumer)
                            if ready[slot] == phase:
                                choices.append(('mma', consumer))
                        else:
                            choices.append(('finish', active))
                        self.assertTrue(choices, 'protocol deadlock')
                        action, step = rng.choice(choices)
                        if action == 'dma':
                            pending.remove(step)
                            iteration, operand = step
                            transfers[iteration] += 1
                            if transfers[iteration] == 2:
                                slot, phase = coords(iteration)
                                ready[slot] = phase
                        elif action == 'produce':
                            slot, phase = coords(step)
                            self.assertIsNone(owners[slot], 'premature overwrite')
                            owners[slot] = step
                            transfers[step] = 0
                            pending.extend(((step, 'W'), (step, 'X')))
                            producer += 1
                        elif action == 'mma':
                            slot, phase = coords(step)
                            self.assertEqual(owners[slot], step)
                            self.assertEqual(transfers[step], 2)
                            active = step
                        else:
                            slot, phase = coords(step)
                            self.assertEqual(owners[slot], step)
                            owners[slot] = None
                            free[slot] = phase
                            completed.append(step)
                            consumer += 1
                            active = None
                    self.assertEqual(completed, list(range(steps)))
                    self.assertEqual(producer, steps)
                    self.assertEqual(pending, [])
                    self.assertEqual(owners, [None] * buffers)

    def test_scalar_resource_diagnostics_are_deduplicated_and_optional(self):
        candidate = HostCandidate(self.source)
        compiled = SimpleNamespace(n_regs=80, n_spills=0,
                                   metadata=SimpleNamespace(name='test_kernel', shared=65568, num_warps=4))
        stream = io.StringIO()
        with redirect_stdout(stream):
            candidate.ns['_report_resources'](compiled, 'pooler', 128, 'hopper_ws')
            candidate.ns['_report_resources'](compiled, 'pooler', 128, 'hopper_ws')
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0].removeprefix('OKC_RESOURCE '))
        self.assertEqual(record['registers'], 80)
        self.assertEqual(record['spills'], 0)
        default = HostCandidate(build_source())
        with redirect_stdout(stream):
            default.ns['_report_resources'](compiled, 'pooler', 128, 'tma')
        self.assertEqual(len(stream.getvalue().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
