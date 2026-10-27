"""Stress the overlapping MMA ring's buffer ownership before GPU validation."""

import ast
from pathlib import Path
import random
import unittest

from projection.shapes import SHAPES, SCORED_TOKENS
from tests.host import HostCandidate
from tools.build import build_source


class HopperOverlapTests(unittest.TestCase):
    def test_only_four_large_cases_change(self):
        control = HostCandidate(build_source('router_reduce'))
        candidate = HostCandidate(build_source('hopper_overlap'))
        for layer in SHAPES:
            for m in (*SCORED_TOKENS, 7, 9, 33, 130, 512):
                before = control.run(m, layer)
                after = candidate.run(m, layer)
                if layer not in ('qkv_gate', 'pooler') or m <= 32:
                    self.assertEqual(before, after)
                else:
                    self.assertEqual(after[0].shape, before[0].shape)
                    self.assertEqual(after[0].dtype, before[0].dtype)
                    self.assertEqual(after[1][0][0], '_hopper_overlap_gemm')
                    self.assertEqual(after[1][0][3]['BUFFERS'], 2)

    def test_delayed_release_never_overwrites_an_outstanding_operand(self):
        tree = ast.parse(Path('projection/kernels/hopper_overlap.py').read_text())
        compute = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        loop = next(n for n in compute.body if isinstance(n, ast.For))
        assignments = {n.targets[0].id: compile(ast.Expression(n.value), '<ring>', 'eval')
                       for n in loop.body if isinstance(n, ast.Assign)
                       and n.targets[0].id in ('slot', 'phase', 'retired')}
        release = next(n.value for n in loop.body if isinstance(n, ast.Expr)
                       and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                       and n.value.func.attr == 'arrive')
        pred = compile(ast.Expression(next(k.value for k in release.keywords if k.arg == 'pred')),
                       '<release predicate>', 'eval')
        for buffers in (2, 3, 4):
            for steps in (1, 2, 3, 9, 32, 65):
                for seed in range(30):
                    rng = random.Random(seed)
                    owners = [None] * buffers
                    free, ready = [1] * buffers, [1] * buffers
                    transfers, pending, mma, done = {}, [], [], set()
                    producer = consumer = 0
                    issued = False
                    def coords(step):
                        env = dict(step=step, BUFFERS=buffers)
                        return {key: eval(expr, {}, env) for key, expr in assignments.items()}
                    while consumer < steps:
                        events = []
                        if producer < steps:
                            c = coords(producer)
                            if free[c['slot']] == c['phase'] ^ 1:
                                events.append(('load', producer))
                        events.extend(('dma', task) for task in pending)
                        if mma:
                            events.append(('finish', mma[0]))
                        c = coords(consumer)
                        if not issued and ready[c['slot']] == c['phase']:
                            events.append(('issue', consumer))
                        if issued and (consumer == 0 or consumer - 1 in done):
                            events.append(('release', consumer))
                        self.assertTrue(events, 'pipeline deadlock')
                        action, step = rng.choice(events)
                        if action == 'load':
                            c = coords(step)
                            self.assertIsNone(owners[c['slot']], 'overwrote an operand before MMA retirement')
                            owners[c['slot']] = step
                            transfers[step] = 0
                            pending.extend(((step, 'W'), (step, 'X')))
                            producer += 1
                        elif action == 'dma':
                            pending.remove(step)
                            iteration, _ = step
                            transfers[iteration] += 1
                            if transfers[iteration] == 2:
                                c = coords(iteration)
                                ready[c['slot']] = c['phase']
                        elif action == 'issue':
                            self.assertEqual(owners[c['slot']], step)
                            self.assertEqual(transfers[step], 2)
                            mma.append(step)
                            self.assertLessEqual(len(mma), 2)
                            issued = True
                        elif action == 'finish':
                            done.add(mma.pop(0))
                        else:
                            if eval(pred, {}, dict(step=step)):
                                slot = c['retired']
                                self.assertEqual(owners[slot], step - 1)
                                self.assertIn(step - 1, done)
                                owners[slot] = None
                                free[slot] = coords(step - 1)['phase']
                            consumer += 1
                            issued = False
                    # The source drains all groups before releasing the last slot.
                    done.update(mma)
                    slot = (steps - 1) % buffers
                    self.assertEqual(owners[slot], steps - 1)
                    owners[slot] = None
                    self.assertEqual(done, set(range(steps)))
                    self.assertEqual(owners, [None] * buffers)
                    self.assertEqual(pending, [])


if __name__ == '__main__':
    unittest.main()
