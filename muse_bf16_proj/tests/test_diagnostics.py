"""CPU tests for diagnostic accuracy; these do not compile or time GPU kernels."""
import ast
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from tools import codegen as inspect_codegen
from tools import budget as projection_budget
HERE = Path(__file__).resolve().parents[1]

class AssemblyReportTests(unittest.TestCase):

    def test_predicated_instructions_and_comments(self):
        ptx = '\n        // ld.global.u8 %r1, [%rd1];\n        /* st.local.b32 [%rd0], %r0; */\n        @%p1 ld.global.u8 %r2, [%rd2];\n        @!%p2 ld.global.v4.b32 {%r3,%r4,%r5,%r6}, [%rd3];\n        st.local.b32 [%rd4], %r5;\n        ld.local.b32 %r6, [%rd4];\n        wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 %r0, %rd0;\n        cp.async.ca.shared.global [%r0], [%rd0], 16;\n        cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%r0], [%rd0, {%r1,%r2}], [%r3];\n        griddepcontrol.wait;\n        griddepcontrol.launch_dependents;\n        '
        result = inspect_codegen.summarize_ptx(ptx, '%1 = ttg.convert_layout %0')
        self.assertEqual(result['global_load_instructions'], {'ld.global.u8': 1, 'ld.global.v4.b32': 1})
        self.assertEqual(result['local_memory_instructions'], {'st.local.b32': 1, 'ld.local.b32': 1})
        self.assertEqual(sum(result['mma_instructions'].values()), 1)
        self.assertEqual(sum(result['cp_async_instructions'].values()), 2)
        self.assertEqual(result['grid_dependency_instructions'], {'griddepcontrol.wait': 1, 'griddepcontrol.launch_dependents': 1})
        self.assertEqual(result['layout_conversions_in_ir'], 1)

    def test_assembler_resources_distinguish_unknown_and_zero(self):
        text = 'ptxas info : 32 bytes stack frame, 8 bytes spill stores, 16 bytes spill loads\nptxas info : Used 128 registers, 49152 bytes smem\n'
        self.assertEqual(inspect_codegen.parse_ptxas(text), {'registers_per_thread': 128, 'stack_frame_bytes': 32, 'spill_store_bytes': 8, 'spill_load_bytes': 16})
        self.assertTrue(all((v is None for v in inspect_codegen.parse_ptxas('').values())))
        self.assertEqual(inspect_codegen.parse_ptxas('0 bytes spill loads')['spill_load_bytes'], 0)

    def test_original_cubin_resources(self):
        result = inspect_codegen.parse_cubin_resources('REG:64 STACK:0 SHARED:49152 LOCAL:16 CONSTANT[0]:384')
        self.assertEqual(result, {'registers_per_thread': 64, 'stack_bytes': 0, 'shared_bytes': 49152, 'local_bytes': 16})

class CompilationManifestTests(unittest.TestCase):

    def test_jit_extraction_does_not_execute_host_code(self):
        with TemporaryDirectory() as d:
            p = Path(d) / 'source.py'
            p.write_text('import torch\nraise RuntimeError("host code must not run")\n@triton.jit\ndef example(X, N: tl.constexpr):\n    return tl.load(X + N)\ndef prepare(weights):\n    raise RuntimeError("not a kernel")\n')
            extracted = inspect_codegen.extract_jit_sources([p])
            self.assertNotIn('torch', extracted)
            self.assertNotIn('RuntimeError', extracted)
            self.assertNotIn('prepare', extracted)
            self.assertIn('def example', extracted)
            ast.parse(extracted)

class ScoreBudgetTests(unittest.TestCase):

    def test_aggregate_weighted_times_before_geometric_mean(self):

        def row(load, weight, time):
            return {'case': {'load': load, 'weight': weight}, 'baseline_ns': 100, 'median_ns': time}
        score, _ = projection_budget.score_rows([row('a', 1, 100), row('a', 9, 50), row('b', 1, 100)])
        self.assertAlmostEqual(score, math.sqrt(1000 / 550))

    def test_actual_result_and_target_budget(self):
        document = json.loads((HERE / 'results/pdl.json').read_text())
        report = projection_budget.build_budget(document, 1.61)
        self.assertAlmostEqual(report['measured_speedup'], document['result']['speedup'])
        self.assertAlmostEqual(report['target_score_us'], 4830.8 / 1.61)
        self.assertTrue(all((r['uniform_reduction_budget_us'] < r['measured_us'] for r in report['load_budgets'].values())))
        self.assertLess(report['optimistic_scenarios']['router_free_impossible_oracle']['speedup'], 1.61)

    def test_reject_inconsistent_result_and_invalid_targets(self):
        document = json.loads((HERE / 'results/pdl.json').read_text())
        for target in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                projection_budget.build_budget(document, target)
        document['result']['speedup'] = 99
        with self.assertRaises(ValueError):
            projection_budget.build_budget(document, 1.61)
if __name__ == '__main__':
    unittest.main()
