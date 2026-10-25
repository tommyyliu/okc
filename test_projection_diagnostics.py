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

import bench_lossless
import inspect_codegen
import projection_budget


HERE = Path(__file__).resolve().parent


class AssemblyReportTests(unittest.TestCase):
    def test_diagnostic_logger_prints_only_compiler_metadata(self):
        tree = ast.parse((HERE / 'submission_triton_lossless_diagnostic.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_report_codegen')
        ns = {'json': json, 're': inspect_codegen.re,
              'triton': SimpleNamespace(__version__='test'), 'torch': SimpleNamespace(__version__='test')}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'diagnostic_logger', 'exec'), ns)
        compiled = SimpleNamespace(name='_lossless_gemm', n_regs=128, n_spills=0,
                                   metadata=SimpleNamespace(shared=49152, num_warps=4, num_stages=3),
                                   asm={'ptx': '// ld.global.u8 fake;\n@%p1 ld.global.u8 %r0, [%rd0];\n'})
        output = io.StringIO()
        with redirect_stdout(output):
            ns['_report_codegen'](compiled, 'pooler', 16, 4096, 8192)
        record = inspect_codegen.runtime_reports(output.getvalue())[0]['metadata']
        self.assertEqual(record['static_memory_opcodes'], {'ld.global.u8': 1})
        self.assertEqual(record['registers_per_thread'], 128)
        self.assertEqual(record['n_spills_raw'], 0)

    def test_runtime_log_metadata_and_truncated_records(self):
        record = {'layer': 'pooler', 'm': 256, 'registers_per_thread': 128, 'n_spills_raw': 0}
        line = 'LOSSLESS_CODEGEN ' + json.dumps(record) + '\n'
        text = 'LOSSLESS_CODEGEN {"incomplete":' + '\n' + line + line
        result = inspect_codegen.runtime_reports(json.dumps({'result': {'log_tail': text}}))
        self.assertEqual(result, [{'metadata': record, 'occurrences': 2}])
        self.assertEqual(inspect_codegen.runtime_reports('unrelated profiler output'), [])

    def test_predicated_instructions_and_comments(self):
        ptx = '''
        // ld.global.u8 %r1, [%rd1];
        /* st.local.b32 [%rd0], %r0; */
        @%p1 ld.global.u8 %r2, [%rd2];
        @!%p2 ld.global.v4.b32 {%r3,%r4,%r5,%r6}, [%rd3];
        st.local.b32 [%rd4], %r5;
        ld.local.b32 %r6, [%rd4];
        wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 %r0, %rd0;
        cp.async.ca.shared.global [%r0], [%rd0], 16;
        cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%r0], [%rd0, {%r1,%r2}], [%r3];
        griddepcontrol.wait;
        griddepcontrol.launch_dependents;
        '''
        result = inspect_codegen.summarize_ptx(ptx, '%1 = ttg.convert_layout %0')
        self.assertEqual(result['global_load_instructions'], {'ld.global.u8': 1, 'ld.global.v4.b32': 1})
        self.assertEqual(result['local_memory_instructions'], {'st.local.b32': 1, 'ld.local.b32': 1})
        self.assertEqual(sum(result['mma_instructions'].values()), 1)
        self.assertEqual(sum(result['cp_async_instructions'].values()), 2)
        self.assertEqual(result['grid_dependency_instructions'],
                         {'griddepcontrol.wait': 1, 'griddepcontrol.launch_dependents': 1})
        self.assertEqual(result['layout_conversions_in_ir'], 1)

    def test_assembler_resources_distinguish_unknown_and_zero(self):
        text = ('ptxas info : 32 bytes stack frame, 8 bytes spill stores, 16 bytes spill loads\n'
                'ptxas info : Used 128 registers, 49152 bytes smem\n')
        self.assertEqual(inspect_codegen.parse_ptxas(text),
                         {'registers_per_thread': 128, 'stack_frame_bytes': 32,
                          'spill_store_bytes': 8, 'spill_load_bytes': 16})
        self.assertTrue(all(v is None for v in inspect_codegen.parse_ptxas('').values()))
        self.assertEqual(inspect_codegen.parse_ptxas('0 bytes spill loads')['spill_load_bytes'], 0)

    def test_original_cubin_resources(self):
        result = inspect_codegen.parse_cubin_resources('REG:64 STACK:0 SHARED:49152 LOCAL:16 CONSTANT[0]:384')
        self.assertEqual(result, {'registers_per_thread': 64, 'stack_bytes': 0,
                                  'shared_bytes': 49152, 'local_bytes': 16})


class CompilationManifestTests(unittest.TestCase):
    def test_diagnostic_variant_preserves_gpu_work(self):
        original = ast.parse((HERE / 'submission_triton_lossless.py').read_text())
        diagnostic = ast.parse((HERE / 'submission_triton_lossless_diagnostic.py').read_text())
        functions = {n.name: n for n in diagnostic.body if isinstance(n, ast.FunctionDef)}
        for node in original.body:
            if isinstance(node, ast.FunctionDef) and node.name != 'kernel_fn':
                self.assertEqual(ast.dump(node), ast.dump(functions[node.name]), node.name)
        def remove_logging(node):
            class Normalize(ast.NodeTransformer):
                def visit_Expr(self, expr):
                    if isinstance(expr.value, ast.Call) and isinstance(expr.value.func, ast.Name) and expr.value.func.id == '_report_codegen':
                        return None
                    return self.generic_visit(expr)
                def visit_Assign(self, assignment):
                    if any(isinstance(t, ast.Name) and t.id == 'compiled' for t in assignment.targets):
                        return ast.Expr(value=assignment.value)
                    return self.generic_visit(assignment)
            return Normalize().visit(node)
        original_fn = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
        self.assertEqual(ast.dump(original_fn), ast.dump(remove_logging(functions['kernel_fn'])))

    def test_jit_extraction_does_not_execute_host_code(self):
        with TemporaryDirectory() as d:
            p = Path(d) / 'source.py'
            p.write_text('import torch\nraise RuntimeError("host code must not run")\n'
                         '@triton.jit\ndef example(X, N: tl.constexpr):\n    return tl.load(X + N)\n'
                         'def prepare(weights):\n    raise RuntimeError("not a kernel")\n')
            extracted = inspect_codegen.extract_jit_sources([p])
            self.assertNotIn('torch', extracted)
            self.assertNotIn('RuntimeError', extracted)
            self.assertNotIn('prepare', extracted)
            self.assertIn('def example', extracted)
            ast.parse(extracted)

    def test_all_job_signatures_are_complete(self):
        bundles = {'selective': [HERE / 'submission_triton_selective.py'],
                   'lossless': [HERE / 'submission_triton_lossless.py'],
                   'probes': [HERE / 'submission_triton_lossless.py', HERE / 'lossless_probes.py']}
        for job in inspect_codegen.compile_jobs():
            tree = ast.parse(inspect_codegen.extract_jit_sources(bundles[job['source']]))
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == job['kernel'])
            names = {a.arg for a in fn.args.args}
            self.assertEqual(set(job['signature']) | set(job['constants']), names, job['name'])
            self.assertFalse(set(job['signature']) & set(job['constants']))

    def test_raw_and_lossless_gemm_launch_settings_match(self):
        jobs = inspect_codegen.compile_jobs(['gemm-pooler-16', 'gemm-pooler-256', 'gemm-unpooler-128'])
        for raw, lossless in zip(jobs[::2], jobs[1::2]):
            self.assertEqual(raw['constants'], lossless['constants'])
            self.assertEqual(raw['num_warps'], lossless['num_warps'])
            c = raw['constants']
            expected = bench_lossless.raw_option(c['M'], c['N'], c['K'])
            self.assertEqual(expected, ('gemm', c['BM'], c['BN'], c['BK'], c['S'], raw['num_warps'], 3))


class ScoreBudgetTests(unittest.TestCase):
    def test_aggregate_weighted_times_before_geometric_mean(self):
        def row(load, weight, time):
            return {'case': {'load': load, 'weight': weight}, 'baseline_ns': 100, 'median_ns': time}
        score, _ = projection_budget.score_rows([row('a', 1, 100), row('a', 9, 50), row('b', 1, 100)])
        self.assertAlmostEqual(score, math.sqrt(1000 / 550))

    def test_actual_result_and_target_budget(self):
        document = json.loads((HERE / 'submission-4').read_text())
        report = projection_budget.build_budget(document, 1.61)
        self.assertAlmostEqual(report['measured_speedup'], document['result']['speedup'])
        self.assertAlmostEqual(report['target_score_us'], 4830.8 / 1.61)
        self.assertTrue(all(r['uniform_reduction_budget_us'] < r['measured_us']
                            for r in report['load_budgets'].values()))
        self.assertLess(report['optimistic_scenarios']['router_free_impossible_oracle']['speedup'], 1.61)

    def test_reject_inconsistent_result_and_invalid_targets(self):
        document = json.loads((HERE / 'submission-4').read_text())
        for target in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                projection_budget.build_budget(document, target)
        document['result']['speedup'] = 99
        with self.assertRaises(ValueError):
            projection_budget.build_budget(document, 1.61)


if __name__ == '__main__':
    unittest.main()
