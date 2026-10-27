"""Protect the standalone export and compile manifest used for experiments."""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from projection.shapes import SHAPES, SCORED_TOKENS
from projection.configs import CONFIGS
from tools.build import build_source
from tools.codegen import compile_jobs, extract_jit_sources, KERNEL_SOURCES
from tools.tune import export_submission, source_choices, options, read_source_choices, write_ablations
from tests.host import HostCandidate


class ToolTests(unittest.TestCase):
    def test_small_gemm_search_covers_controls_and_independent_axes(self):
        for layer in ('qkv_gate', 'pooler'):
            for m in (8, 16, 32):
                n, k = SHAPES[layer]
                choices = list(options(m, n, k, layer))
                self.assertEqual(len(choices), len(set(choices)))
                self.assertIn(('torch',), choices)
                self.assertIn(CONFIGS['pdl'][(layer, m)], choices)
                gemms = [c for c in choices if len(c) == 7]
                self.assertEqual({c[0] for c in gemms}, {'gemm', 'weight_major'})
                self.assertEqual({c[3] for c in gemms}, {64, 128, 256})
                self.assertEqual({c[6] for c in gemms}, {2, 3, 4})
                for family, bm, bn, bk, splits, warps, stages in gemms:
                    self.assertEqual(k % (splits * bk), 0)
                    if family == 'weight_major':
                        self.assertIn(('gemm', bm, bn, bk, splits, warps, stages), choices)
        restricted = list(options(16, 4096, 8192, 'pooler', stages=(4,),
                                  k_tiles=(256,), families=('weight_major',), splits=(2, 4)))
        self.assertEqual(len(restricted), 7)  # cuBLAS plus three tiles x two splits
        self.assertTrue(all(c[0] == 'torch' or (c[3], c[6]) == (256, 4) for c in restricted))
        self.assertFalse(any(c[0] == 'weight_major' for c in options(1, 4096, 8192, 'pooler')))
        self.assertFalse(any(c[0] in ('gemm', 'weight_major', 'torch')
                             for c in options(8, 256, 8192, 'router')))

    def test_weight_major_dispatch_and_manifest_include_fp32_pdl_reduction(self):
        candidate = HostCandidate(build_source())
        definitions = {node.name: node for node in ast.parse(extract_jit_sources(KERNEL_SOURCES)).body
                       if isinstance(node, ast.FunctionDef)}
        for m in (7, 8, 16, 17, 32, 33):
            for splits in (1, 4, 8):
                option = ('weight_major', 16, 64, 128, splits, 4, 4)
                candidate.ns['TUNED_CONFIGS'][('pooler', m)] = option
                out, calls = candidate.run(m, 'pooler')
                self.assertEqual(out.shape, (m, 4096))
                self.assertEqual(out.dtype, 'bf16')
                self.assertEqual(calls[0][1], (((m + 15) // 16) * 64, splits))
                self.assertEqual(calls[0][2][3].shape,
                                 (splits, m, 4096) if splits > 1 else (m, 4096))
                self.assertEqual(calls[0][2][3].dtype, 'fp32' if splits > 1 else 'bf16')
                self.assertEqual([c[0] for c in calls], ['_weight_major_stream_gemm'] +
                                 (['_reduce'] if splits > 1 else []))
                jobs = compile_jobs(layers=('pooler',), tokens=(m,),
                                    overrides={('pooler', m): option})
                self.assertEqual(len(jobs), len(calls))
                for call, job in zip(calls, jobs):
                    name, _, args, opts = call
                    self.assertEqual(name, job['kernel'])
                    values = dict(zip((a.arg for a in definitions[name].args.args), args)) | opts
                    self.assertEqual(set(job['signature']) | set(job['constants']),
                                     {a.arg for a in definitions[name].args.args})
                    self.assertEqual({k: values[k] for k in job['constants']}, job['constants'])
                    self.assertEqual(opts.get('launch_pdl', False), job.get('launch_pdl', False))

    def test_remote_ablations_are_scoped_and_change_one_field(self):
        source = build_source()
        control = HostCandidate(source)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_ablations(source, root, ['qkv_gate', 'pooler'], [8, 16, 32])
            self.assertFalse(manifest['measured'])
            self.assertEqual((root / 'control.py').read_text(), source)
            self.assertEqual(len(manifest['experiments']), 8)
            streaming = HostCandidate((root / 'streaming.py').read_text())
            for experiment in manifest['experiments']:
                candidate_source = (root / experiment['file']).read_text()
                candidate = HostCandidate(candidate_source)
                choices = read_source_choices(candidate_source)
                for layer in SHAPES:
                    for m in (*SCORED_TOKENS, 7, 9, 17, 31, 33, 130, 512):
                        if layer not in ('qkv_gate', 'pooler') or m not in (8, 16, 32):
                            self.assertEqual(candidate.run(m, layer), control.run(m, layer))
                        else:
                            self.assertEqual(choices[(layer, m)], candidate.ns['select_config'](
                                dict(layer=layer, m=m, n=SHAPES[layer][0], k=8192)))
                            if experiment['file'] == 'streaming.py':
                                if (layer, m) != ('qkv_gate', 32):
                                    self.assertEqual(candidate.run(m, layer), control.run(m, layer))
                            else:
                                before = streaming.ns['TUNED_CONFIGS'][(layer, m)]
                                after = choices[(layer, m)]
                                self.assertEqual(sum(a != b for a, b in zip(before, after)), 1)
            self.assertEqual(manifest['experiments'][0]['changed_cases'][0]['m'], 32)
            self.assertEqual(len(manifest['experiments'][0]['changed_cases']), 1)

    def test_source_inspection_preserves_experimental_nearby_policy(self):
        source = build_source('tma')
        choices = read_source_choices(source, layers=('pooler',), tokens=(7, 33, 130))
        candidate = HostCandidate(source)
        for m in (7, 33, 130):
            self.assertEqual(choices[('pooler', m)], candidate.ns['select_config'](
                dict(m=m, n=4096, k=8192, layer='pooler')))
            self.assertEqual(choices[('pooler', m)][0], 'tma')

    def test_compile_manifest_matches_launches(self):
        definitions = {node.name: node for node in ast.parse(extract_jit_sources(KERNEL_SOURCES)).body
                       if isinstance(node, ast.FunctionDef)}
        for preset in CONFIGS:
            candidate = HostCandidate(build_source(preset))
            for job in compile_jobs(preset, tokens=(*SCORED_TOKENS, 7, 9, 33, 130, 1000)):
                _, layer, m, *tail = job['name'].split('-')
                _, calls = candidate.run(int(m), layer)
                call = calls[-1] if tail else calls[0]
                name, grid, args, opts = call
                self.assertEqual(name, job['kernel'], job['name'])
                arg_names = [a.arg for a in definitions[name].args.args]
                values = dict(zip(arg_names, args)) | opts
                self.assertEqual(set(job['constants']) | set(job['signature']), set(arg_names))
                self.assertFalse(set(job['constants']) & set(job['signature']))
                self.assertEqual({k: values[k] for k in job['constants']}, job['constants'], job['name'])
                self.assertEqual(opts['num_warps'], job['num_warps'])
                self.assertEqual(opts.get('num_stages', 3), job['num_stages'])
                self.assertEqual(opts.get('launch_pdl', False), job.get('launch_pdl', False))
                self.assertEqual(opts.get('maxnreg'), job.get('maxnreg'))
                self.assertEqual(opts.get('num_ctas', 1), job.get('num_ctas', 1))
                if name == '_hopper_ws_gemm':
                    for descriptor, param in zip(args[:2], ('WD', 'XD')):
                        self.assertEqual(job['signature'][param], descriptor.__mangle__())
                if name in ('_tma_weight_major', '_tma_weight_major_early'):
                    for descriptor, param in zip(args[:2], ('WD', 'XD')):
                        rows, cols = descriptor.block_shape
                        self.assertEqual(job['signature'][param], f'tensordesc<bf16[{rows},{cols}]>')

    def test_partial_tuning_export_preserves_other_cases_and_kernels(self):
        source = build_source('tma')
        before = HostCandidate(source)
        choices = source_choices(SimpleNamespace(**before.ns))
        choices[('pooler', 128)] = ('torch',)
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'nested/tuned.py'
            export_submission(source, output, choices)
            after_source = output.read_text()
        after = HostCandidate(after_source)
        for layer in SHAPES:
            for m in (*SCORED_TOKENS, 7, 17, 33, 130, 512):
                if (layer, m) == ('pooler', 128):
                    self.assertEqual(after.run(m, layer)[1][0][0], 'torch.mm')
                else:
                    self.assertEqual(before.run(m, layer), after.run(m, layer), (layer, m))
        def kernels(text):
            return [ast.dump(n) for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.decorator_list]
        self.assertEqual(kernels(source), kernels(after_source))


if __name__ == '__main__':
    unittest.main()
