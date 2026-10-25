"""Protect the standalone export and compile manifest used for experiments."""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from projection.shapes import SHAPES, SCORED_TOKENS
from tools.build import build_source
from tools.codegen import compile_jobs, extract_jit_sources, KERNEL_SOURCES
from tools.tune import export_submission, source_choices
from tests.host import HostCandidate


class ToolTests(unittest.TestCase):
    def test_compile_manifest_matches_launches(self):
        definitions = {node.name: node for node in ast.parse(extract_jit_sources(KERNEL_SOURCES)).body
                       if isinstance(node, ast.FunctionDef)}
        for preset in ('pdl', 'combined', 'tma'):
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
                if name == '_tma_weight_major':
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
