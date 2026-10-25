"""CPU control-flow checks; actual GPU correctness still requires okc check."""

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest

import inspect_codegen


HERE = Path(__file__).resolve().parent


@dataclass
class Descriptor:
    base: object
    shape: list
    strides: list
    block_shape: list


@dataclass
class Tensor:
    name: str
    shape: tuple
    dtype: str = 'bf16'
    device: str = 'cuda'

    def __getitem__(self, selection):
        return Tensor(self.name + '[:n]', (selection.stop, self.shape[1]), self.dtype)

    @property
    def T(self):
        return Tensor(self.name + '.T', self.shape[::-1], self.dtype)


class HostCandidate:
    """Execute candidate host dispatch with recorded GPU calls, without imports."""

    def __init__(self, filename):
        tree = ast.parse((HERE / filename).read_text())
        self.calls = []
        owner = self

        class Kernel:
            def __init__(self, name):
                self.name = name

            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    owner.calls.append((self.name, grid, args, kwargs))
                return launch

        def empty(shape, device, dtype):
            return Tensor('allocated', tuple(shape), dtype, device)

        def mm(x, w):
            self.calls.append(('torch.mm', None, (x, w), {}))
            return Tensor('mm', (x.shape[0], w.shape[1]), x.dtype)

        ns = {'torch': SimpleNamespace(empty=empty, mm=mm, float32='fp32', bfloat16='bf16'),
              'TensorDescriptor': Descriptor,
              'triton': SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b,
                                       next_power_of_2=lambda a: 1 << (a-1).bit_length())}
        body = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.decorator_list:
                    ns[node.name] = Kernel(node.name)
                else:
                    body.append(node)
            elif isinstance(node, ast.Assign):
                body.append(node)
        exec(compile(ast.Module(body=body, type_ignores=[]), filename, 'exec'), ns)
        self.ns = ns

    def run(self, m, layer):
        self.calls.clear()
        n, k = {'qkv_gate': (2304, 8192), 'o': (8192, 1024),
                'pooler': (4096, 8192), 'unpooler': (1024 if m <= 8 else 8192, 4096),
                'router': (256, 8192)}[layer]
        data = {'x': Tensor('x', (m, k)),
                'config': dict(m=m, n=n, k=k, layer=layer)}
        state = {'weights': {layer: Tensor('w', (n, k), 'fp32' if layer == 'router' else 'bf16')},
                 'router_hi': Tensor('hi', (256, 8192)),
                 'router_lo': Tensor('lo', (256, 8192))}
        self.last_data, self.last_state = data, state
        out = self.ns['kernel_fn'](data, state)
        return out, copy.deepcopy(self.calls)


class ExperimentTests(unittest.TestCase):
    def test_tma_dispatch_and_current_call_descriptor_addresses(self):
        control = HostCandidate('submission_triton_combined.py')
        tma = HostCandidate('submission_triton_tma.py')
        for m in (1, 2, 7, 8, 9, 16, 17, 32, 33, 64, 128, 130, 256, 257, 512, 1000):
            for layer in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router'):
                expected, before = control.run(m, layer)
                actual, after = tma.run(m, layer)
                changed = 1 < m <= 256 and (layer in ('qkv_gate', 'pooler') or
                                           (layer == 'unpooler' and m > 8))
                if not changed:
                    self.assertEqual((actual, after), (expected, before), (m, layer))
                    continue
                self.assertEqual(after[0][0], '_tma_weight_major')
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, 'bf16')
                data, state = tma.last_data, tma.last_state
                cfg = data['config']
                # A new independent prepared-state copy must bind new addresses.
                clone = copy.deepcopy(state)
                tma.calls.clear()
                tma.ns['kernel_fn'](data, clone)
                _, _, args, options = tma.calls[0]
                wd, xd = args[:2]
                self.assertIs(wd.base, clone['weights'][layer])
                self.assertIs(xd.base, data['x'])
                self.assertEqual(wd.shape, [cfg['n'], cfg['k']])
                self.assertEqual(xd.shape, [m, cfg['k']])
                self.assertEqual(wd.strides, [cfg['k'], 1])
                self.assertEqual(xd.strides, [cfg['k'], 1])
                bm, bn, bk, s = tma.ns['tma_config'](cfg)
                self.assertEqual(wd.block_shape, [bn, bk])
                self.assertEqual(xd.block_shape, [bm, bk])
                self.assertEqual(cfg['k'] % (s * bk), 0)
                # K partitions cover every element once, without an atomic workspace.
                ranges = [(i * (cfg['k']//s), (i+1) * (cfg['k']//s)) for i in range(s)]
                self.assertEqual(ranges[0][0], 0)
                self.assertEqual(ranges[-1][1], cfg['k'])
                self.assertTrue(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
                if s > 1:
                    self.assertEqual(tma.calls[-1][0], '_reduce')
                    self.assertIs(tma.calls[-1][3]['launch_pdl'], True)
                else:
                    self.assertEqual(len(tma.calls), 1)

    def test_combination_uses_checked_kernels_and_dispatch(self):
        def kernels(filename):
            return {n.name: ast.dump(n) for n in ast.parse((HERE / filename).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.decorator_list}

        pdl_kernels = kernels('submission_triton_pdl.py')
        orientation_kernels = kernels('submission_triton_orientation.py')
        combined_kernels = kernels('submission_triton_combined.py')
        self.assertEqual(combined_kernels,
                         dict(pdl_kernels, _weight_major_gemm=orientation_kernels['_weight_major_gemm']))
        candidates = {name: HostCandidate(f'submission_triton_{name}.py')
                      for name in ('pdl', 'orientation', 'combined')}
        for m in (1, 2, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128,
                  130, 256, 257, 512, 1000):
            for layer in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router'):
                selected = 'orientation' if layer == 'o' and 1 < m <= 32 else 'pdl'
                self.assertEqual(candidates['combined'].run(m, layer),
                                 candidates[selected].run(m, layer), (m, layer))

    def test_pdl_preserves_launches_and_dispatch_across_all_paths(self):
        control = HostCandidate('submission_triton_selective.py')
        pdl = HostCandidate('submission_triton_pdl.py')
        for m in (1, 2, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128,
                  130, 256, 257, 512, 1000):
            for layer in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router'):
                expected, before = control.run(m, layer)
                actual, after = pdl.run(m, layer)
                for name, _, _, options in after:
                    if name == '_reduce':
                        self.assertIs(options.pop('launch_pdl'), True)
                    else:
                        self.assertNotIn('launch_pdl', options)
                self.assertEqual(before, after, (m, layer))
                self.assertEqual(expected, actual, (m, layer))

    def test_pdl_keeps_original_gpu_math(self):
        class RemoveGDC(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    if node.value.func.attr in ('gdc_wait', 'gdc_launch_dependents'):
                        return None
                return self.generic_visit(node)

            def visit_If(self, node):
                node = self.generic_visit(node)
                return node if node.body else None

        def kernels(filename):
            return {n.name: n for n in ast.parse((HERE / filename).read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.decorator_list}
        original = kernels('submission_triton_selective.py')
        changed = kernels('submission_triton_pdl.py')
        for name, node in original.items():
            self.assertEqual(ast.dump(node), ast.dump(RemoveGDC().visit(changed[name])), name)

    def test_orientation_changes_only_small_token_o(self):
        control = HostCandidate('submission_triton_selective.py')
        layout = HostCandidate('submission_triton_orientation.py')
        for m in (1, 2, 8, 16, 17, 32, 33, 128, 256, 1000):
            for layer in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router'):
                expected, before = control.run(m, layer)
                actual, after = layout.run(m, layer)
                if layer == 'o' and 1 < m <= 32:
                    self.assertEqual(len(after), 1)
                    self.assertEqual(after[0][0], '_weight_major_gemm')
                    self.assertEqual(actual.shape, (m, 8192))
                    self.assertEqual(actual.dtype, 'bf16')
                else:
                    self.assertEqual(before, after, (m, layer))
                    self.assertEqual(expected, actual, (m, layer))

    def test_compile_manifest_matches_actual_host_settings(self):
        for job in inspect_codegen.compile_jobs(('orientation', 'pdl', 'tma')):
            candidate = HostCandidate('submission_triton_' + job['source'] + '.py')
            if job['kernel'] == '_reduce':
                continue
            m = job['constants']['M']
            layer = job['name'].split('-')[1]
            if layer == 'qkv':
                layer = 'qkv_gate'
            _, calls = candidate.run(m, layer)
            name, _, args, opts = calls[0]
            self.assertEqual(name, job['kernel'], job['name'])
            tree = ast.parse((HERE / ('submission_triton_' + job['source'] + '.py')).read_text())
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            constants = dict(zip((a.arg for a in fn.args.args), args))
            constants.update(opts)
            self.assertEqual({k: constants[k] for k in job['constants']}, job['constants'], job['name'])
            self.assertEqual(opts['num_warps'], job['num_warps'])
            self.assertEqual(opts['num_stages'], job['num_stages'])


if __name__ == '__main__':
    unittest.main()
