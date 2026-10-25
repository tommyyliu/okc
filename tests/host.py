"""CPU control-flow checks; actual GPU correctness still requires okc check."""

import ast
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest



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
        tree = ast.parse(filename)
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
