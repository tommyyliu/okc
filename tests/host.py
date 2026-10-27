"""CPU control-flow checks; actual GPU correctness still requires okc check."""

import ast
import copy
from dataclasses import dataclass
import math
import json
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
class SharedLayout:
    swizzle_byte_width: int = 128
    element_bitwidth: int = 16
    rank: int = 2
    transposed: bool = False
    fp4_padded: bool = False
    cga_layout: object = None

    @staticmethod
    def get_default_for(shape, dtype):
        assert shape[1] == 128 and dtype == 'bf16'
        return SharedLayout(cga_layout=[])

    def __repr__(self):
        return (f"NVMMASharedLayout(swizzle_byte_width={self.swizzle_byte_width}, "
                f"element_bitwidth={self.element_bitwidth}, rank={self.rank}, "
                f"transposed={self.transposed}, fp4_padded={self.fp4_padded}, "
                f"cga_layout={self.cga_layout})")


@dataclass
class GluonDescriptor(Descriptor):
    layout: object

    @classmethod
    def from_tensor(cls, tensor, block_shape, layout):
        return cls(tensor, list(tensor.shape), [tensor.shape[1], 1], block_shape, layout)

    def __mangle__(self):
        return f"tensordesc<bf16[{','.join(map(str, self.block_shape))}],{self.layout!r}>"


@dataclass
class Tensor:
    name: str
    shape: tuple
    dtype: str = 'bf16'
    device: str = 'cuda'

    def __getitem__(self, selection):
        if isinstance(selection, tuple):
            return Tensor(self.name + '[:, :n]', (self.shape[0], selection[1].stop), self.dtype)
        return Tensor(self.name + '[:n]', (selection.stop, self.shape[1]), self.dtype)

    def contiguous(self):
        return Tensor(self.name + '.contiguous', self.shape, self.dtype)

    def to(self, dtype):
        return Tensor(self.name + '.to', self.shape, dtype)

    def float(self):
        return self.to('fp32')

    def __sub__(self, other):
        return Tensor(self.name + '-other', self.shape, self.dtype)

    def reshape(self, *shape):
        shape = list(shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)
        if -1 in shape:
            index = shape.index(-1)
            shape[index] = math.prod(self.shape) // math.prod(n for n in shape if n != -1)
        assert math.prod(shape) == math.prod(self.shape)
        return Tensor(self.name + '.reshape', tuple(shape), self.dtype)

    def permute(self, *order):
        return Tensor(self.name + '.permute', tuple(self.shape[i] for i in order), self.dtype)

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

        def stack(tensors, dim=0):
            shape = list(tensors[0].shape)
            shape.insert(dim, len(tensors))
            return Tensor(f'stack(dim={dim})', tuple(shape), tensors[0].dtype)

        ns = {'torch': SimpleNamespace(empty=empty, mm=mm, stack=stack, float32='fp32', bfloat16='bf16'),
              'TensorDescriptor': Descriptor, 'GluonTensorDescriptor': GluonDescriptor,
              'gl': SimpleNamespace(NVMMASharedLayout=SharedLayout, bfloat16='bf16'),
              'json': json,
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
        state['weights_transposed'] = {layer: Tensor('wt', (k, n))}
        state['weights_tiled'] = {(layer, 64, 128): Tensor('wtiles', (n // 64, k // 128, 64, 128))}
        state['router_pair'] = Tensor('router_pair', (512, 8192))
        self.last_data, self.last_state = data, state
        out = self.ns['kernel_fn'](data, state)
        return out, copy.deepcopy(self.calls)
