"""CPU layout/launch checks; numerical GPU validation remains an OKC check."""

import ast
import copy
import itertools
import math
from pathlib import Path
import struct
import unittest

from projection.configs import KEEPER_CONFIGS, ROUND4_CONFIGS
from projection.shapes import SHAPES, SCORED_TOKENS
from tests.host import HostCandidate, Tensor
from tools.build import build_source


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


class Array:
    """Small numerical stand-in for the Torch preparation operations only."""

    def __init__(self, values, shape, dtype='bf16'):
        self.values = tuple(values)
        self.shape = tuple(shape)
        self.dtype = dtype
        assert len(self.values) == math.prod(self.shape)

    def at(self, indices):
        index = 0
        for i, width in zip(indices, self.shape):
            index = index * width + i
        return self.values[index]

    def contiguous(self):
        return self

    def reshape(self, *shape):
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = len(self.values) // math.prod(n for n in shape if n != -1)
        return Array(self.values, shape, self.dtype)

    def permute(self, *order):
        shape = tuple(self.shape[i] for i in order)
        values = []
        for indices in itertools.product(*(range(n) for n in shape)):
            original = [0] * len(order)
            for axis, old_axis in enumerate(order):
                original[old_axis] = indices[axis]
            values.append(self.at(original))
        return Array(values, shape, self.dtype)

    @property
    def T(self):
        return self.permute(*reversed(range(len(self.shape))))

    def to(self, dtype):
        return Array(map(bf16 if dtype == 'bf16' else fp32, self.values), self.shape, dtype)

    def float(self):
        return self.to('fp32')

    def __sub__(self, other):
        return Array((fp32(x - y) for x, y in zip(self.values, other.values)), self.shape, self.dtype)


def stack_arrays(tensors, dim=0):
    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    values = []
    for indices in itertools.product(*(range(n) for n in shape)):
        indices = list(indices)
        tensor = tensors[indices.pop(dim)]
        values.append(tensor.at(indices))
    return Array(values, shape, tensors[0].dtype)


class Index:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _):
        return self.value


class Round4Tests(unittest.TestCase):
    def test_preparation_covers_fallback_after_scored_overrides(self):
        candidate = HostCandidate(build_source('round4'))
        candidate.ns['TUNED_CONFIGS'].update(KEEPER_CONFIGS)
        weights = {layer: Tensor(layer, shape, 'fp32' if layer == 'router' else 'bf16')
                   for layer, shape in SHAPES.items()}
        state = candidate.ns['prepare'](weights)
        self.assertEqual(set(state['weights_tiled']),
                         {('qkv_gate', 64, 128), ('pooler', 64, 128)})
        self.assertIn('router_pair', state)

    def test_prepared_tiles_match_actual_kernel_offsets_across_splits(self):
        candidate = HostCandidate(build_source('round4'))
        candidate.ns['torch'].stack = stack_arrays
        # Distinct exact BF16 bit patterns expose wrong tile/row/K permutations.
        values = [struct.unpack('<f', struct.pack('<I',
                  (0x1000 + (i % 16384) + (0x8000 if i >= 16384 else 0)) << 16))[0]
                  for i in range(128 * 256)]
        weights = {'qkv_gate': Array(values, (128, 256)),
                   'pooler': Array(reversed(values), (128, 256)),
                   'o': Array(range(32), (4, 8)),
                   'unpooler': Array(range(32), (4, 8)),
                   'router': Array((fp32(1 + i / 1000) for i in range(24)), (3, 8), 'fp32')}
        state = candidate.ns['prepare'](weights)
        tree = ast.parse(Path('projection/kernels/layout.py').read_text())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_weight_major_tiled_gemm')
        offsets = next(n.value for n in kernel.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'woffsets' for t in n.targets))
        expression = compile(ast.Expression(offsets), '<GPU weight address>', 'eval')
        for layer in ('qkv_gate', 'pooler'):
            packed = state['weights_tiled'][layer, 64, 128]
            self.assertEqual(packed.shape, (2, 2, 64, 128))
            self.assertEqual(packed.dtype, 'bf16')
            for splits in (1, 2):
                for n_tile, split, ni, ki in itertools.product(range(2), range(splits),
                                                              (0, 31, 63), (0, 63, 127)):
                    for step in range(256 // (splits * 128)):
                        offset = eval(expression, {}, dict(n_tile=n_tile, BN=64, K=256, S=splits,
                                      split=split, ni=Index(ni), BK=128, ki=Index(ki))) + step * 64 * 128
                        k = split * (256 // splits) + step * 128 + ki
                        self.assertEqual(packed.values[offset], weights[layer].at((n_tile * 64 + ni, k)))

        pair = state['router_pair']
        self.assertEqual(pair.shape, (6, 8))
        self.assertEqual(pair.dtype, 'bf16')
        for expert, k in itertools.product(range(3), range(8)):
            self.assertEqual(pair.at((2 * expert, k)), state['router_hi'].at((expert, k)))
            self.assertEqual(pair.at((2 * expert + 1, k)), state['router_lo'].at((expert, k)))
        # Independent high/low dot products reconstruct original FP32 routing weights.
        x = [1.0, -2.0, 0.5, 8.0, -0.25, 4.0, 2.0, -1.0]
        for expert in range(3):
            actual = sum(v * pair.at((2 * expert, k)) for k, v in enumerate(x))
            actual += sum(v * pair.at((2 * expert + 1, k)) for k, v in enumerate(x))
            reference = sum(v * weights['router'].at((expert, k)) for k, v in enumerate(x))
            self.assertLess(abs(actual - reference) / abs(reference), 1e-4)

    def test_new_families_keep_launch_parameters_and_unchanged_case_calls(self):
        before, after = HostCandidate(build_source('keepers')), HostCandidate(build_source('round4'))
        changed = 0
        for layer in SHAPES:
            for m in SCORED_TOKENS:
                old, new = KEEPER_CONFIGS[layer, m], ROUND4_CONFIGS[layer, m]
                if old == new:
                    self.assertEqual(before.run(m, layer), after.run(m, layer))
                else:
                    changed += 1
                    self.assertEqual(old[1:], new[1:], (layer, m))
        self.assertEqual(changed, 20)

    def test_native_outputs_return_transpose_views_with_correct_shapes(self):
        candidate = HostCandidate(build_source('round4'))
        for m in (2, 8, 17, 32):
            out, calls = candidate.run(m, 'o')
            self.assertEqual([c[0] for c in calls], ['_weight_major_native_gemm'])
            self.assertEqual(calls[0][2][2].shape, (8192, m))
            self.assertEqual(out.shape, (m, 8192))
            self.assertEqual(out.name, 'allocated.T')
        for layer, m in [('qkv_gate', 128), ('qkv_gate', 512), ('unpooler', 16),
                         ('unpooler', 130), ('unpooler', 1000)]:
            out, calls = candidate.run(m, layer)
            self.assertEqual([c[0] for c in calls], ['torch.mm'])
            left, right = calls[0][2]
            self.assertEqual(left.shape, SHAPES[layer])
            self.assertEqual(right.shape, (SHAPES[layer][1], m))
            self.assertEqual(out.shape, (m, SHAPES[layer][0]))
            self.assertEqual(out.name, 'mm.T')

    def test_current_prepared_state_is_used_on_nearby_shapes(self):
        candidate = HostCandidate(build_source('round4'))
        for layer, m in [('pooler', 7), ('qkv_gate', 17), ('router', 9), ('router', 129)]:
            candidate.run(m, layer)
            state = copy.deepcopy(candidate.last_state)
            packed = (state['router_pair'] if layer == 'router'
                      else state['weights_tiled'][layer, 64, 128])
            packed.name = 'current_clone'
            candidate.calls.clear()
            candidate.ns['kernel_fn'](candidate.last_data, state)
            self.assertIs(candidate.calls[0][2][1], packed)
            self.assertEqual(candidate.calls[-1][0], '_reduce')
            self.assertTrue(candidate.calls[-1][3]['launch_pdl'])

    def test_router_pair_does_not_change_logical_grid_or_partial_precision(self):
        before, after = HostCandidate(build_source('keepers')), HostCandidate(build_source('round4'))
        for m in (8, 9, 16, 32, 128, 256):
            _, old = before.run(m, 'router')
            out, new = after.run(m, 'router')
            self.assertEqual(new[0][0], '_router_interleaved_gemm')
            self.assertEqual(new[0][1], old[0][1])
            self.assertEqual(new[0][2][3].shape, old[0][2][4].shape)
            self.assertEqual(new[0][2][3].dtype, 'fp32')
            self.assertEqual(out.dtype, 'fp32')
            self.assertEqual(new[1], old[1])


if __name__ == '__main__':
    unittest.main()
