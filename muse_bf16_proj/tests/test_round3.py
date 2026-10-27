"""Host checks for weight preparation and the round-three launch contracts.

These checks do not establish GPU numerical correctness or performance.
"""

import copy
import unittest

from projection.shapes import SHAPES
from tests.host import HostCandidate, Tensor
from tools.build import build_source
from tools.codegen import compile_jobs


class Round3Tests(unittest.TestCase):
    def test_transposes_are_prepared_at_original_precision_only_when_selected(self):
        weights = {layer: Tensor(layer, shape, 'fp32' if layer == 'router' else 'bf16')
                   for layer, shape in SHAPES.items()}
        default = HostCandidate(build_source())
        self.assertEqual(set(default.ns['prepare'](weights)),
                         {'weights', 'router_hi', 'router_lo'})
        candidate = HostCandidate(build_source('round3'))
        state = candidate.ns['prepare'](weights)
        self.assertEqual(set(state['weights_transposed']), {'qkv_gate', 'pooler', 'unpooler'})
        for layer, packed in state['weights_transposed'].items():
            self.assertEqual(packed.shape, SHAPES[layer][::-1])
            self.assertEqual(packed.dtype, 'bf16')
            self.assertEqual(packed.name, layer + '.contiguous.T.contiguous')
        self.assertEqual(state['weights']['router'].dtype, 'fp32')

    def test_transposed_mm_reads_current_cloned_state_for_nearby_shapes(self):
        candidate = HostCandidate(build_source('round3'))
        for layer, m in [('qkv_gate', 128), ('qkv_gate', 512), ('pooler', 129),
                         ('unpooler', 16), ('unpooler', 130), ('unpooler', 1000)]:
            out, calls = candidate.run(m, layer)
            self.assertEqual([c[0] for c in calls], ['torch.mm'])
            self.assertEqual(calls[0][2][1].shape, SHAPES[layer][::-1])
            self.assertEqual(out.shape, (m, SHAPES[layer][0]))
            state = copy.deepcopy(candidate.last_state)
            state['weights_transposed'][layer].name = 'cloned_transpose'
            candidate.calls.clear()
            candidate.ns['kernel_fn'](candidate.last_data, state)
            self.assertEqual(candidate.calls[0][2][1].name, 'cloned_transpose[:, :n]')

    def test_router_keeps_grid_size_and_halves_scratch_with_hi_lo_operands(self):
        before = HostCandidate(build_source())
        after = HostCandidate(build_source('round3'))
        for m in (8, 9, 16, 32):
            _, old_calls = before.run(m, 'router')
            out, calls = after.run(m, 'router')
            producer, reducer = calls
            self.assertEqual([c[0] for c in calls], ['_gemm', '_reduce'])
            self.assertEqual(producer[1][0] * producer[1][1] * producer[1][2],
                             old_calls[0][1][0] * old_calls[0][1][1] * old_calls[0][1][2])
            self.assertEqual(producer[2][4].shape, (16, m, 256))
            self.assertEqual(old_calls[0][2][4].shape, (32, m, 256))
            self.assertEqual([a.name for a in producer[2][1:3]], ['hi', 'lo'])
            self.assertTrue(producer[2][9])  # ROUTER: both BF16 components are accumulated.
            self.assertEqual(out.dtype, 'fp32')
            self.assertTrue(reducer[3]['launch_pdl'])

    def test_chunked_gemv_uses_original_weight_precision_and_no_partials(self):
        candidate = HostCandidate(build_source('round3'))
        jobs = {job['name'].split('-')[1]: job for job in compile_jobs('round3', tokens=(1,))}
        for layer in ('qkv_gate', 'pooler', 'unpooler', 'router'):
            out, calls = candidate.run(1, layer)
            self.assertEqual([c[0] for c in calls], ['_gemv_chunked'])
            call = calls[0]
            self.assertEqual(call[2][1].name, 'w')
            expected_dtype = 'fp32' if layer == 'router' else 'bf16'
            self.assertEqual(call[2][1].dtype, expected_dtype)
            self.assertEqual(out.dtype, expected_dtype)
            self.assertEqual(jobs[layer]['signature']['W'], '*' + expected_dtype)
            self.assertEqual(call[3]['BLOCK_K'], 1024)
            self.assertEqual(call[3]['ROWS'], 1)
        self.assertEqual(candidate.run(1, 'o')[1][0][0], '_gemv_full')


if __name__ == '__main__':
    unittest.main()
