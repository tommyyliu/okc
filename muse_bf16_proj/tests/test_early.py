"""Early producer signals preserve grids, storage and the consumer launch."""

import unittest

from projection.shapes import SHAPES, SCORED_TOKENS
from tests.host import HostCandidate
from tools.build import build_source


class EarlyPDLTests(unittest.TestCase):
    def test_only_producer_entry_changes_and_reducer_still_uses_pdl(self):
        control = HostCandidate(build_source('round7'))
        candidate = HostCandidate(build_source('round8_early'))
        changed = 0
        for layer in SHAPES:
            for m in (*SCORED_TOKENS, 2, 7, 9, 17, 31, 33, 130, 512, 1000):
                before, old_calls = control.run(m, layer)
                after, new_calls = candidate.run(m, layer)
                self.assertEqual(before, after, (layer, m))
                expected = list(old_calls)
                if len(old_calls) == 2 and old_calls[-1][0] == '_reduce':
                    name, grid, args, options = expected[0]
                    expected[0] = name + '_early', grid, args, options
                    self.assertTrue(new_calls[-1][3]['launch_pdl'])
                    changed += 1
                self.assertEqual(new_calls, expected, (layer, m))
        self.assertGreater(changed, 12)


if __name__ == '__main__':
    unittest.main()
