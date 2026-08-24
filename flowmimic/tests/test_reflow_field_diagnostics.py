import unittest

import torch

from flowmimic.tools.diagnose_reflow_field import (
    _parse_model_spec,
    _parse_times,
    heun_advance,
)


class ReflowFieldDiagnosticTests(unittest.TestCase):
    def test_parse_times_requires_ordered_endpoints(self):
        self.assertEqual(_parse_times("0,0.5,1"), [0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            _parse_times("0.1,1")
        with self.assertRaises(ValueError):
            _parse_times("0,0.5,0.4,1")

    def test_parse_labeled_checkpoint(self):
        spec = _parse_model_spec("candidate=/tmp/model.pt")
        self.assertEqual(spec.label, "candidate")
        self.assertEqual(spec.path, "/tmp/model.pt")

    def test_heun_advance_is_exact_for_constant_velocity(self):
        x = torch.zeros(2, 3, 4)

        def velocity(state, time):
            del time
            return torch.ones_like(state) * 2.0

        result = heun_advance(velocity, x, 0.2, 0.7)
        torch.testing.assert_close(result, torch.ones_like(x))


if __name__ == "__main__":
    unittest.main()
