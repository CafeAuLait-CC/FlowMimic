import unittest

import torch

from flowmimic.src.model.flow.checkpoint import (
    flow_state_uses_relative_time_bias,
    infer_relative_time_hidden_dim,
    load_flow_state_dict,
)
from flowmimic.src.model.flow.flow_net import FlowNet
from flowmimic.src.model.flow.teacher import EMA


def _make_flow(relative_time_bias):
    return FlowNet(
        d_z=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        ffn_dim=32,
        cond_dim=16,
        dropout=0.0,
        relative_time_bias=relative_time_bias,
        relative_time_hidden_dim=12,
    )


class RelativeTimeAttentionTest(unittest.TestCase):
    def test_zero_initialized_migration_preserves_output(self):
        torch.manual_seed(7)
        source = _make_flow(relative_time_bias=False).eval()
        migrated = _make_flow(relative_time_bias=True).eval()
        result = load_flow_state_dict(
            migrated,
            source.state_dict(),
            allow_relative_time_migration=True,
        )
        self.assertTrue(result.missing_keys)
        self.assertTrue(
            all(".relative_time_bias." in key for key in result.missing_keys)
        )

        x_t = torch.randn(3, 5, 8)
        t_flow = torch.rand(3)
        tau_out = torch.linspace(0.0, 1.0, 5)
        tau_cond = torch.tensor(
            [[0.0, 0.2, 1.0], [0.0, 0.6, 1.0], [0.0, 0.8, 1.0]]
        )
        mem = torch.randn(3, 3, 16)
        g = torch.randn(3, 16)

        expected = source(x_t, t_flow, tau_out, mem, g)
        actual = migrated(
            x_t,
            t_flow,
            tau_out,
            mem,
            g,
            tau_cond=tau_cond,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_bias_parameters_receive_gradients(self):
        torch.manual_seed(11)
        model = _make_flow(relative_time_bias=True)
        output = model(
            torch.randn(2, 4, 8),
            torch.rand(2),
            torch.linspace(0.0, 1.0, 4),
            torch.randn(2, 3, 16),
            torch.randn(2, 16),
            mem_mask=torch.tensor([[False, False, True], [False, False, False]]),
            tau_cond=torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.25, 1.0]]),
        )
        output.square().mean().backward()
        for block in model.blocks:
            grad = block.relative_time_bias.net[-1].weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(torch.count_nonzero(grad).item(), 0)

    def test_checkpoint_detection_and_ema_migration(self):
        source = _make_flow(relative_time_bias=False)
        migrated = _make_flow(relative_time_bias=True)
        self.assertFalse(flow_state_uses_relative_time_bias(source.state_dict()))
        self.assertTrue(flow_state_uses_relative_time_bias(migrated.state_dict()))
        self.assertEqual(infer_relative_time_hidden_dim(migrated.state_dict()), 12)

        ema = EMA(migrated, decay=0.99)
        ema.load_state_dict(source.state_dict(), allow_missing=True)
        self.assertEqual(set(ema.state_dict()), set(migrated.state_dict()))
        ema.update(migrated)

    def test_relative_model_requires_condition_timestamps(self):
        model = _make_flow(relative_time_bias=True)
        with self.assertRaisesRegex(ValueError, "tau_out and tau_cond"):
            model(
                torch.randn(1, 4, 8),
                torch.rand(1),
                torch.linspace(0.0, 1.0, 4),
                torch.randn(1, 3, 16),
                torch.randn(1, 16),
            )


if __name__ == "__main__":
    unittest.main()
