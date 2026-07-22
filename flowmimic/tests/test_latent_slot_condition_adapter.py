import unittest

import torch

from flowmimic.src.model.flow.checkpoint import (
    flow_state_uses_latent_slot_adapter,
    infer_latent_slot_adapter_config,
    load_flow_state_dict,
)
from flowmimic.src.model.flow.flow_net import FlowNet
from flowmimic.src.model.flow.teacher import EMA


def _make_flow(latent_slot_adapter):
    return FlowNet(
        d_z=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        ffn_dim=32,
        cond_dim=16,
        dropout=0.0,
        latent_len=4,
        latent_slot_adapter=latent_slot_adapter,
        latent_slot_adapter_heads=4,
        latent_slot_adapter_ffn_dim=24,
    )


def _inputs():
    return {
        "x_t": torch.randn(3, 4, 8),
        "t_flow": torch.rand(3),
        "tau_out": torch.linspace(0.0, 1.0, 4),
        "mem": torch.randn(3, 5, 16),
        "g": torch.randn(3, 16),
        "mem_mask": torch.tensor(
            [
                [False, False, False, True, True],
                [False, False, False, False, True],
                [False, False, False, False, False],
            ]
        ),
    }


class LatentSlotConditionAdapterTest(unittest.TestCase):
    def test_zero_initialized_migration_preserves_output(self):
        torch.manual_seed(7)
        source = _make_flow(latent_slot_adapter=False).eval()
        migrated = _make_flow(latent_slot_adapter=True).eval()
        result = load_flow_state_dict(
            migrated,
            source.state_dict(),
            allow_latent_slot_adapter_migration=True,
        )
        self.assertTrue(result.missing_keys)
        self.assertTrue(
            all("slot_condition_adapter." in key for key in result.missing_keys)
        )
        inputs = _inputs()
        torch.testing.assert_close(
            migrated(**inputs),
            source(**inputs),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_vq_slot_initialization_and_output_gradient(self):
        torch.manual_seed(11)
        model = _make_flow(latent_slot_adapter=True)
        adapter = model.slot_condition_adapter
        vq_slots = torch.randn(1, 4, 16)
        adapter.initialize_slot_queries(vq_slots)
        torch.testing.assert_close(adapter.slot_queries, vq_slots)

        output = model(**_inputs())
        output.square().mean().backward()
        gradient = adapter.out_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(torch.count_nonzero(gradient).item(), 0)

    def test_checkpoint_detection_config_and_ema_migration(self):
        source = _make_flow(latent_slot_adapter=False)
        migrated = _make_flow(latent_slot_adapter=True)
        self.assertFalse(flow_state_uses_latent_slot_adapter(source.state_dict()))
        self.assertTrue(flow_state_uses_latent_slot_adapter(migrated.state_dict()))
        self.assertEqual(
            infer_latent_slot_adapter_config(migrated.state_dict()),
            {"latent_len": 4, "ffn_dim": 24},
        )

        ema = EMA(migrated, decay=0.99)
        ema.load_state_dict(source.state_dict(), allow_missing=True)
        self.assertEqual(set(ema.state_dict()), set(migrated.state_dict()))
        ema.update(migrated)


if __name__ == "__main__":
    unittest.main()
