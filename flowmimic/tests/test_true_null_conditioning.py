import unittest

import torch

from flowmimic.src.model.flow.checkpoint import (
    flow_state_uses_true_null_condition,
    load_flow_state_dict,
)
from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow


def _build_flow(true_null_condition):
    return ConditionalRectFlow(
        d_z=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ffn_dim=32,
        dropout=0.0,
        num_styles=4,
        style_dim=4,
        cond_dim=16,
        num_joints_2d=25,
        cond_layers=1,
        cond_heads=2,
        p_style_drop=0.0,
        latent_len=2,
        true_null_condition=true_null_condition,
    )


class TrueNullConditioningTest(unittest.TestCase):
    def test_null_rows_have_one_learned_token_and_preserve_other_rows(self):
        flow = _build_flow(True)
        batch_size, token_count, width = 3, 5, 16
        global_context = torch.randn(batch_size, width)
        memory = torch.randn(batch_size, token_count, width)
        tau_cond = torch.rand(batch_size, token_count)
        mask_cond = torch.ones(batch_size, token_count, dtype=torch.bool)
        style_id = torch.tensor([1, 2, 3])
        domain_id = torch.ones(batch_size, dtype=torch.long)
        null_mask = torch.tensor([True, False, True])

        g_out, mem_out, tau_out, mask_out = flow.apply_true_null_condition(
            global_context,
            memory,
            tau_cond,
            mask_cond,
            style_id,
            domain_id,
            null_mask,
        )

        self.assertTrue(torch.equal(mem_out[1], memory[1]))
        self.assertTrue(torch.equal(tau_out[1], tau_cond[1]))
        self.assertTrue(torch.equal(mask_out[1], mask_cond[1]))
        self.assertTrue(torch.equal(mask_out[0], torch.tensor([1, 0, 0, 0, 0]).bool()))
        self.assertTrue(torch.equal(mask_out[2], torch.tensor([1, 0, 0, 0, 0]).bool()))
        self.assertTrue(torch.equal(tau_out[null_mask], torch.zeros(2, token_count)))
        self.assertTrue(
            torch.equal(
                mem_out[null_mask, 0],
                flow.null_memory[0, 0].expand(2, -1),
            )
        )
        self.assertTrue(torch.equal(mem_out[null_mask, 1:], torch.zeros(2, 4, width)))

        expected_g, _, _, _ = flow.encode_null_condition(
            style_id[null_mask],
            domain_id[null_mask],
        )
        self.assertTrue(torch.allclose(g_out[null_mask], expected_g))

    def test_legacy_checkpoint_migrates_only_when_requested(self):
        legacy = _build_flow(False)
        true_null = _build_flow(True)
        state = legacy.state_dict()

        self.assertFalse(flow_state_uses_true_null_condition(state))
        with self.assertRaises(RuntimeError):
            load_flow_state_dict(true_null, state)
        load_flow_state_dict(
            true_null,
            state,
            allow_true_null_migration=True,
        )
        self.assertTrue(flow_state_uses_true_null_condition(true_null.state_dict()))

    def test_mixed_forward_uses_null_condition_without_copying_real_memory(self):
        flow = _build_flow(True)
        batch_size, token_count, width = 4, 5, 16
        x_t = torch.randn(batch_size, 2, 8)
        t_flow = torch.rand(batch_size)
        tau_out = torch.linspace(0.0, 1.0, 2)
        memory = torch.randn(
            batch_size,
            token_count,
            width,
            requires_grad=True,
        )
        global_context = torch.randn(
            batch_size,
            width,
            requires_grad=True,
        )
        tau_cond = torch.rand(batch_size, token_count)
        style_id = torch.tensor([1, 2, 3, 1])
        domain_id = torch.ones(batch_size, dtype=torch.long)
        null_mask = torch.tensor([False, True, False, True])

        pred = flow(
            x_t,
            t_flow,
            tau_out,
            mem=memory,
            g=global_context,
            mem_mask=torch.zeros(batch_size, token_count, dtype=torch.bool),
            tau_cond=tau_cond,
            null_mask=null_mask,
            style_id=style_id,
            domain_id=domain_id,
        )
        changed_memory = memory.detach().clone()
        changed_memory[null_mask] = torch.randn_like(changed_memory[null_mask]) * 100
        changed_global = global_context.detach().clone()
        changed_global[null_mask] = (
            torch.randn_like(changed_global[null_mask]) * 100
        )
        pred_changed = flow(
            x_t,
            t_flow,
            tau_out,
            mem=changed_memory,
            g=changed_global,
            mem_mask=torch.zeros(batch_size, token_count, dtype=torch.bool),
            tau_cond=tau_cond,
            null_mask=null_mask,
            style_id=style_id,
            domain_id=domain_id,
        )

        self.assertTrue(torch.allclose(pred[null_mask], pred_changed[null_mask]))
        pred.square().mean().backward()
        self.assertIsNotNone(flow.null_memory.grad)
        self.assertIsNotNone(flow.null_global.grad)
        self.assertTrue(torch.isfinite(flow.null_memory.grad).all())
        self.assertTrue(torch.isfinite(flow.null_global.grad).all())

    def test_mixed_forward_reports_conditional_null_velocity_distance(self):
        flow = _build_flow(True)
        batch_size, token_count, width = 4, 5, 16
        x_t = torch.randn(batch_size, 2, 8)
        t_flow = torch.rand(batch_size)
        tau_out = torch.linspace(0.0, 1.0, 2)
        memory = torch.randn(batch_size, token_count, width)
        global_context = torch.randn(batch_size, width)
        tau_cond = torch.rand(batch_size, token_count)
        style_id = torch.tensor([1, 2, 3, 1])
        domain_id = torch.ones(batch_size, dtype=torch.long)
        null_mask = torch.tensor([False, True, False, True])

        pred, diagnostics = flow(
            x_t,
            t_flow,
            tau_out,
            mem=memory,
            g=global_context,
            mem_mask=torch.zeros(batch_size, token_count, dtype=torch.bool),
            tau_cond=tau_cond,
            null_mask=null_mask,
            style_id=style_id,
            domain_id=domain_id,
            return_cfg_diagnostics=True,
        )

        self.assertEqual(pred.shape, x_t.shape)
        self.assertEqual(diagnostics["guidance_delta_l2"].shape, (2,))
        self.assertTrue(torch.isfinite(diagnostics["guidance_delta_l2"]).all())
        self.assertTrue((diagnostics["guidance_delta_l2"] > 0).all())


if __name__ == "__main__":
    unittest.main()
