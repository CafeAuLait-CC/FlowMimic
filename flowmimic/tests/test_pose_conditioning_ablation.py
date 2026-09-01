import unittest

import torch

from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow


def _build_flow(mode):
    torch.manual_seed(7)
    flow = ConditionalRectFlow(
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
        true_null_condition=True,
        pose_conditioning=mode,
    )
    flow.eval()
    return flow


def _inputs():
    torch.manual_seed(11)
    batch_size, cond_frames = 2, 7
    return {
        "x_t": torch.randn(batch_size, 2, 8),
        "t_flow": torch.rand(batch_size),
        "tau_out": torch.linspace(0.0, 1.0, 2),
        "k2d": torch.randn(batch_size, cond_frames, 25, 2),
        "tau_cond": torch.linspace(0.0, 1.0, cond_frames).expand(batch_size, -1),
        "vis_mask": torch.ones(batch_size, cond_frames, 25),
        "style_id": torch.tensor([1, 2]),
        "domain_id": torch.ones(batch_size, dtype=torch.long),
    }


class PoseConditioningAblationTest(unittest.TestCase):
    def _predict(self, mode, k2d):
        flow = _build_flow(mode)
        inputs = _inputs()
        return flow(
            inputs["x_t"],
            inputs["t_flow"],
            inputs["tau_out"],
            k2d=k2d,
            tau_cond=inputs["tau_cond"],
            style_id=inputs["style_id"],
            domain_id=inputs["domain_id"],
            apply_style_dropout=False,
            vis_mask=inputs["vis_mask"],
        )

    def test_pose_changes_only_reach_enabled_paths(self):
        inputs = _inputs()
        pose_a = inputs["k2d"]
        pose_b = pose_a + torch.randn_like(pose_a) * 2.0

        for mode in ("full", "memory_only", "global_only"):
            with self.subTest(mode=mode):
                pred_a = self._predict(mode, pose_a)
                pred_b = self._predict(mode, pose_b)
                self.assertFalse(torch.allclose(pred_a, pred_b))

        pred_a = self._predict("style_only", pose_a)
        pred_b = self._predict("style_only", pose_b)
        self.assertTrue(torch.equal(pred_a, pred_b))

    def test_memory_only_global_context_is_pose_invariant(self):
        flow = _build_flow("memory_only")
        inputs = _inputs()
        pose_b = inputs["k2d"] + torch.randn_like(inputs["k2d"])
        g_a, memory_a, _ = flow.encode_cond(
            inputs["k2d"],
            inputs["tau_cond"],
            inputs["style_id"],
            inputs["domain_id"],
            apply_style_dropout=False,
            vis_mask=inputs["vis_mask"],
        )
        g_b, memory_b, _ = flow.encode_cond(
            pose_b,
            inputs["tau_cond"],
            inputs["style_id"],
            inputs["domain_id"],
            apply_style_dropout=False,
            vis_mask=inputs["vis_mask"],
        )

        self.assertTrue(torch.equal(g_a, g_b))
        self.assertFalse(torch.allclose(memory_a, memory_b))

    def test_style_only_true_null_branch_matches_conditional_branch(self):
        flow = _build_flow("style_only")
        inputs = _inputs()
        g, memory, _ = flow.encode_cond(
            inputs["k2d"],
            inputs["tau_cond"],
            inputs["style_id"],
            inputs["domain_id"],
            apply_style_dropout=False,
            vis_mask=inputs["vis_mask"],
        )
        conditional, null = flow(
            inputs["x_t"],
            inputs["t_flow"],
            inputs["tau_out"],
            mem=memory,
            g=g,
            mem_mask=torch.zeros(2, 7, dtype=torch.bool),
            tau_cond=inputs["tau_cond"],
            style_id=inputs["style_id"],
            domain_id=inputs["domain_id"],
            return_cfg_pair=True,
        )

        self.assertTrue(torch.equal(conditional, null))

    def test_disabled_parameters_remain_in_backward_graph(self):
        flow = _build_flow("style_only")
        inputs = _inputs()
        prediction = flow(
            inputs["x_t"],
            inputs["t_flow"],
            inputs["tau_out"],
            k2d=inputs["k2d"],
            tau_cond=inputs["tau_cond"],
            style_id=inputs["style_id"],
            domain_id=inputs["domain_id"],
            apply_style_dropout=False,
            vis_mask=inputs["vis_mask"],
        )
        prediction.square().mean().backward()

        missing = [
            name
            for name, parameter in flow.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
