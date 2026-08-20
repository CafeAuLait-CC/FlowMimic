import unittest

import torch

from flowmimic.src.model.flow.rect_flow import ConditionalRectFlow
from flowmimic.src.model.flow.teacher import Teacher


class ReflowTeacherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        model = ConditionalRectFlow(
            d_z=8,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_dim=64,
            dropout=0.5,
            num_styles=4,
            style_dim=8,
            cond_dim=32,
            cond_layers=1,
            cond_heads=4,
            p_style_drop=0.5,
            latent_len=4,
            true_null_condition=True,
        )
        self.teacher = Teacher(
            model,
            solver_cfg={"num_steps": 2, "method": "heun"},
        )

    def _raw_condition(self):
        batch_size = 3
        cond_len = 5
        return {
            "k2d": torch.randn(batch_size, cond_len, 25, 2),
            "tau_cond": torch.linspace(0, 1, cond_len).repeat(batch_size, 1),
            "vis_mask": torch.ones(batch_size, cond_len, 25),
            "mask_cond": torch.ones(batch_size, cond_len, dtype=torch.bool),
            "style_id": torch.tensor([1, 2, 3]),
            "domain_id": torch.zeros(batch_size, dtype=torch.long),
            "tau_out": torch.linspace(0, 1, 4),
            "null_mask": torch.tensor([False, True, False]),
        }

    def test_teacher_is_frozen_and_in_eval_mode(self):
        self.assertFalse(self.teacher.model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in self.teacher.model.parameters())
        )

    def test_teacher_owns_condition_encoding_and_is_deterministic(self):
        calls = []
        handle = self.teacher.model.cond_encoder.register_forward_hook(
            lambda *_args: calls.append(1)
        )
        try:
            raw = self._raw_condition()
            cond_a = self.teacher.build_condition_batch(**raw)
            self.assertEqual(len(calls), 1)
            x0 = torch.randn(3, 4, 8)
            out_a = self.teacher.generate_x1_hat(x0, cond_a)
            cond_b = self.teacher.build_condition_batch(**raw)
            out_b = self.teacher.generate_x1_hat(x0, cond_b)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(out_a, out_b, rtol=0.0, atol=0.0)

    def test_true_null_rows_use_one_valid_teacher_token(self):
        cond = self.teacher.build_condition_batch(**self._raw_condition())
        self.assertEqual(tuple(cond["mem"].shape), (3, 5, 32))
        self.assertFalse(bool(cond["mem_mask"][1, 0]))
        self.assertTrue(bool(cond["mem_mask"][1, 1:].all()))


if __name__ == "__main__":
    unittest.main()
