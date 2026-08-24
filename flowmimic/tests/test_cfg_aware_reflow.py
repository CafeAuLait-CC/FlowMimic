import unittest

import torch

from flowmimic.scripts.train_flow import _student_rollout_states
from flowmimic.src.model.flow.solver import (
    cfg_branch_preservation_loss,
    combine_cfg_velocities,
    solve_flow_to_time,
)
from flowmimic.src.training.reflow_sampling import (
    FLOW_TIME_CATEGORY_NAMES,
    normalized_flow_time_probabilities,
    sample_reflow_times,
)


class CfgAwareReflowTest(unittest.TestCase):
    def test_per_row_guidance_matches_cfg_equation(self):
        v_cond = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
        v_null = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        scales = torch.tensor([2.5, 1.0, 0.0])

        combined = combine_cfg_velocities(v_cond, v_null, scales)

        expected = torch.tensor([1.75, 2.0, 0.5]).view(3, 1, 1)
        torch.testing.assert_close(combined, expected)

    def test_scalar_guidance_remains_backward_compatible(self):
        v_cond = torch.randn(4, 2, 8)
        v_null = torch.randn(4, 2, 8)

        combined = combine_cfg_velocities(v_cond, v_null, 1.2)

        torch.testing.assert_close(
            combined,
            v_null + 1.2 * (v_cond - v_null),
        )

    def test_rejects_non_batch_vector(self):
        with self.assertRaises(ValueError):
            combine_cfg_velocities(
                torch.zeros(3, 2, 8),
                torch.zeros(3, 2, 8),
                torch.ones(2),
            )

    def test_branch_preservation_matches_teacher_and_propagates_student_gradients(self):
        student_cond = torch.tensor([[[1.0]], [[3.0]]], requires_grad=True)
        student_null = torch.tensor([[[0.0]], [[2.0]]], requires_grad=True)
        teacher_cond = torch.tensor([[[0.0]], [[1.0]]], requires_grad=True)
        teacher_null = torch.tensor([[[1.0]], [[1.0]]], requires_grad=True)

        loss, cond_loss, null_loss = cfg_branch_preservation_loss(
            student_cond,
            student_null,
            teacher_cond,
            teacher_null,
        )

        self.assertAlmostEqual(cond_loss.item(), 2.5)
        self.assertAlmostEqual(null_loss.item(), 1.0)
        self.assertAlmostEqual(loss.item(), 1.75)
        loss.backward()
        self.assertIsNotNone(student_cond.grad)
        self.assertIsNotNone(student_null.grad)
        self.assertIsNone(teacher_cond.grad)
        self.assertIsNone(teacher_null.grad)

    def test_branch_preservation_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            cfg_branch_preservation_loss(
                torch.zeros(2, 3, 4),
                torch.zeros(2, 3, 4),
                torch.zeros(2, 2, 4),
                torch.zeros(2, 3, 4),
            )

    def test_partial_solver_supports_per_row_end_times(self):
        def constant_flow(x, t, *args, **kwargs):
            del t, args, kwargs
            return torch.ones_like(x) * 2.0

        x0 = torch.zeros(3, 2, 4)
        result = solve_flow_to_time(
            constant_flow,
            x0,
            {"tau_out": torch.linspace(0.0, 1.0, 2), "mem": None, "g": None},
            torch.tensor([0.0, 0.25, 1.0]),
            num_steps=2,
            method="heun",
        )
        expected = torch.tensor([0.0, 0.5, 2.0]).view(3, 1, 1).expand_as(x0)
        torch.testing.assert_close(result, expected)

    def test_endpoint_aware_time_sampling(self):
        config = {
            "uniform": 0.0,
            "exact_start": 0.5,
            "near_start": 0.0,
            "near_end": 0.0,
            "exact_end": 0.5,
            "near_width": 0.1,
        }
        torch.manual_seed(7)
        times, categories = sample_reflow_times(
            256,
            config,
            torch.device("cpu"),
        )
        self.assertTrue(bool(((times == 0.0) | (times == 1.0)).all()))
        self.assertEqual(categories.shape, (256,))
        self.assertEqual(len(FLOW_TIME_CATEGORY_NAMES), 5)

    def test_time_sampling_validation(self):
        with self.assertRaises(ValueError):
            normalized_flow_time_probabilities({"uniform": 0.0})
        with self.assertRaises(ValueError):
            normalized_flow_time_probabilities(
                {"uniform": 1.0, "near_width": 0.75}
            )

    def test_student_rollout_states_are_subsetted_and_detached(self):
        class ConstantFlowModel:
            def __init__(self):
                self.training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def flow(self, x, t, tau_out, mem, g, mem_mask, tau_cond):
                del t, tau_out, mem, g, mem_mask, tau_cond
                return torch.full_like(x, 2.0)

        model = ConstantFlowModel()
        x0 = torch.zeros(3, 2, 4)
        end_time = torch.tensor([0.0, 0.25, 0.75])
        rows = torch.tensor([False, True, True])
        condition = {
            "tau_out": torch.linspace(0.0, 1.0, 2),
            "tau_cond": torch.zeros(3, 1),
            "mem": torch.zeros(3, 1, 2),
            "g": torch.zeros(3, 2),
            "mem_mask": torch.zeros(3, 1, dtype=torch.bool),
            "guidance_scale": torch.ones(3),
        }

        states, indices = _student_rollout_states(
            model,
            x0,
            end_time,
            condition,
            rows,
            (1,),
            "heun",
            1,
        )

        torch.testing.assert_close(indices, torch.tensor([1, 2]))
        expected = torch.tensor([0.5, 1.5]).view(2, 1, 1).expand_as(states)
        torch.testing.assert_close(states, expected)
        self.assertFalse(states.requires_grad)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
