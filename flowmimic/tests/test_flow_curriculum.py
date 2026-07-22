import unittest

from flowmimic.src.training.flow_curriculum import (
    SparsePatternPhase1Curriculum,
    UnifiedRound0Curriculum,
)


CONFIG = {
    "reference_updates_per_epoch": 10,
    "warmup_end_update": 50,
    "dense_end_update": 400,
    "condition_start_update": 600,
    "condition_full_update": 800,
    "mid_end_update": 800,
    "sparse_end_update": 1400,
    "max_updates": 1400,
    "optional_max_updates": 1600,
    "eval_every_updates": 50,
    "lr_peak": 2e-4,
    "lr_mid": 1e-4,
    "lr_final": 5e-5,
    "condition_choices": [196, 98, 49, 28, 14, 7],
    "dense_probs": [1, 0, 0, 0, 0, 0],
    "mid_probs": [0.375, 0.1875, 0.25, 0.1875, 0, 0],
    "final_probs": [0.30, 0.15, 0.20, 0.15, 0.12, 0.08],
    "solver_steps_dense": [16],
    "solver_steps_sparse": [8, 16],
}

PHASE1_CONFIG = {
    "name": "sparse_pattern_phase1",
    "source_optimizer_updates": 1000,
    "reference_updates_per_epoch": 10,
    "pattern_ramp_updates": 100,
    "relative_max_updates": 300,
    "relative_optional_max_updates": 400,
    "eval_every_updates": 50,
    "learning_rate": 5e-5,
    "condition_weight_scale": 1.0,
    "condition_choices": [196, 98, 49, 28, 14, 7],
    "condition_probs": [0.30, 0.15, 0.20, 0.15, 0.12, 0.08],
    "condition_pattern_choices": ["even", "random", "boundary_gap"],
    "pattern_start_probs": [1.0, 0.0, 0.0],
    "pattern_final_probs": [0.50, 0.30, 0.20],
    "solver_steps": [8, 16],
}


class UnifiedRound0CurriculumTest(unittest.TestCase):
    def setUp(self):
        self.curriculum = UnifiedRound0Curriculum(CONFIG)

    def test_learning_rate_schedule(self):
        self.assertAlmostEqual(self.curriculum.state(0).learning_rate, 4e-6)
        self.assertAlmostEqual(self.curriculum.state(49).learning_rate, 2e-4)
        self.assertAlmostEqual(self.curriculum.state(400).learning_rate, 2e-4)
        self.assertAlmostEqual(self.curriculum.state(800).learning_rate, 1e-4)
        self.assertAlmostEqual(self.curriculum.state(1400).learning_rate, 5e-5)

    def test_condition_weight_ramp(self):
        self.assertEqual(self.curriculum.state(600).condition_weight_scale, 0.0)
        self.assertAlmostEqual(
            self.curriculum.state(700).condition_weight_scale, 0.5
        )
        self.assertEqual(self.curriculum.state(800).condition_weight_scale, 1.0)

    def test_density_schedule(self):
        self.assertEqual(
            self.curriculum.state(0).condition_probs,
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        mid = self.curriculum.state(800)
        self.assertEqual(mid.phase, "sparse_density_ramp")
        self.assertEqual(mid.solver_steps, (8, 16))
        for actual, expected in zip(mid.condition_probs, CONFIG["mid_probs"]):
            self.assertAlmostEqual(actual, expected)
        final = self.curriculum.state(1400)
        for actual, expected in zip(final.condition_probs, CONFIG["final_probs"]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(final.condition_probs), 1.0)

    def test_integrated_pattern_and_joint_quota_ramps(self):
        config = dict(CONFIG)
        config.update(
            {
                "name": "unified_round0_phase1d",
                "max_updates": 1800,
                "optional_max_updates": 2000,
                "condition_pattern_choices": ["even", "random", "boundary_gap"],
                "pattern_start_probs": [1.0, 0.0, 0.0],
                "pattern_final_probs": [0.50, 0.30, 0.20],
                "pattern_ramp_start_update": 400,
                "pattern_ramp_end_update": 800,
                "joint_quota_start_update": 800,
                "joint_quota_full_update": 1200,
                "condition_joint_quotas": [
                    {"pattern": "boundary_gap", "count": 7, "fraction": 0.08},
                    {"pattern": "random", "count": 7, "fraction": 0.05},
                    {"pattern": "boundary_gap", "count": 14, "fraction": 0.04},
                ],
            }
        )
        curriculum = UnifiedRound0Curriculum(config)

        start = curriculum.state(400)
        self.assertEqual(start.condition_pattern_probs, (1.0, 0.0, 0.0))
        self.assertEqual(start.condition_joint_quotas, ())

        pattern_mid = curriculum.state(600)
        for actual, expected in zip(
            pattern_mid.condition_pattern_probs,
            (0.75, 0.15, 0.10),
        ):
            self.assertAlmostEqual(actual, expected)

        quota_mid = curriculum.state(1000)
        self.assertEqual(quota_mid.condition_pattern_probs, (0.50, 0.30, 0.20))
        for actual, expected in zip(
            quota_mid.condition_joint_quotas,
            (
                ("boundary_gap", 7, 0.04),
                ("random", 7, 0.025),
                ("boundary_gap", 14, 0.02),
            ),
        ):
            self.assertEqual(actual[:2], expected[:2])
            self.assertAlmostEqual(actual[2], expected[2])

        final = curriculum.state(1400)
        self.assertEqual(
            final.condition_joint_quotas,
            (
                ("boundary_gap", 7, 0.08),
                ("random", 7, 0.05),
                ("boundary_gap", 14, 0.04),
            ),
        )
        self.assertEqual(curriculum.metadata()["name"], "unified_round0_phase1d")


class SparsePatternPhase1CurriculumTest(unittest.TestCase):
    def setUp(self):
        self.curriculum = SparsePatternPhase1Curriculum(PHASE1_CONFIG)

    def test_absolute_update_bounds(self):
        self.assertEqual(self.curriculum.max_updates, 1300)
        self.assertEqual(self.curriculum.optional_max_updates, 1400)

    def test_pattern_ramp_is_relative_to_source_checkpoint(self):
        start = self.curriculum.state(1000)
        self.assertEqual(start.phase, "pattern_ramp")
        self.assertEqual(start.condition_pattern_probs, (1.0, 0.0, 0.0))

        midpoint = self.curriculum.state(1050)
        for actual, expected in zip(
            midpoint.condition_pattern_probs,
            (0.75, 0.15, 0.10),
        ):
            self.assertAlmostEqual(actual, expected)

        final = self.curriculum.state(1100)
        self.assertEqual(final.phase, "pattern_hold")
        for actual, expected in zip(
            final.condition_pattern_probs,
            (0.50, 0.30, 0.20),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_phase1_keeps_density_lr_and_regularization_fixed(self):
        state = self.curriculum.state(1250)
        for actual, expected in zip(
            state.condition_probs,
            PHASE1_CONFIG["condition_probs"],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(state.learning_rate, 5e-5)
        self.assertEqual(state.condition_weight_scale, 1.0)
        self.assertEqual(state.solver_steps, (8, 16))

    def test_joint_condition_quotas_are_exposed_in_state(self):
        config = dict(PHASE1_CONFIG)
        config["condition_joint_quotas"] = [
            {"pattern": "boundary_gap", "count": 7, "fraction": 0.10},
            {"pattern": "boundary_gap", "count": 14, "fraction": 0.05},
        ]
        state = SparsePatternPhase1Curriculum(config).state(1250)
        self.assertEqual(
            state.condition_joint_quotas,
            (("boundary_gap", 7, 0.10), ("boundary_gap", 14, 0.05)),
        )


if __name__ == "__main__":
    unittest.main()
