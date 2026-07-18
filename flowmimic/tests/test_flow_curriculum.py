import unittest

from flowmimic.src.training.flow_curriculum import UnifiedRound0Curriculum


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


if __name__ == "__main__":
    unittest.main()
