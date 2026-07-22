import unittest

import numpy as np
import torch

from flowmimic.scripts.train_flow import (
    _apply_condition_count_schedule,
    _merge_batches,
)
from flowmimic.src.model.vae.datasets.condition_sampling import (
    condition_sample_key,
    deterministic_condition_indices,
    sample_condition_indices,
)


class ConditionSamplingTest(unittest.TestCase):
    def test_single_dataset_merge_preserves_batch_storage(self):
        motion = torch.randn(2, 196, 263)
        batch = {
            "motion": motion,
            "domain_id": torch.zeros(2, dtype=torch.long),
            "style_id": torch.zeros(2, dtype=torch.long),
            "mask": torch.ones(2, 196, dtype=torch.bool),
            "meta": {"path": ["a", "b"], "start": [0, 1]},
            "k2d": torch.randn(2, 196, 25, 2),
            "vis": torch.ones(2, 196, 25),
            "conf": torch.ones(2, 196, 25),
            "tau_cond": torch.randn(2, 196),
            "mask_cond": torch.ones(2, 196, dtype=torch.bool),
        }

        merged = _merge_batches([batch])

        self.assertEqual(merged[0].data_ptr(), motion.data_ptr())
        self.assertEqual(merged[4][1]["path"], "b")

    def test_even_indices_are_sorted_unique_and_exact(self):
        for count in (7, 14, 28, 49, 98, 196):
            with self.subTest(count=count):
                indices = sample_condition_indices(196, count, pattern="even")
                self.assertEqual(indices.shape, (count,))
                self.assertTrue(np.all(np.diff(indices) > 0))
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-1], 195)

    def test_random_indices_are_reproducible_per_sample(self):
        key = condition_sample_key(
            "/tmp/gBR_sBM_cAll_d01_mBR0_ch01.pkl", "01", 0, 196
        )
        first = deterministic_condition_indices(196, 28, "random", 20260720, key)
        second = deterministic_condition_indices(196, 28, "random", 20260720, key)
        other = deterministic_condition_indices(
            196,
            28,
            "random",
            20260720,
            key.replace("ch01", "ch02"),
        )

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(np.all(np.diff(first) > 0))

    def test_boundary_gap_preserves_boundaries_and_long_gap(self):
        for count in (7, 14, 28, 49, 98):
            with self.subTest(count=count):
                indices = sample_condition_indices(
                    196,
                    count,
                    pattern="boundary_gap",
                    rng=np.random.RandomState(1234),
                )
                self.assertEqual(indices.shape, (count,))
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-1], 195)
                self.assertTrue(np.all(np.diff(indices) > 0))
                expected_min_gap = min(round(0.25 * 196), 196 - count)
                largest_empty_interval = int((np.diff(indices) - 1).max())
                self.assertGreaterEqual(largest_empty_interval, expected_min_gap)

    def test_dense_boundary_gap_reduces_to_all_frames(self):
        indices = sample_condition_indices(196, 196, pattern="boundary_gap")
        np.testing.assert_array_equal(indices, np.arange(196))

    def test_unknown_pattern_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported condition pattern"):
            sample_condition_indices(196, 14, pattern="unknown")

    def test_training_random_pattern_keeps_exact_condition_count(self):
        tensors = self._dense_condition_batch(batch_size=32)
        _, _, _, mask = _apply_condition_count_schedule(
            *tensors,
            condition_choices=(14,),
            condition_probs=(1.0,),
            condition_pattern_choices=("random",),
            condition_pattern_probs=(1.0,),
        )
        self.assertTrue(torch.all(mask.sum(dim=1) == 14))
        self.assertGreater(torch.unique(mask, dim=0).shape[0], 1)

    def test_training_boundary_gap_preserves_boundaries_and_long_gap(self):
        tensors = self._dense_condition_batch(batch_size=32)
        k2d, vis, conf, mask = _apply_condition_count_schedule(
            *tensors,
            condition_choices=(14,),
            condition_probs=(1.0,),
            condition_pattern_choices=("boundary_gap",),
            condition_pattern_probs=(1.0,),
        )
        self.assertTrue(torch.all(mask.sum(dim=1) == 14))
        self.assertTrue(torch.all(mask[:, 0]))
        self.assertTrue(torch.all(mask[:, -1]))
        for row in mask:
            indices = torch.nonzero(row, as_tuple=False).flatten()
            largest_empty_interval = int((indices.diff() - 1).max())
            self.assertGreaterEqual(largest_empty_interval, round(0.25 * 196))
        self.assertTrue(torch.all(k2d[~mask] == 0))
        self.assertTrue(torch.all(vis[~mask] == 0))
        self.assertTrue(torch.all(conf[~mask] == 0))

    def test_joint_quotas_reserve_exact_batch_fractions(self):
        tensors = self._dense_condition_batch(batch_size=100)
        _, _, _, mask = _apply_condition_count_schedule(
            *tensors,
            condition_choices=(196, 14, 7),
            condition_probs=(1.0, 0.0, 0.0),
            condition_pattern_choices=("even", "boundary_gap"),
            condition_pattern_probs=(1.0, 0.0),
            condition_joint_quotas=(
                ("boundary_gap", 7, 0.10),
                ("boundary_gap", 14, 0.05),
            ),
        )
        counts = mask.sum(dim=1)
        self.assertEqual(int((counts == 7).sum()), 10)
        self.assertEqual(int((counts == 14).sum()), 5)
        self.assertEqual(int((counts == 196).sum()), 85)

    @staticmethod
    def _dense_condition_batch(batch_size):
        k2d = torch.ones(batch_size, 196, 25, 2)
        vis = torch.ones(batch_size, 196, 25)
        conf = torch.ones(batch_size, 196, 25)
        mask = torch.ones(batch_size, 196, dtype=torch.bool)
        return k2d, vis, conf, mask


if __name__ == "__main__":
    unittest.main()
