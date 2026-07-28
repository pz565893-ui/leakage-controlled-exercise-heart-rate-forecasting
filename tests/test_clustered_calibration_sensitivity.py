from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_clustered_calibration_sensitivity import (  # noqa: E402
    hierarchical_origin_weights,
    paired_replicate_user_bootstrap,
    sampled_thresholds,
    weighted_quantile_higher,
)


class ClusteredCalibrationSensitivityTests(unittest.TestCase):
    def test_hierarchical_weights_equalize_users_and_sessions(self) -> None:
        users = np.array([1, 1, 1, 2, 2, 2])
        sessions = np.array([10, 10, 11, 20, 20, 20])
        weights = hierarchical_origin_weights(users, sessions)
        self.assertAlmostEqual(float(weights[users == 1].sum()), 0.5)
        self.assertAlmostEqual(float(weights[users == 2].sum()), 0.5)
        self.assertAlmostEqual(float(weights[sessions == 10].sum()), 0.25)
        self.assertAlmostEqual(float(weights[sessions == 11].sum()), 0.25)
        self.assertAlmostEqual(float(weights[sessions == 20].sum()), 0.5)

    def test_weighted_higher_quantile_uses_cumulative_weight(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 3.0])
        weights = np.array([0.1, 0.2, 0.6, 0.1])
        self.assertEqual(weighted_quantile_higher(values, weights, 0.50), 2.0)
        self.assertEqual(weighted_quantile_higher(values, weights, 0.95), 3.0)

    def test_sampled_thresholds_use_cluster_count_rank_and_nonnegative_expansion(self) -> None:
        scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        sampled = np.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]])
        thresholds = sampled_thresholds(scores, sampled, 0.50)
        np.testing.assert_allclose(thresholds, np.array([0.0, 0.0]))

    def test_paired_replicate_bootstrap_preserves_replicate_rows(self) -> None:
        values = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        result = paired_replicate_user_bootstrap(values, seed=7)
        np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
