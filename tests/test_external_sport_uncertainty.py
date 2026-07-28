from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootstrap_external_sport_uncertainty import (  # noqa: E402
    bootstrap_user_means,
    calibrated_bounds,
    hierarchical_user_metrics,
    weighted_interval_score,
)


class ExternalSportUncertaintyTests(unittest.TestCase):
    def test_hierarchical_metrics_equalize_origins_sessions_and_users(self) -> None:
        values = np.array([0.0, 2.0, 10.0, 5.0])
        users = np.array([1, 1, 1, 2])
        sessions = np.array([10, 10, 11, 20])
        result = hierarchical_user_metrics(
            {"metric": values}, users, sessions
        )
        self.assertAlmostEqual(float(result.loc[1, "metric"]), 5.5)
        self.assertAlmostEqual(float(result.loc[2, "metric"]), 5.0)
        self.assertAlmostEqual(float(result["metric"].mean()), 5.25)

    def test_calibrated_bounds_apply_frozen_expansion_and_clipping(self) -> None:
        prediction = np.array(
            [
                [25.0, 28.0, 31.0, 100.0, 180.0, 235.0, 239.0],
                [40.0, 50.0, 60.0, 100.0, 140.0, 150.0, 160.0],
            ]
        )
        thresholds = {
            "0.5": [2.0, 0.0, 0.0],
            "0.8": [3.0, 0.0, 0.0],
            "0.9": [4.0, 0.0, 0.0],
        }
        bounds = calibrated_bounds(prediction, thresholds, 0)
        np.testing.assert_allclose(bounds[0.50][0], np.array([30.0, 58.0]))
        np.testing.assert_allclose(bounds[0.50][1], np.array([182.0, 142.0]))
        np.testing.assert_allclose(bounds[0.90][0], np.array([30.0, 36.0]))
        np.testing.assert_allclose(bounds[0.90][1], np.array([240.0, 164.0]))

    def test_weighted_interval_score_rewards_exact_narrow_intervals(self) -> None:
        target = np.array([100.0, 120.0])
        median = target.copy()
        exact = {
            0.50: (target.copy(), target.copy()),
            0.80: (target.copy(), target.copy()),
            0.90: (target.copy(), target.copy()),
        }
        np.testing.assert_allclose(
            weighted_interval_score(median, target, exact), np.zeros(2)
        )

    def test_bootstrap_resamples_user_metric_vectors_jointly(self) -> None:
        values = np.array([[1.0, 10.0], [3.0, 30.0]])
        result = bootstrap_user_means(values, replicates=50, seed=7)
        np.testing.assert_allclose(result[:, 1], 10.0 * result[:, 0])
        self.assertEqual(result.shape, (50, 2))


if __name__ == "__main__":
    unittest.main()
