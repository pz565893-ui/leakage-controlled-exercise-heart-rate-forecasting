from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootstrap_external_sport_uncertainty import (  # noqa: E402
    calibrated_bounds,
    weighted_interval_score,
)
from standardize_external_sport_uncertainty import (  # noqa: E402
    METRIC_UNITS,
    PUBLIC_RESULT_COLUMNS,
    bootstrap_family_nanmean,
    bootstrap_matrix_mean,
    comparison_rows,
    hierarchical_user_family_metrics,
    hierarchical_user_metrics,
    origin_uncertainty_metrics,
    weighted_family_metrics,
)


class ExternalSportUncertaintyStandardizationTests(unittest.TestCase):
    def test_hierarchical_metrics_equalize_origins_sessions_and_users(self) -> None:
        metrics = {
            "first": np.array([0.0, 2.0, 10.0, 5.0]),
            "second": np.array([0.0, 20.0, 100.0, 50.0]),
        }
        users = np.array([1, 1, 1, 2])
        sessions = np.array([10, 10, 11, 20])
        result = hierarchical_user_metrics(metrics, users, sessions)
        self.assertAlmostEqual(float(result.loc[1, "first"]), 5.5)
        self.assertAlmostEqual(float(result.loc[2, "first"]), 5.0)
        self.assertAlmostEqual(float(result["first"].mean()), 5.25)
        np.testing.assert_allclose(
            result["second"].to_numpy(), 10.0 * result["first"].to_numpy()
        )

    def test_family_cube_preserves_missing_families_and_session_weighting(self) -> None:
        metrics = {
            "metric": np.array([0.0, 2.0, 10.0, 5.0, 9.0]),
        }
        users = np.array([1, 1, 1, 2, 2])
        sessions = np.array([10, 10, 11, 20, 21])
        sports = np.array([1, 1, 3, 1, 2])
        user_ids, cube = hierarchical_user_family_metrics(
            metrics, users, sessions, sports
        )
        np.testing.assert_array_equal(user_ids, np.array([1, 2]))
        self.assertEqual(cube.shape, (2, 3, 1))
        self.assertAlmostEqual(float(cube[0, 0, 0]), 1.0)
        self.assertTrue(np.isnan(cube[0, 1, 0]))
        self.assertAlmostEqual(float(cube[0, 2, 0]), 10.0)
        self.assertAlmostEqual(float(cube[1, 0, 0]), 5.0)
        self.assertAlmostEqual(float(cube[1, 1, 0]), 9.0)

    def test_bootstrap_reuses_joint_user_draws(self) -> None:
        indices = np.array([[0, 1], [0, 0], [1, 1]])
        natural = np.array([[1.0, 10.0], [3.0, 30.0]])
        natural_bootstrap = bootstrap_matrix_mean(natural, indices)
        np.testing.assert_allclose(
            natural_bootstrap[:, 1], 10.0 * natural_bootstrap[:, 0]
        )

        family = np.array(
            [
                [[1.0], [np.nan], [5.0]],
                [[3.0], [7.0], [np.nan]],
            ]
        )
        family_indices = np.array([[0, 1], [1, 0]])
        family_bootstrap = bootstrap_family_nanmean(family, family_indices)
        np.testing.assert_allclose(
            family_bootstrap[0, :, 0], np.array([2.0, 7.0, 5.0])
        )
        np.testing.assert_allclose(
            family_bootstrap[1, :, 0], np.array([2.0, 7.0, 5.0])
        )

    def test_weighted_family_metrics_handles_points_and_replicates(self) -> None:
        weights = np.array([0.5, 0.25, 0.25])
        points = np.array(
            [
                [2.0, 20.0],
                [4.0, 40.0],
                [8.0, 80.0],
            ]
        )
        np.testing.assert_allclose(
            weighted_family_metrics(points, weights), np.array([4.0, 40.0])
        )
        replicates = np.stack([points, points * 2.0])
        np.testing.assert_allclose(
            weighted_family_metrics(replicates, weights),
            np.array([[4.0, 40.0], [8.0, 80.0]]),
        )
        with self.assertRaises(ValueError):
            weighted_family_metrics(points, np.array([0.5, 0.5, 0.5]))

    def test_origin_metrics_reuse_frozen_post_cqr_metric_definitions(self) -> None:
        prediction = np.array(
            [
                [50.0, 60.0, 70.0, 100.0, 130.0, 140.0, 150.0],
                [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0],
            ]
        )
        target = np.array([151.0, 100.0])
        thresholds = {
            "0.5": [1.0, 0.0, 0.0],
            "0.8": [2.0, 0.0, 0.0],
            "0.9": [3.0, 0.0, 0.0],
        }
        result = origin_uncertainty_metrics(
            prediction, target, thresholds, horizon_position=0
        )
        bounds = calibrated_bounds(prediction, thresholds, 0)
        lower_90, upper_90 = bounds[0.90]
        np.testing.assert_allclose(
            result["picp_90"],
            ((target >= lower_90) & (target <= upper_90)).astype(float),
        )
        np.testing.assert_allclose(
            result["mean_90_interval_width_bpm"], upper_90 - lower_90
        )
        np.testing.assert_allclose(
            result["weighted_interval_score"],
            weighted_interval_score(prediction[:, 3], target, bounds),
        )

    def test_public_comparison_rows_are_aggregate_only(self) -> None:
        internal_point = np.array([0.90, 20.0, 4.0])
        external_point = np.array([0.80, 24.0, 5.0])
        internal_bootstrap = np.tile(internal_point, (4, 1))
        external_bootstrap = np.tile(external_point, (4, 1))
        support = {"users": 10, "sessions": 20, "origins": 30}
        rows = comparison_rows(
            scope="test",
            family="all",
            horizon=60,
            internal_point=internal_point,
            external_point=external_point,
            internal_bootstrap=internal_bootstrap,
            external_bootstrap=external_bootstrap,
            internal_support=support,
            external_support=support,
            replicates=4,
        )
        self.assertEqual([row["metric"] for row in rows], list(METRIC_UNITS))
        self.assertEqual(tuple(rows[0]), PUBLIC_RESULT_COLUMNS)
        self.assertAlmostEqual(rows[0]["external_minus_internal"], -0.10)
        forbidden = {
            "user",
            "user_id",
            "user_index",
            "session",
            "session_id",
            "session_index",
            "row_index",
        }
        self.assertFalse(forbidden.intersection(PUBLIC_RESULT_COLUMNS))
        self.assertFalse(rows[0]["causal_source_or_device_effect_claimed"])
        self.assertFalse(rows[0]["user_level_conformal_guarantee_claimed"])


if __name__ == "__main__":
    unittest.main()
