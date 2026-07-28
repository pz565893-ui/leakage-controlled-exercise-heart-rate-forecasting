from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from standardize_external_sport_composition import (  # noqa: E402
    bootstrap_nanmean,
    hierarchical_user_mae,
    user_family_mae_matrix,
    weighted_family_average,
)


class ExternalSportStandardizationTests(unittest.TestCase):
    def test_hierarchical_user_mae_equalizes_sessions(self) -> None:
        prediction = np.array([0.0, 2.0, 10.0, 5.0])
        target = np.zeros(4)
        users = np.array([1, 1, 1, 2])
        sessions = np.array([10, 10, 11, 20])
        result = hierarchical_user_mae(prediction, target, users, sessions)
        self.assertAlmostEqual(float(result.loc[1]), 5.5)
        self.assertAlmostEqual(float(result.loc[2]), 5.0)

    def test_user_family_matrix_preserves_missing_families(self) -> None:
        prediction = np.array([1.0, 3.0, 2.0, 4.0])
        target = np.zeros(4)
        users = np.array([1, 1, 2, 2])
        sessions = np.array([10, 11, 20, 21])
        sports = np.array([1, 3, 1, 2])
        user_ids, matrix = user_family_mae_matrix(
            prediction, target, users, sessions, sports
        )
        np.testing.assert_array_equal(user_ids, np.array([1, 2]))
        np.testing.assert_allclose(matrix[0, [0, 2]], np.array([1.0, 3.0]))
        self.assertTrue(np.isnan(matrix[0, 1]))
        np.testing.assert_allclose(matrix[1, [0, 1]], np.array([2.0, 4.0]))

    def test_bootstrap_nanmean_ignores_unsupported_users(self) -> None:
        values = np.array([1.0, np.nan, 3.0])
        indices = np.array([[0, 1, 2], [0, 0, 1]])
        result = bootstrap_nanmean(values, indices)
        np.testing.assert_allclose(result, np.array([2.0, 1.0]))

    def test_weighted_family_average_handles_point_and_bootstrap_arrays(self) -> None:
        weights = np.array([0.5, 0.25, 0.25])
        self.assertAlmostEqual(
            float(weighted_family_average(np.array([2.0, 4.0, 8.0]), weights)),
            4.0,
        )
        bootstrap = np.array([[2.0, 4.0, 8.0], [4.0, 4.0, 4.0]])
        np.testing.assert_allclose(
            weighted_family_average(bootstrap, weights), np.array([4.0, 4.0])
        )

    def test_weighted_family_average_rejects_invalid_weights(self) -> None:
        with self.assertRaises(ValueError):
            weighted_family_average(
                np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5])
            )


if __name__ == "__main__":
    unittest.main()
