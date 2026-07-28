from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from characterize_source_shift import (  # noqa: E402
    bootstrap_mean,
    grid_bin_count,
    origin_to_user_values,
    session_to_user_values,
    sport_user_session_share,
)


class SourceShiftCharacterizationTests(unittest.TestCase):
    def test_origin_hierarchy_equalizes_sessions_and_users(self) -> None:
        values = np.array([0.0, 2.0, 10.0, 4.0])
        users = np.array([1, 1, 1, 2])
        sessions = np.array([10, 10, 11, 20])
        result = origin_to_user_values(values, users, sessions)
        self.assertAlmostEqual(float(result.loc[1]), 5.5)
        self.assertAlmostEqual(float(result.loc[2]), 4.0)

    def test_session_hierarchy_equalizes_users(self) -> None:
        values = np.array([1.0, 3.0, 8.0])
        users = np.array([1, 1, 2])
        result = session_to_user_values(values, users)
        self.assertAlmostEqual(float(result.loc[1]), 2.0)
        self.assertAlmostEqual(float(result.loc[2]), 8.0)

    def test_grid_bin_count_matches_right_closed_ceil_rule(self) -> None:
        result = grid_bin_count(
            np.array([0.0, 1.0, 11.0]), np.array([20.0, 20.0, 31.0])
        )
        np.testing.assert_array_equal(result, np.array([3, 2, 3]))

    def test_sport_share_includes_zero_share_users(self) -> None:
        users = np.array([1, 1, 2, 2])
        sports = np.array([1, 3, 3, 3])
        result = sport_user_session_share(users, sports, 1)
        self.assertAlmostEqual(float(result.loc[1]), 0.5)
        self.assertAlmostEqual(float(result.loc[2]), 0.0)
        self.assertAlmostEqual(float(result.mean()), 0.25)

    def test_bootstrap_mean_uses_requested_draws(self) -> None:
        values = np.array([1.0, 3.0, 8.0])
        indices = np.array([[0, 1, 2], [1, 1, 1]])
        np.testing.assert_allclose(bootstrap_mean(values, indices), [4.0, 3.0])


if __name__ == "__main__":
    unittest.main()
