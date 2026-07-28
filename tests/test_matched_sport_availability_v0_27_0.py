from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_matched_sport_availability_v0_27_0 import (  # noqa: E402
    aligned_positions,
    bootstrap_mean_ci,
    per_user_metrics,
)


class MatchedSportAvailabilityTests(unittest.TestCase):
    def test_alignment_requires_exact_subset(self) -> None:
        reference = np.array([2, 4, 8, 10])
        np.testing.assert_array_equal(
            aligned_positions(reference, np.array([4, 10])), np.array([1, 3])
        )
        with self.assertRaises(AssertionError):
            aligned_positions(reference, np.array([4, 9]))

    def test_per_user_metrics_aggregate_origin_then_session_then_user(self) -> None:
        target = np.zeros((4, 3), dtype=np.float32)
        full = np.array([[0, 0, 0], [2, 2, 2], [10, 10, 10], [5, 5, 5]], dtype=np.float32)
        held = full + 1.0
        users = np.array([1, 1, 1, 2])
        sessions = np.array([10, 10, 11, 20])
        result = per_user_metrics(target, full, held, users, sessions)
        horizon = result[result.horizon_seconds == 60].set_index("user")
        self.assertAlmostEqual(float(horizon.loc[1, "full_absolute_error"]), 5.5)
        self.assertAlmostEqual(float(horizon.loc[2, "full_absolute_error"]), 5.0)
        self.assertAlmostEqual(float(horizon.loc[1, "delta_mae_bpm"]), 1.0)

    def test_bootstrap_is_deterministic_and_centred_on_user_mean(self) -> None:
        values = np.array([-1.0, 0.0, 2.0, 3.0])
        first = bootstrap_mean_ci(values, 1000, 7)
        second = bootstrap_mean_ci(values, 1000, 7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 1.0)
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])


if __name__ == "__main__":
    unittest.main()
