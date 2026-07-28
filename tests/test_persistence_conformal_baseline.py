from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_persistence_conformal_baseline import (  # noqa: E402
    finite_sample_radius,
    last_observed_hr,
    weighted_interval_score,
)


class PersistenceConformalBaselineTests(unittest.TestCase):
    def test_finite_sample_radius_uses_higher_order_statistic(self) -> None:
        scores = np.arange(1.0, 11.0)
        # ceil((10 + 1) * 0.8) = 9, hence the ninth order statistic.
        self.assertEqual(finite_sample_radius(scores, 0.8), 9.0)

    def test_finite_sample_radii_are_monotone_in_coverage(self) -> None:
        scores = np.array([0.2, 0.5, 1.0, 1.5, 3.0, 8.0])
        radii = [
            finite_sample_radius(scores, coverage)
            for coverage in (0.5, 0.8, 0.9)
        ]
        self.assertEqual(radii, sorted(radii))

    def test_last_observed_hr_uses_mask_not_final_bin(self) -> None:
        values = np.zeros((2, 4, 3), dtype=np.float32)
        masks = np.zeros((2, 4, 3), dtype=np.uint8)
        values[0, :, 0] = [100.0, 101.0, 102.0, 0.0]
        masks[0, :3, 0] = 1
        values[1, :, 0] = [120.0, 0.0, 121.0, 122.0]
        masks[1, [0, 2, 3], 0] = 1
        prediction = last_observed_hr(values, masks, np.array([0, 1]), chunk_size=1)
        np.testing.assert_allclose(prediction, [102.0, 122.0])

    def test_weighted_interval_score_is_zero_for_perfect_degenerate_forecast(self) -> None:
        point = np.array([100.0, 120.0])
        target = point.copy()
        bounds = {
            0.5: (point.copy(), point.copy()),
            0.8: (point.copy(), point.copy()),
            0.9: (point.copy(), point.copy()),
        }
        np.testing.assert_allclose(weighted_interval_score(point, target, bounds), 0.0)


if __name__ == "__main__":
    unittest.main()
