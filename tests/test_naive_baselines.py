from __future__ import annotations

import sqlite3
import sys
import unittest
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_naive_baselines import (  # noqa: E402
    ewma_prediction,
    extract_hr_context,
    linear_trend_predictions,
    persistence_prediction,
    regimes_for_row,
    valid_observations,
)


class NaiveBaselineTests(unittest.TestCase):
    def test_context_slice_uses_thirty_right_closed_bins(self) -> None:
        values = array("f", range(100))
        mask = array("B", [1] * 100)
        context, context_mask = extract_hr_context(600.0, 1, 100, values, mask)
        self.assertEqual(context, list(range(30, 60)))
        self.assertEqual(sum(context_mask), 30)

    def test_persistence_uses_last_observed_value(self) -> None:
        observations = valid_observations([100.0, 0.0, 130.0], [1, 0, 1])
        self.assertEqual(persistence_prediction(observations), 130.0)

    def test_ewma_skips_missing_bins(self) -> None:
        observations = [(-20.0, 100.0), (0.0, 140.0)]
        self.assertAlmostEqual(ewma_prediction(observations, 0.25), 110.0)

    def test_linear_trend_extrapolates_and_clips(self) -> None:
        observations = [(-20.0, 100.0), (-10.0, 110.0), (0.0, 120.0)]
        predictions = linear_trend_predictions(observations, (10, 180))
        self.assertAlmostEqual(predictions[0], 130.0)
        self.assertEqual(predictions[1], 240.0)

    def test_regime_assignment_is_explicit(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT 'Endomondo' AS dataset, 'test' AS within_user_temporal_partition,
                   'test' AS unseen_user_partition, 1 AS sport_shift_candidate,
                   'running' AS sport_family, 'test' AS joint_shift_user_partition,
                   '' AS primary_external_partition
            """
        ).fetchone()
        self.assertEqual(
            regimes_for_row(row),
            [
                "internal_temporal_test",
                "unseen_user_test",
                "unseen_sport__running",
                "joint_user_sport__running",
            ],
        )
        connection.close()

    def test_external_regime_includes_overall_and_sport_family(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT 'GoldenCheetah' AS dataset, '' AS within_user_temporal_partition,
                   '' AS unseen_user_partition, 0 AS sport_shift_candidate,
                   'outdoor_cycling' AS sport_family,
                   '' AS joint_shift_user_partition,
                   'frozen_external_test' AS primary_external_partition
            """
        ).fetchone()
        self.assertEqual(
            regimes_for_row(row),
            [
                "goldencheetah_frozen_external",
                "goldencheetah_external__outdoor_cycling",
            ],
        )
        connection.close()


if __name__ == "__main__":
    unittest.main()
