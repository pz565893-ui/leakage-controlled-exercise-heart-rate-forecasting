from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_forecast_origins import (  # noqa: E402
    ORIGIN_COLUMNS,
    build_session_origins,
    context_bin_index,
    evenly_select_mapping,
    interpolate_target,
)


def row_dict(row: tuple[object, ...]) -> dict[str, object]:
    return dict(zip(ORIGIN_COLUMNS, row))


class ForecastOriginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split = {
            "user_id": "user-1",
            "sport_family": "running",
            "unseen_user_partition": "test",
            "within_user_temporal_partition": "test",
            "sport_shift_candidate": "True",
            "joint_shift_user_partition": "test",
        }

    def test_regular_sequence_constructs_causal_multi_horizon_origins(self) -> None:
        timestamps = [float(value) for value in range(0, 1001, 10)]
        heart_rate = [100.0 + 0.01 * value for value in timestamps]
        rows, candidates, rejections = build_session_origins(
            "Endomondo", "1", self.split, timestamps, heart_rate
        )
        self.assertEqual(candidates, 7)
        self.assertEqual(len(rows), 7)
        self.assertFalse(rejections)
        for packed in rows:
            row = row_dict(packed)
            self.assertLessEqual(row["input_last_time"], row["origin_time"])
            self.assertGreater(row["target_min_source_time"], row["origin_time"])
            self.assertEqual(row["context_valid_bins"], 30)
            self.assertEqual(row["context_hr_coverage"], 1.0)
            self.assertAlmostEqual(
                row["target_hr_60"], 100.0 + 0.01 * (row["origin_time"] + 60)
            )

    def test_context_gap_is_rejected(self) -> None:
        timestamps = [
            float(value)
            for value in range(0, 1001, 10)
            if not 150 <= value <= 200
        ]
        heart_rate = [120.0 for _ in timestamps]
        rows, candidates, rejections = build_session_origins(
            "Endomondo", "2", self.split, timestamps, heart_rate
        )
        self.assertGreater(candidates, 0)
        self.assertGreater(rejections["context_gap"], 0)
        self.assertLess(len(rows), candidates)

    def test_target_interpolation_is_bounded(self) -> None:
        times = [0.0, 20.0]
        values = [100.0, 120.0]
        result = interpolate_target(times, values, 10.0)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result[0], 110.0)
        self.assertEqual(result[1], 20.0)
        self.assertIsNone(interpolate_target([0.0, 40.0], values, 10.0))

    def test_non_monotonic_timestamps_reject_session(self) -> None:
        rows, candidates, rejections = build_session_origins(
            "Endomondo",
            "3",
            self.split,
            [0.0, 10.0, 10.0, 20.0, 700.0],
            [100.0, 101.0, 102.0, 103.0, 104.0],
        )
        self.assertEqual(rows, [])
        self.assertEqual(candidates, 0)
        self.assertEqual(rejections["non_monotonic_timestamps"], 1)

    def test_bounded_audit_selection_is_evenly_spread(self) -> None:
        rows = {str(index): {"value": str(index)} for index in range(10)}
        selected = evenly_select_mapping(rows, 3)
        self.assertEqual(list(selected), ["1", "5", "8"])
        self.assertEqual(len(selected), 3)

    def test_context_bins_use_right_closed_intervals(self) -> None:
        self.assertIsNone(context_bin_index(100.0, 100.0))
        self.assertEqual(context_bin_index(101.0, 100.0), 0)
        self.assertEqual(context_bin_index(110.0, 100.0), 0)
        self.assertEqual(context_bin_index(110.1, 100.0), 1)
        self.assertEqual(context_bin_index(400.0, 100.0), 29)


if __name__ == "__main__":
    unittest.main()
