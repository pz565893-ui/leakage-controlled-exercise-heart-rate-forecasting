from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_causal_history import Accumulator, build  # noqa: E402


def row(start: float, sport: int, hr: float = 120.0) -> pd.Series:
    return pd.Series(
        {
            "duration_seconds": 1000.0,
            "hr_mean": hr,
            "hr_std": 10.0,
            "speed_mean": 20.0,
            "altitude_std": 5.0,
            "sport_code": sport,
            "session_start_time": start,
            "session_end_time": start + 1000.0,
        }
    )


class CausalHistoryTests(unittest.TestCase):
    def test_empty_history_is_zero(self) -> None:
        accumulator = Accumulator()
        self.assertTrue((accumulator.features(1, 100.0) == 0).all())

    def test_history_uses_only_prior_updates(self) -> None:
        accumulator = Accumulator()
        accumulator.update(row(100.0, 1, 110.0))
        features = accumulator.features(1, 200.0)
        self.assertEqual(features[3], 110.0)
        self.assertEqual(features[10], 110.0)
        self.assertEqual(accumulator.count, 1)

    def test_excluded_sport_can_be_skipped_by_caller(self) -> None:
        accumulator = Accumulator()
        excluded = row(100.0, 3, 160.0)
        if int(excluded["sport_code"]) != 3:
            accumulator.update(excluded)
        self.assertEqual(accumulator.count, 0)

    def test_overlapping_session_is_not_available_until_completed(self) -> None:
        sessions = pd.DataFrame(
            [
                {
                    "session_index": 0,
                    "dataset": "Endomondo",
                    "session_key": "a",
                    "user_index": 0,
                    "sport_code": 1,
                    "session_start_time": 100.0,
                    "session_end_time": 300.0,
                    "duration_seconds": 200.0,
                    "hr_mean": 110.0,
                    "hr_std": 5.0,
                    "speed_mean": 20.0,
                    "altitude_std": 3.0,
                },
                {
                    "session_index": 1,
                    "dataset": "Endomondo",
                    "session_key": "b",
                    "user_index": 0,
                    "sport_code": 1,
                    "session_start_time": 200.0,
                    "session_end_time": 250.0,
                    "duration_seconds": 50.0,
                    "hr_mean": 150.0,
                    "hr_std": 5.0,
                    "speed_mean": 20.0,
                    "altitude_std": 3.0,
                },
                {
                    "session_index": 2,
                    "dataset": "Endomondo",
                    "session_key": "c",
                    "user_index": 0,
                    "sport_code": 1,
                    "session_start_time": 400.0,
                    "session_end_time": 500.0,
                    "duration_seconds": 100.0,
                    "hr_mean": 130.0,
                    "hr_std": 5.0,
                    "speed_mean": 20.0,
                    "altitude_std": 3.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "sessions.csv"
            sessions.to_csv(manifest, index=False)
            payload = build(manifest, root / "history")
            prior = np.load(root / "history" / "session_prior_count.npy")
        self.assertEqual(int(prior[0]), 0)
        self.assertEqual(int(prior[1]), 0)
        self.assertEqual(int(prior[2]), 2)
        self.assertEqual(payload["overlap_guarded_current_sessions"], 1)


if __name__ == "__main__":
    unittest.main()
