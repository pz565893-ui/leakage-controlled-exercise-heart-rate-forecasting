from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_horizon_specific_expanded_arrays_v0_30_0 import (  # noqa: E402
    extra_origins,
    session_regime_flags,
)
from evaluate_horizon_specific_eligibility_v0_29_0 import SessionSpec  # noqa: E402


class HorizonSpecificExpandedArrayTests(unittest.TestCase):
    def test_extra_origins_exclude_complete_three_target_rows(self) -> None:
        timestamps = [float(value) for value in range(0, 1301, 10)]
        heart_rate = [100.0 + value / 100.0 for value in timestamps]
        rows = extra_origins(timestamps, heart_rate)
        self.assertEqual(len(rows), 1)
        origin, targets, mask = rows[0]
        self.assertEqual(origin, 1200.0)
        self.assertEqual(mask, (1, 0, 0))
        self.assertTrue(math.isfinite(targets[0]))
        self.assertTrue(math.isnan(targets[1]))
        self.assertTrue(math.isnan(targets[2]))

    def test_irregular_target_missingness_is_represented_independently(self) -> None:
        timestamps = [
            float(value)
            for value in range(0, 1001, 10)
            if value not in {650, 660, 670}
        ]
        heart_rate = [120.0 for _ in timestamps]
        rows = extra_origins(timestamps, heart_rate)
        selected = {origin: mask for origin, _, mask in rows}
        self.assertEqual(selected[600.0], (0, 1, 1))

    def test_regime_flags_preserve_overlapping_internal_session_roles(self) -> None:
        spec = SessionSpec(
            dataset="Endomondo",
            session_key="1",
            session_index=7,
            user_index=3,
            sport_code=1,
            regimes=("strict_temporal", "unseen_user"),
        )
        self.assertEqual(session_regime_flags(spec), 3)

    def test_invalid_session_emits_no_extra_rows(self) -> None:
        self.assertEqual(extra_origins([0.0, 10.0, 5.0], [100.0, 101.0, 102.0]), [])


if __name__ == "__main__":
    unittest.main()
