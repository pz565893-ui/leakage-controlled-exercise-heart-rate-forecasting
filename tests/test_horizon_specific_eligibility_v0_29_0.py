from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_horizon_specific_eligibility_v0_29_0 import (  # noqa: E402
    build_session_errors,
    candidate_origins,
    context_is_eligible,
    first_evaluation_time,
)


class HorizonSpecificEligibilityTests(unittest.TestCase):
    def test_evaluation_alignment_is_absolute_and_five_minutely(self) -> None:
        self.assertEqual(first_evaluation_time(301.0), 600)
        self.assertEqual(list(candidate_origins([1.0, 1_001.0])), [600, 900])

    def test_context_rule_matches_thirty_ten_second_bins(self) -> None:
        times = [float(value) for value in range(0, 301, 10)]
        self.assertTrue(context_is_eligible(times, 300.0))
        sparse = [float(value) for value in range(0, 301, 20)]
        self.assertFalse(context_is_eligible(sparse, 300.0))

    def test_shorter_horizons_gain_only_the_last_origin(self) -> None:
        times = [float(value) for value in range(0, 801, 10)]
        heart_rate = [100.0 + value / 100.0 for value in times]

        def persistence(_: float) -> float:
            return 100.0

        errors = build_session_errors(times, heart_rate, persistence)
        self.assertEqual(len(errors[(60, "common_three_target")]), 1)
        self.assertEqual(len(errors[(60, "horizon_specific")]), 2)
        self.assertEqual(len(errors[(180, "common_three_target")]), 1)
        self.assertEqual(len(errors[(180, "horizon_specific")]), 2)
        self.assertEqual(len(errors[(300, "common_three_target")]), 1)
        self.assertEqual(len(errors[(300, "horizon_specific")]), 1)
        self.assertEqual(errors[(300, "common_three_target")], errors[(300, "horizon_specific")])

    def test_nonmonotonic_session_is_rejected(self) -> None:
        errors = build_session_errors([0.0, 10.0, 5.0], [100.0, 101.0, 102.0], lambda _: 100.0)
        self.assertEqual(errors, {})


if __name__ == "__main__":
    unittest.main()
