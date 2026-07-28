from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_probabilistic_metrics import weighted_interval_score  # noqa: E402


class ProbabilisticEvaluationTests(unittest.TestCase):
    def test_wis_rewards_exact_narrow_forecast(self) -> None:
        target = np.array([100.0])
        exact = weighted_interval_score(
            target,
            target,
            {0.5: (target, target), 0.8: (target, target), 0.9: (target, target)},
        )
        self.assertEqual(float(exact[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
