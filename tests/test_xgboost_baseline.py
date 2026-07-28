from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_xgboost_baseline import (  # noqa: E402
    hierarchical_metrics,
    hierarchical_training_weights,
)


class XGBoostBaselineTests(unittest.TestCase):
    def test_training_weights_equalize_users_and_sessions(self) -> None:
        users = np.array([0, 0, 0, 1, 1])
        sessions = np.array([0, 0, 1, 2, 2])
        weights = hierarchical_training_weights(users, sessions)
        self.assertAlmostEqual(weights[users == 0].sum(), weights[users == 1].sum())
        self.assertAlmostEqual(weights[sessions == 0].sum(), weights[sessions == 1].sum())

    def test_hierarchical_metric_does_not_overweight_long_session(self) -> None:
        predictions = np.array([0.0, 0.0, 0.0, 10.0])
        targets = np.zeros(4)
        users = np.array([0, 0, 0, 0])
        sessions = np.array([0, 0, 0, 1])
        metrics = hierarchical_metrics(predictions, targets, users, sessions)
        self.assertEqual(metrics["mae_bpm"], 5.0)
        self.assertEqual(metrics["sessions"], 2)
        self.assertEqual(metrics["origins"], 4)


if __name__ == "__main__":
    unittest.main()
