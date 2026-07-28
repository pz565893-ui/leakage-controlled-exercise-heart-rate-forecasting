from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_horizon_specific_frozen_models_v0_30_0 import (  # noqa: E402
    bootstrap_interval,
    model_paths,
    per_user_mae,
)


class HorizonSpecificFrozenModelTests(unittest.TestCase):
    def test_per_user_mae_equalizes_sessions_before_users(self) -> None:
        prediction = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target = np.asarray([1.0, 1.0, 9.0, 5.0], dtype=np.float32)
        users = np.asarray([0, 0, 0, 1], dtype=np.int32)
        sessions = np.asarray([0, 0, 1, 2], dtype=np.int32)
        result = per_user_mae(prediction, target, users, sessions)
        self.assertAlmostEqual(float(result.loc[0]), 5.0)
        self.assertAlmostEqual(float(result.loc[1]), 5.0)

    def test_bootstrap_is_deterministic_for_constant_effect(self) -> None:
        values = np.full(25, 0.125, dtype=np.float64)
        lower, upper = bootstrap_interval(values, np.random.default_rng(7))
        self.assertAlmostEqual(lower, 0.125)
        self.assertAlmostEqual(upper, 0.125)

    def test_model_paths_keep_protocol_artifacts_separate(self) -> None:
        temporal = model_paths(Path("root"), 20260722, "temporal")
        unseen = model_paths(Path("root"), 20260722, "unseen")
        self.assertIn("temporal_main", str(temporal["checkpoint"]))
        self.assertIn("unseen_main", str(unseen["checkpoint"]))
        self.assertIn("development_predictions", unseen)
        self.assertIn("external_predictions", unseen)

    def test_unknown_protocol_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            model_paths(Path("root"), 20260722, "other")


if __name__ == "__main__":
    unittest.main()
