from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_multiseed_balanced_calibration import (  # noqa: E402
    Hierarchy,
    bootstrap_method_difference,
    inspect_strict_temporal_artifacts,
    origin_pooled_finite_sample_threshold,
    summarize_median_min_max,
    weighted_quantile_higher,
)


class MultiseedBalancedCalibrationTests(unittest.TestCase):
    def test_hierarchy_equalizes_sessions_within_user(self) -> None:
        users = np.array([1, 1, 1, 2, 2, 2])
        sessions = np.array([10, 10, 11, 20, 20, 20])
        values = np.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
        hierarchy = Hierarchy.build(users, sessions)

        np.testing.assert_allclose(
            hierarchy.per_user_mean(values), np.array([3.5, 4.0])
        )
        self.assertAlmostEqual(float(hierarchy.global_origin_weights.sum()), 1.0)
        self.assertAlmostEqual(
            float(hierarchy.global_origin_weights[users == 1].sum()), 0.5
        )
        self.assertAlmostEqual(
            float(hierarchy.global_origin_weights[users == 2].sum()), 0.5
        )

    def test_weighted_and_origin_pooled_thresholds_are_distinct_estimands(self) -> None:
        scores = np.array([0.0, 1.0, 2.0, 3.0])
        weights = np.array([0.05, 0.05, 0.10, 0.80])
        self.assertEqual(weighted_quantile_higher(scores, weights, 0.25), 3.0)
        self.assertEqual(origin_pooled_finite_sample_threshold(scores, 0.25), 2.0)

    def test_paired_user_bootstrap_preserves_constant_method_difference(self) -> None:
        origin_coverage = np.array([0.4, 0.5, 0.6])
        balanced_coverage = origin_coverage + 0.1
        origin_width = np.array([10.0, 11.0, 12.0])
        balanced_width = origin_width + 2.0
        result = bootstrap_method_difference(
            origin_coverage,
            balanced_coverage,
            origin_width,
            balanced_width,
            nominal_coverage=0.8,
            replicates=1_000,
            seed=7,
        )
        self.assertAlmostEqual(result["delta_picp"], 0.1)
        self.assertAlmostEqual(result["delta_picp_ci_low"], 0.1)
        self.assertAlmostEqual(result["delta_picp_ci_high"], 0.1)
        self.assertAlmostEqual(result["delta_absolute_coverage_error"], -0.1)
        self.assertAlmostEqual(result["delta_mean_interval_width_bpm"], 2.0)

    def test_summary_emits_median_min_max(self) -> None:
        rows = [
            {"regime": "r", "mode": "m", "value": value}
            for value in (1.0, 3.0, 2.0)
        ]
        summary = summarize_median_min_max(
            rows, ("regime", "mode"), ("value",)
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["seed_count"], 3)
        self.assertEqual(summary[0]["value_median"], 2.0)
        self.assertEqual(summary[0]["value_min"], 1.0)
        self.assertEqual(summary[0]["value_max"], 3.0)

    def test_strict_temporal_inspection_proves_missing_calibration_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "conformal_thresholds.json").write_text(
                json.dumps({"history_informed": {}}), encoding="utf-8"
            )
            np.savez_compressed(
                root / "predictions.npz",
                row_index=np.array([3, 4, 5]),
                history_quantiles=np.zeros((3, 3, 7), dtype=np.float32),
            )
            partition = np.array([1, 2, 3, 4, 4, 4], dtype=np.uint8)

            result = inspect_strict_temporal_artifacts(root, partition)

        self.assertEqual(result["prediction_archive_count"], 1)
        self.assertEqual(result["persisted_calibration_prediction_rows"], 0)
        self.assertEqual(
            result["balanced_recalibration_status"],
            "unavailable_from_persisted_predictions",
        )
        self.assertEqual(
            result["prediction_archives"][0]["strict_temporal_partition_counts"],
            {"4": 3},
        )


if __name__ == "__main__":
    unittest.main()
