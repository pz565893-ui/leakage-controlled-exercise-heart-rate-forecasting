from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aggregate_q1_multiseed_results import (  # noqa: E402
    aggregate,
    build_expected_jobs,
    default_layout,
    inspect_job,
    load_layout,
    main_history_pairs,
    output_paths,
    sha256_file,
    summarize_seed_variability,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point_rows(model: str, regimes: list[str], modes: list[str] | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for regime in regimes:
        for mode in modes or [None]:
            for horizon in [60, 180, 300]:
                row: dict[str, object] = {
                    "regime": regime,
                    "model": model,
                    "horizon_seconds": horizon,
                    "mae_bpm": 5.0 + horizon / 100.0 + (0.1 if mode == "history_informed" else 0.0),
                    "rmse_bpm": 6.0 + horizon / 100.0,
                    "bias_bpm": -0.2,
                    "users": 10,
                    "sessions": 100,
                    "origins": 1000,
                }
                if mode is not None:
                    row["mode"] = mode
                rows.append(row)
    return rows


def interval_rows(model: str, regime: str, mode: str) -> list[dict[str, object]]:
    return [
        {
            "regime": regime,
            "model": model,
            "mode": mode,
            "horizon_seconds": horizon,
            "nominal_coverage": 0.9,
            "calibrated": True,
            "picp": 0.89,
            "absolute_coverage_error": 0.01,
            "mean_interval_width_bpm": 20.0,
            "conformal_adjustment_bpm": 1.0,
            "users": 10,
            "sessions": 100,
            "origins": 1000,
        }
        for horizon in [60, 180, 300]
    ]


class Q1MultiSeedAggregationTests(unittest.TestCase):
    def test_default_layout_expands_to_37_jobs(self) -> None:
        config = {
            "seeds": {
                "primary_models": [1, 2, 3, 4, 5],
                "learned_comparators": [1, 2, 3],
                "held_sport_models": [1, 2, 3],
            },
            "held_sport_main": {
                "sport_families": ["a", "b", "c", "d", "e"]
            },
        }
        jobs = build_expected_jobs(config, default_layout(), Path("root"))
        self.assertEqual(len(jobs), 37)
        self.assertEqual(len({job.job_id for job in jobs}), 37)

    def test_partial_fixture_reports_pending_and_incomplete(self) -> None:
        config = {
            "seeds": {
                "primary_models": [11],
                "learned_comparators": [11],
                "held_sport_models": [11],
            },
            "held_sport_main": {"sport_families": ["running"]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs = build_expected_jobs(config, default_layout(), root)
            unseen = next(job for job in jobs if job.experiment == "unseen_main")
            unseen.directory.mkdir(parents=True)
            (unseen.directory / "development_point_metrics.csv").write_text(
                "regime,horizon_seconds,users,sessions,origins,mae_bpm,rmse_bpm,bias_bpm\n",
                encoding="utf-8",
            )
            record, rows = inspect_job(unseen, root)
            self.assertEqual(record["status"], "incomplete")
            self.assertFalse(rows)
            temporal = next(job for job in jobs if job.experiment == "temporal_gru")
            record, _ = inspect_job(temporal, root)
            self.assertEqual(record["status"], "pending")

    def test_history_pair_summary_uses_median_minimum_maximum_only(self) -> None:
        rows: list[dict[str, object]] = []
        for seed, history, zero in [(1, 8.0, 8.2), (2, 8.3, 8.1), (3, 8.1, 8.2)]:
            for mode, value in [("history_informed", history), ("zero_history", zero)]:
                rows.append(
                    {
                        "seed": seed,
                        "experiment": "temporal_main",
                        "family": "",
                        "phase": "evaluation",
                        "source_kind": "point",
                        "model": "history_quantile_tcn",
                        "source_model_version": "test",
                        "source_analysis_version": "test",
                        "protocol": "temporal_main",
                        "regime": "within_user_temporal_test",
                        "mode": mode,
                        "horizon_seconds": 300,
                        "nominal_coverage": "",
                        "calibrated": "",
                        "metric": "mae_bpm",
                        "value": value,
                        "users": 10,
                        "sessions": 100,
                        "origins": 1000,
                        "aggregation": "hierarchical",
                    }
                )
        per_seed, summary = main_history_pairs(rows)
        self.assertEqual(len(per_seed), 3)
        self.assertAlmostEqual(summary[0]["difference_median_bpm"], -0.1)
        self.assertFalse(summary[0]["seed_inferential_test"])
        self.assertNotIn("p_value", summary[0])
        variability = summarize_seed_variability(rows)
        self.assertTrue(all("value_median" in row for row in variability))

    def test_complete_files_with_mismatched_audit_seed_are_invalid(self) -> None:
        config = {
            "seeds": {
                "primary_models": [11],
                "learned_comparators": [11],
                "held_sport_models": [11],
            },
            "held_sport_main": {"sport_families": ["running"]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = next(
                value
                for value in build_expected_jobs(config, default_layout(), root)
                if value.experiment == "temporal_gru"
            )
            rows = point_rows("gru", ["within_user_temporal_test"], None)
            write_csv(job.directory / "point_metrics.csv", rows)
            write_json(
                job.directory / "audit.json",
                {
                    "seed": 99,
                    "model": "gru",
                    "metric_rows": len(rows),
                    "all_assertions_pass": True,
                },
            )
            record, normalized = inspect_job(job, root)
            self.assertEqual(record["status"], "invalid")
            self.assertFalse(normalized)
            self.assertTrue(any("seed mismatch" in error for error in record["errors"]))

    def test_complete_synthetic_fixture_emits_strict_final_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "runs"
            output = root / "aggregation"
            config_path = base / "config.json"
            layout_path = base / "layout.json"
            config = {
                "analysis_version": "0.21.0",
                "seeds": {
                    "primary_models": [7],
                    "learned_comparators": [7],
                    "held_sport_models": [7],
                },
                "held_sport_main": {"sport_families": ["running"]},
            }
            write_json(config_path, config)
            enabled = {"unseen_main", "unseen_gru", "unseen_tcn"}
            overrides = {
                name: {"enabled": name in enabled}
                for name in default_layout()
            }
            write_json(layout_path, {"experiments": overrides})
            main_dir = root / "seed_7" / "unseen_main"
            development_point = point_rows(
                "history_quantile_tcn",
                ["unseen_user_validation", "unseen_user_test"],
                ["history_informed", "zero_history"],
            )
            development_interval = interval_rows(
                "history_quantile_tcn", "unseen_user_test", "zero_history"
            )
            external_point = point_rows(
                "history_quantile_tcn",
                ["goldencheetah_frozen_external"],
                ["zero_history"],
            )
            external_interval = interval_rows(
                "history_quantile_tcn",
                "goldencheetah_frozen_external",
                "zero_history",
            )
            write_csv(main_dir / "development_point_metrics.csv", development_point)
            write_csv(main_dir / "development_interval_metrics.csv", development_interval)
            write_csv(main_dir / "external_point_metrics.csv", external_point)
            write_csv(main_dir / "external_interval_metrics.csv", external_interval)
            development_audit = {
                "seed": 7,
                "model": "history_quantile_tcn",
                "development_only": True,
                "external_inference_performed": False,
                "point_metric_rows": len(development_point),
                "uncertainty_metric_rows": len(development_interval),
                "all_assertions_pass": True,
            }
            write_json(main_dir / "development_audit.json", development_audit)
            freeze = {
                "seed": 7,
                "status": "frozen_before_external_inference",
                "external_outcomes_used_for_selection": False,
                "external_adaptation_or_recalibration_allowed": False,
                "development_audit": {
                    "sha256": sha256_file(main_dir / "development_audit.json")
                },
            }
            write_json(main_dir / "freeze_record.json", freeze)
            external_audit = {
                "seed": 7,
                "model": "history_quantile_tcn",
                "external_adaptation_or_recalibration": False,
                "freeze_record_sha256": sha256_file(main_dir / "freeze_record.json"),
                "outputs": {
                    "point_metrics_sha256": sha256_file(
                        main_dir / "external_point_metrics.csv"
                    ),
                    "interval_metrics_sha256": sha256_file(
                        main_dir / "external_interval_metrics.csv"
                    ),
                },
                "all_assertions_pass": True,
            }
            write_json(main_dir / "external_audit.json", external_audit)
            for experiment, model in [("unseen_gru", "gru"), ("unseen_tcn", "tcn")]:
                directory = root / "seed_7" / experiment
                rows = point_rows(
                    model,
                    [
                        "unseen_user_validation",
                        "unseen_user_test",
                        "goldencheetah_frozen_external",
                    ],
                    None,
                )
                write_csv(directory / "point_metrics.csv", rows)
                write_json(
                    directory / "audit.json",
                    {
                        "seed": 7,
                        "model": model,
                        "metric_rows": len(rows),
                        "all_assertions_pass": True,
                    },
                )
            args = argparse.Namespace(
                root=root,
                config=config_path,
                layout_config=layout_path,
                output_dir=output,
            )
            progress = aggregate(args)
            self.assertEqual(progress["status"], "complete")
            self.assertTrue(progress["final_artifacts_emitted"])
            paths = output_paths(output)
            for key, path in paths.items():
                self.assertTrue(path.exists(), key)
            audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
            self.assertTrue(audit["all_assertions_pass"])
            self.assertFalse(
                audit["statistical_policy"]["seeds_treated_as_independent_participants"]
            )
            comparator = pd.read_csv(paths["comparator_summary"])
            self.assertEqual(set(comparator["n_seed_pairs"]), {1})
            self.assertNotIn("p_value", comparator.columns)


if __name__ == "__main__":
    unittest.main()
