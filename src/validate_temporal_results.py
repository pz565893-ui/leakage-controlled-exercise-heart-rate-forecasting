from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_aligned_sport_shift_baselines import causal_baseline_predictions
from validate_sport_shift_results import (
    HORIZONS,
    QUANTILE_INTERVALS,
    assert_close,
    hierarchical_average,
    hierarchical_summary,
)


MODEL_VERSION = "0.13.0"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate(args: argparse.Namespace) -> dict[str, object]:
    audit = json.loads(args.training_audit.read_text(encoding="utf-8"))
    if not audit.get("all_assertions_pass"):
        raise AssertionError("temporal training audit failed")
    if any(audit["session_overlaps"].values()):
        raise AssertionError("session overlap in temporal training audit")
    if audit["strict_temporal_order_failures"]:
        raise AssertionError("chronological ordering failed")

    values = np.load(args.array_dir / "sequence_values.npy", mmap_mode="r")
    masks = np.load(args.array_dir / "sequence_masks.npy", mmap_mode="r")
    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    users = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(args.array_dir / "session_index.npy", mmap_mode="r")
    dataset = np.load(args.array_dir / "dataset_code.npy", mmap_mode="r")
    evaluation = np.load(args.array_dir / "evaluation_origin.npy", mmap_mode="r")
    temporal = np.load(args.temporal_partition, mmap_mode="r")
    with np.load(args.predictions) as prediction_file:
        row_index = prediction_file["row_index"]
        predictions = {
            "history_informed": prediction_file["history_quantiles"],
            "zero_history": prediction_file["zero_history_quantiles"],
        }
    if len(np.unique(row_index)) != len(row_index):
        raise AssertionError("duplicate temporal prediction rows")
    if not np.all(dataset[row_index] == 0):
        raise AssertionError("non-Endomondo temporal prediction row")
    if not np.all(evaluation[row_index] == 1):
        raise AssertionError("non-evaluation temporal prediction row")
    if not np.all(temporal[row_index] == 4):
        raise AssertionError("prediction row outside strict temporal test")

    point = pd.read_csv(args.point_metrics)
    interval = pd.read_csv(args.interval_metrics)
    thresholds = json.loads(
        (args.model_dir / "conformal_thresholds.json").read_text(encoding="utf-8")
    )
    if len(point) != 6 or len(interval) != 36:
        raise AssertionError("unexpected temporal result row count")
    max_point_delta = 0.0
    max_interval_delta = 0.0
    test_targets = np.asarray(targets[row_index], dtype=np.float32)
    test_users = np.asarray(users[row_index])
    test_sessions = np.asarray(sessions[row_index])
    for mode, prediction in predictions.items():
        if np.any(np.diff(prediction, axis=2) < -1e-6):
            raise AssertionError(f"{mode}: quantile crossing")
        for position, horizon in enumerate(HORIZONS):
            expected = hierarchical_summary(
                prediction[:, position, 3],
                test_targets[:, position],
                test_users,
                test_sessions,
            )
            stored = point[
                (point["mode"] == mode)
                & (point["horizon_seconds"] == horizon)
            ]
            if len(stored) != 1:
                raise AssertionError(f"missing point row {mode}/{horizon}")
            stored_row = stored.iloc[0]
            for metric in ("mae_bpm", "rmse_bpm", "bias_bpm"):
                delta = abs(float(stored_row[metric]) - float(expected[metric]))
                max_point_delta = max(max_point_delta, delta)
                assert_close(
                    stored_row[metric], expected[metric], f"{mode}/{horizon}/{metric}"
                )
            for metric in ("users", "sessions", "origins"):
                if int(stored_row[metric]) != int(expected[metric]):
                    raise AssertionError(f"{mode}/{horizon}/{metric}")

        for coverage, (lower_position, upper_position) in QUANTILE_INTERVALS.items():
            for calibrated in (False, True):
                for position, horizon in enumerate(HORIZONS):
                    adjustment = (
                        float(thresholds[mode][str(coverage)][position])
                        if calibrated
                        else 0.0
                    )
                    lower = np.clip(
                        prediction[:, position, lower_position] - adjustment,
                        30.0,
                        240.0,
                    )
                    upper = np.clip(
                        prediction[:, position, upper_position] + adjustment,
                        30.0,
                        240.0,
                    )
                    covered = (
                        (test_targets[:, position] >= lower)
                        & (test_targets[:, position] <= upper)
                    ).astype(np.float32)
                    width = upper - lower
                    picp, n_users, n_sessions, n_origins = hierarchical_average(
                        covered, test_users, test_sessions
                    )
                    mean_width, _, _, _ = hierarchical_average(
                        width, test_users, test_sessions
                    )
                    stored = interval[
                        (interval["mode"] == mode)
                        & (interval["horizon_seconds"] == horizon)
                        & np.isclose(interval["nominal_coverage"], coverage)
                        & (
                            interval["calibrated"].astype(str).str.lower()
                            == str(calibrated).lower()
                        )
                    ]
                    if len(stored) != 1:
                        raise AssertionError(
                            f"missing interval row {mode}/{horizon}/{coverage}/{calibrated}"
                        )
                    stored_row = stored.iloc[0]
                    expected_values = {
                        "picp": picp,
                        "absolute_coverage_error": abs(picp - coverage),
                        "mean_interval_width_bpm": mean_width,
                        "conformal_adjustment_bpm": adjustment,
                    }
                    for metric, expected_value in expected_values.items():
                        delta = abs(float(stored_row[metric]) - expected_value)
                        max_interval_delta = max(max_interval_delta, delta)
                        assert_close(
                            stored_row[metric],
                            expected_value,
                            f"{mode}/{horizon}/{coverage}/{calibrated}/{metric}",
                        )
                    if (
                        int(stored_row["users"]) != n_users
                        or int(stored_row["sessions"]) != n_sessions
                        or int(stored_row["origins"]) != n_origins
                    ):
                        raise AssertionError("interval denominator mismatch")

    baselines = causal_baseline_predictions(values[row_index], masks[row_index])
    baseline_rows: list[dict[str, object]] = []
    for model, prediction in baselines.items():
        for position, horizon in enumerate(HORIZONS):
            baseline_rows.append(
                {
                    "model_version": MODEL_VERSION,
                    "regime": "within_user_temporal_test",
                    "model": model,
                    "horizon_seconds": horizon,
                    **hierarchical_summary(
                        prediction[:, position],
                        test_targets[:, position],
                        test_users,
                        test_sessions,
                    ),
                }
            )
    write_csv(args.baseline_output, baseline_rows)
    args.baseline_predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.baseline_predictions,
        row_index=row_index,
        **baselines,
    )
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "prediction_rows": int(len(row_index)),
        "users": int(len(np.unique(test_users))),
        "sessions": int(len(np.unique(test_sessions))),
        "point_rows": int(len(point)),
        "interval_rows": int(len(interval)),
        "baseline_rows": len(baseline_rows),
        "maximum_absolute_point_recomputation_delta": max_point_delta,
        "maximum_absolute_interval_recomputation_delta": max_interval_delta,
        "checks": {
            "training_audit_pass": True,
            "session_overlaps_zero": True,
            "strict_temporal_order": True,
            "prediction_rows_unique": True,
            "prediction_rows_match_strict_test": True,
            "point_metrics_independently_recomputed": True,
            "interval_metrics_independently_recomputed": True,
            "aligned_causal_baselines_finite": all(
                np.isfinite(item).all() for item in baselines.values()
            ),
        },
        "all_assertions_pass": True,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--temporal-partition", type=Path, required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--point-metrics", type=Path, required=True)
    result.add_argument("--interval-metrics", type=Path, required=True)
    result.add_argument("--training-audit", type=Path, required=True)
    result.add_argument("--baseline-output", type=Path, required=True)
    result.add_argument("--baseline-predictions", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parser().parse_args())))
