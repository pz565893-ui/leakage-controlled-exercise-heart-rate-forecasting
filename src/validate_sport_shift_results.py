from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_VERSION = "0.12.0"
HORIZONS = (60, 180, 300)
QUANTILE_INTERVALS = {
    0.5: (2, 4),
    0.8: (1, 5),
    0.9: (0, 6),
}
FAMILIES = {
    1: "outdoor_cycling",
    2: "indoor_virtual_cycling",
    3: "running",
    4: "walking_hiking",
    7: "strength_cross_training",
}


def hierarchical_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, float | int]:
    frame = pd.DataFrame(
        {
            "user": users,
            "session": sessions,
            "error": prediction.astype(np.float64) - target.astype(np.float64),
        }
    )
    frame["absolute_error"] = frame["error"].abs()
    frame["squared_error"] = frame["error"] ** 2
    by_session = frame.groupby(["user", "session"], sort=False).agg(
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        bias=("error", "mean"),
        origins=("error", "size"),
    )
    by_session["rmse"] = np.sqrt(by_session["mse"])
    by_user = by_session.groupby(level="user", sort=False).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        bias=("bias", "mean"),
        sessions=("mae", "size"),
        origins=("origins", "sum"),
    )
    return {
        "mae_bpm": float(by_user["mae"].mean()),
        "rmse_bpm": float(by_user["rmse"].mean()),
        "bias_bpm": float(by_user["bias"].mean()),
        "users": int(len(by_user)),
        "sessions": int(by_user["sessions"].sum()),
        "origins": int(by_user["origins"].sum()),
    }


def hierarchical_average(
    value: np.ndarray, users: np.ndarray, sessions: np.ndarray
) -> tuple[float, int, int, int]:
    frame = pd.DataFrame({"value": value, "user": users, "session": sessions})
    by_session = frame.groupby(["user", "session"], sort=False)["value"].mean()
    by_user = by_session.groupby(level="user", sort=False).mean()
    return float(by_user.mean()), int(len(by_user)), int(len(by_session)), int(len(value))


def assert_close(actual: float, expected: float, label: str, tolerance: float = 5e-6) -> None:
    difference = abs(float(actual) - float(expected))
    if difference > tolerance:
        raise AssertionError(f"{label}: actual={actual}, expected={expected}, delta={difference}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate(args: argparse.Namespace) -> dict[str, object]:
    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    users_all = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions_all = np.load(args.array_dir / "session_index.npy", mmap_mode="r")
    sport_all = np.load(args.array_dir / "sport_code.npy", mmap_mode="r")
    dataset_all = np.load(args.array_dir / "dataset_code.npy", mmap_mode="r")
    evaluation_all = np.load(args.array_dir / "evaluation_origin.npy", mmap_mode="r")
    unseen_all = np.load(args.array_dir / "unseen_user_partition.npy", mmap_mode="r")

    combined_point: list[dict[str, object]] = []
    combined_interval: list[dict[str, object]] = []
    family_audits: dict[str, object] = {}
    max_point_delta = 0.0
    max_interval_delta = 0.0
    repaired_files: list[str] = []

    for held_code, family in FAMILIES.items():
        audit_path = args.audit_dir / f"sport_shift_{family}_v0_12_0.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("all_assertions_pass"):
            raise AssertionError(f"{family}: training audit failed")
        if audit["held_family_training_rows"] != 0:
            raise AssertionError(f"{family}: held family entered training")
        if any(audit["train_validation_calibration_user_overlaps"].values()):
            raise AssertionError(f"{family}: user split overlap")
        if audit["quantile_crossing_failures"] != 0:
            raise AssertionError(f"{family}: quantile crossing")

        prediction_path = args.prediction_dir / f"sport_shift_{family}_v0_12_0.npz"
        with np.load(prediction_path) as prediction_file:
            row_index = prediction_file["row_index"]
            same_user_count = int(prediction_file["same_user_rows"][0])
            quantiles_by_mode = {
                "history_informed": prediction_file["history_quantiles"],
                "zero_history": prediction_file["zero_history_quantiles"],
            }

        if len(np.unique(row_index)) != len(row_index):
            raise AssertionError(f"{family}: duplicate prediction rows")
        if not np.all(sport_all[row_index] == held_code):
            raise AssertionError(f"{family}: prediction rows contain another sport")
        if not np.all(dataset_all[row_index] == 0):
            raise AssertionError(f"{family}: non-Endomondo row in sport-shift evaluation")
        if not np.all(evaluation_all[row_index] == 1):
            raise AssertionError(f"{family}: non-evaluation origin in predictions")
        if not np.all(unseen_all[row_index[:same_user_count]] == 1):
            raise AssertionError(f"{family}: same-user block has wrong partition")
        if not np.all(unseen_all[row_index[same_user_count:]] == 4):
            raise AssertionError(f"{family}: joint-shift block has wrong partition")
        if same_user_count != audit["same_user_sport_shift_rows"]:
            raise AssertionError(f"{family}: same-user count differs from audit")
        if len(row_index) - same_user_count != audit["joint_user_sport_shift_rows"]:
            raise AssertionError(f"{family}: joint count differs from audit")

        point_path = args.result_dir / f"sport_shift_{family}_point_v0_12_0.csv"
        point = pd.read_csv(point_path)
        if set(point["model_version"].astype(str)) != {MODEL_VERSION}:
            raise AssertionError(f"{family}: point-result version mismatch")
        if len(point) != 12:
            raise AssertionError(f"{family}: expected 12 point rows")

        interval_path = args.result_dir / f"sport_shift_{family}_interval_v0_12_0.csv"
        interval = pd.read_csv(interval_path)
        versions = set(interval["model_version"].astype(str))
        if versions != {MODEL_VERSION}:
            interval["model_version"] = MODEL_VERSION
            repaired_files.append(str(interval_path))
            write_csv(interval_path, interval.to_dict(orient="records"))
        if len(interval) != 72:
            raise AssertionError(f"{family}: expected 72 interval rows")

        thresholds = json.loads(
            (args.model_dir / family / "conformal_thresholds.json").read_text(
                encoding="utf-8"
            )
        )
        regimes = {
            f"unseen_sport__{family}": slice(0, same_user_count),
            f"joint_user_sport__{family}": slice(same_user_count, len(row_index)),
        }
        for regime, selected in regimes.items():
            selected_rows = row_index[selected]
            selected_targets = np.asarray(targets[selected_rows], dtype=np.float32)
            selected_users = np.asarray(users_all[selected_rows])
            selected_sessions = np.asarray(sessions_all[selected_rows])
            for mode, all_predictions in quantiles_by_mode.items():
                prediction = np.asarray(all_predictions[selected], dtype=np.float32)
                for position, horizon in enumerate(HORIZONS):
                    expected = hierarchical_summary(
                        prediction[:, position, 3],
                        selected_targets[:, position],
                        selected_users,
                        selected_sessions,
                    )
                    stored = point[
                        (point["regime"] == regime)
                        & (point["mode"] == mode)
                        & (point["horizon_seconds"] == horizon)
                    ]
                    if len(stored) != 1:
                        raise AssertionError(f"{family}: missing point row {regime}/{mode}/{horizon}")
                    stored_row = stored.iloc[0]
                    for metric in ("mae_bpm", "rmse_bpm", "bias_bpm"):
                        delta = abs(float(stored_row[metric]) - float(expected[metric]))
                        max_point_delta = max(max_point_delta, delta)
                        assert_close(stored_row[metric], expected[metric], f"{family}/{regime}/{mode}/{horizon}/{metric}")
                    for metric in ("users", "sessions", "origins"):
                        if int(stored_row[metric]) != int(expected[metric]):
                            raise AssertionError(f"{family}/{regime}/{mode}/{horizon}/{metric}")

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
                                (selected_targets[:, position] >= lower)
                                & (selected_targets[:, position] <= upper)
                            ).astype(np.float32)
                            width = upper - lower
                            picp, n_users, n_sessions, n_origins = hierarchical_average(
                                covered, selected_users, selected_sessions
                            )
                            mean_width, _, _, _ = hierarchical_average(
                                width, selected_users, selected_sessions
                            )
                            stored = interval[
                                (interval["regime"] == regime)
                                & (interval["mode"] == mode)
                                & (interval["horizon_seconds"] == horizon)
                                & np.isclose(interval["nominal_coverage"], coverage)
                                & (interval["calibrated"].astype(str).str.lower() == str(calibrated).lower())
                            ]
                            if len(stored) != 1:
                                raise AssertionError(
                                    f"{family}: missing interval row {regime}/{mode}/{horizon}/{coverage}/{calibrated}"
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
                                    f"{family}/{regime}/{mode}/{horizon}/{coverage}/{calibrated}/{metric}",
                                )
                            for metric, expected_value in {
                                "users": n_users,
                                "sessions": n_sessions,
                                "origins": n_origins,
                            }.items():
                                if int(stored_row[metric]) != expected_value:
                                    raise AssertionError(
                                        f"{family}/{regime}/{mode}/{horizon}/{coverage}/{calibrated}/{metric}"
                                    )

        combined_point.extend(point.to_dict(orient="records"))
        combined_interval.extend(interval.to_dict(orient="records"))
        family_audits[family] = {
            "held_sport_code": held_code,
            "training_rows": audit["training_rows"],
            "same_user_sport_shift_rows": audit["same_user_sport_shift_rows"],
            "joint_user_sport_shift_rows": audit["joint_user_sport_shift_rows"],
            "best_epoch": audit["best_epoch"],
            "all_assertions_pass": True,
        }

    write_csv(args.point_output, combined_point)
    write_csv(args.interval_output, combined_interval)
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "families": family_audits,
        "point_rows": len(combined_point),
        "interval_rows": len(combined_interval),
        "maximum_absolute_point_recomputation_delta": max_point_delta,
        "maximum_absolute_interval_recomputation_delta": max_interval_delta,
        "source_metadata_files_repaired": repaired_files,
        "checks": {
            "training_audits_pass": True,
            "held_sports_absent_from_training": True,
            "user_split_overlaps_zero": True,
            "prediction_rows_unique": True,
            "prediction_rows_match_held_sport": True,
            "prediction_partitions_match_regime": True,
            "point_metrics_independently_recomputed": True,
            "interval_metrics_independently_recomputed": True,
        },
        "all_assertions_pass": True,
    }
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--result-dir", type=Path, required=True)
    result.add_argument("--prediction-dir", type=Path, required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--audit-dir", type=Path, required=True)
    result.add_argument("--point-output", type=Path, required=True)
    result.add_argument("--interval-output", type=Path, required=True)
    result.add_argument("--output-audit", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parser().parse_args())))
