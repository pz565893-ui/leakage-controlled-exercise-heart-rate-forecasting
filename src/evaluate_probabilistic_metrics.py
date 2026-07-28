from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from train_uncertainty_model import INTERVALS, MODEL_VERSION, QUANTILES
from train_xgboost_baseline import EXTERNAL_FROZEN, HORIZONS, PARTITION_TEST, PARTITION_VALIDATION


def hierarchical_average(values: np.ndarray, users: np.ndarray, sessions: np.ndarray) -> float:
    frame = pd.DataFrame({"value": values, "user": users, "session": sessions})
    session = frame.groupby(["user", "session"], sort=False)["value"].mean()
    user = session.groupby(level="user", sort=False).mean()
    return float(user.mean())


def mean_user_spearman(
    width: np.ndarray, absolute_error: np.ndarray, users: np.ndarray
) -> tuple[float, int]:
    correlations = []
    for user in np.unique(users):
        selected = users == user
        if selected.sum() < 3:
            continue
        if np.ptp(width[selected]) == 0 or np.ptp(absolute_error[selected]) == 0:
            continue
        correlation = spearmanr(width[selected], absolute_error[selected]).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
    return (
        float(np.mean(correlations)) if correlations else float("nan"),
        len(correlations),
    )


def pinball_values(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = target[:, None] - prediction
    return np.maximum(QUANTILES[None, :] * error, (QUANTILES[None, :] - 1) * error).mean(axis=1)


def calibrated_bounds(
    prediction: np.ndarray,
    thresholds: dict[str, list[float]],
    horizon_position: int,
    calibrated: bool,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for coverage, (lower_position, upper_position) in INTERVALS.items():
        adjustment = thresholds[str(coverage)][horizon_position] if calibrated else 0.0
        result[coverage] = (
            np.clip(prediction[:, lower_position] - adjustment, 30.0, 240.0),
            np.clip(prediction[:, upper_position] + adjustment, 30.0, 240.0),
        )
    return result


def weighted_interval_score(
    median: np.ndarray,
    target: np.ndarray,
    bounds: dict[float, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    total = 0.5 * np.abs(target - median)
    for coverage, (lower, upper) in bounds.items():
        alpha = 1.0 - coverage
        interval_score = (
            upper
            - lower
            + (2.0 / alpha) * (lower - target) * (target < lower)
            + (2.0 / alpha) * (target - upper) * (target > upper)
        )
        total = total + (alpha / 2.0) * interval_score
    return total / (len(bounds) + 0.5)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    arrays = args.array_dir
    targets = np.load(arrays / "targets.npy", mmap_mode="r")
    dataset = np.load(arrays / "dataset_code.npy", mmap_mode="r")
    unseen = np.load(arrays / "unseen_user_partition.npy", mmap_mode="r")
    external = np.load(arrays / "primary_external_partition.npy", mmap_mode="r")
    sport = np.load(arrays / "sport_code.npy", mmap_mode="r")
    users = np.load(arrays / "user_index.npy", mmap_mode="r")
    sessions = np.load(arrays / "session_index.npy", mmap_mode="r")
    predictions = np.load(args.predictions)
    row_index = predictions["row_index"]
    history_prediction = predictions["history_quantiles"]
    zero_prediction = predictions["zero_history_quantiles"]
    threshold_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
    thresholds = threshold_payload["thresholds"]
    regimes: dict[str, np.ndarray] = {
        "unseen_user_validation": (dataset[row_index] == 0)
        & (unseen[row_index] == PARTITION_VALIDATION),
        "unseen_user_test": (dataset[row_index] == 0)
        & (unseen[row_index] == PARTITION_TEST),
        "goldencheetah_frozen_external": (dataset[row_index] == 1)
        & (external[row_index] == EXTERNAL_FROZEN),
    }
    for code, family in ((1, "outdoor_cycling"), (2, "indoor_virtual_cycling"), (3, "running")):
        regimes[f"goldencheetah_external__{family}"] = (
            (dataset[row_index] == 1)
            & (external[row_index] == EXTERNAL_FROZEN)
            & (sport[row_index] == code)
        )
    rows: list[dict[str, object]] = []
    for regime, regime_mask in regimes.items():
        selected_rows = row_index[regime_mask]
        modes = {"zero_history": zero_prediction[regime_mask]}
        if not regime.startswith("goldencheetah"):
            modes["history_informed"] = history_prediction[regime_mask]
        for mode, prediction in modes.items():
            for calibrated in (False, True):
                for horizon_position, horizon in enumerate(HORIZONS):
                    target = np.asarray(targets[selected_rows, horizon_position], dtype=np.float32)
                    median = prediction[:, horizon_position, 3]
                    bounds = calibrated_bounds(
                        prediction[:, horizon_position],
                        thresholds[mode],
                        horizon_position,
                        calibrated,
                    )
                    wis = weighted_interval_score(median, target, bounds)
                    pinball = pinball_values(prediction[:, horizon_position], target)
                    width_90 = bounds[0.90][1] - bounds[0.90][0]
                    absolute_error = np.abs(median - target)
                    correlation, correlation_users = mean_user_spearman(
                        width_90,
                        absolute_error,
                        np.asarray(users[selected_rows]),
                    )
                    rows.append(
                        {
                            "model_version": MODEL_VERSION,
                            "regime": regime,
                            "mode": mode,
                            "horizon_seconds": horizon,
                            "calibrated": calibrated,
                            "pinball_loss": hierarchical_average(
                                pinball,
                                np.asarray(users[selected_rows]),
                                np.asarray(sessions[selected_rows]),
                            ),
                            "weighted_interval_score": hierarchical_average(
                                wis,
                                np.asarray(users[selected_rows]),
                                np.asarray(sessions[selected_rows]),
                            ),
                            "normalized_90_interval_width": hierarchical_average(
                                width_90 / np.maximum(target, 1.0),
                                np.asarray(users[selected_rows]),
                                np.asarray(sessions[selected_rows]),
                            ),
                            "mean_user_spearman_width_absolute_error": correlation,
                            "users_with_defined_spearman": correlation_users,
                            "users": int(len(np.unique(users[selected_rows]))),
                            "sessions": int(len(np.unique(sessions[selected_rows]))),
                            "origins": int(len(selected_rows)),
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = np.asarray(
        [
            [
                row["pinball_loss"],
                row["weighted_interval_score"],
                row["normalized_90_interval_width"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "prediction_rows": int(len(row_index)),
        "metric_rows": len(rows),
        "regimes": sorted(regimes),
        "nonfinite_core_metrics": int((~np.isfinite(numeric)).sum()),
        "output": str(args.output),
    }
    payload["all_assertions_pass"] = payload["nonfinite_core_metrics"] == 0
    atomic_json = args.audit.with_suffix(args.audit.suffix + ".tmp")
    atomic_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    atomic_json.replace(args.audit)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate probabilistic forecast metrics.")
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
