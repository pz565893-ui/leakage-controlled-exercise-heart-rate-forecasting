from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_probabilistic_metrics import (
    calibrated_bounds,
    hierarchical_average,
    mean_user_spearman,
    pinball_values,
    weighted_interval_score,
)
from train_xgboost_baseline import HORIZONS


MODEL_VERSION = "0.13.0"


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    users = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(args.array_dir / "session_index.npy", mmap_mode="r")
    with np.load(args.predictions) as prediction_file:
        row_index = prediction_file["row_index"]
        modes = {
            "history_informed": prediction_file["history_quantiles"],
            "zero_history": prediction_file["zero_history_quantiles"],
        }
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for mode, prediction in modes.items():
        for calibrated in (False, True):
            for position, horizon in enumerate(HORIZONS):
                target = np.asarray(targets[row_index, position], dtype=np.float32)
                median = prediction[:, position, 3]
                bounds = calibrated_bounds(
                    prediction[:, position], thresholds[mode], position, calibrated
                )
                pinball = pinball_values(prediction[:, position], target)
                wis = weighted_interval_score(median, target, bounds)
                width_90 = bounds[0.9][1] - bounds[0.9][0]
                correlation, correlation_users = mean_user_spearman(
                    width_90,
                    np.abs(median - target),
                    np.asarray(users[row_index]),
                )
                rows.append(
                    {
                        "model_version": MODEL_VERSION,
                        "regime": "within_user_temporal_test",
                        "mode": mode,
                        "horizon_seconds": horizon,
                        "calibrated": calibrated,
                        "pinball_loss": hierarchical_average(
                            pinball,
                            np.asarray(users[row_index]),
                            np.asarray(sessions[row_index]),
                        ),
                        "weighted_interval_score": hierarchical_average(
                            wis,
                            np.asarray(users[row_index]),
                            np.asarray(sessions[row_index]),
                        ),
                        "normalized_90_interval_width": hierarchical_average(
                            width_90 / np.maximum(target, 1.0),
                            np.asarray(users[row_index]),
                            np.asarray(sessions[row_index]),
                        ),
                        "mean_user_spearman_width_absolute_error": correlation,
                        "users_with_defined_spearman": correlation_users,
                        "users": int(len(np.unique(users[row_index]))),
                        "sessions": int(len(np.unique(sessions[row_index]))),
                        "origins": int(len(row_index)),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    core = np.asarray(
        [
            [
                row["pinball_loss"],
                row["weighted_interval_score"],
                row["normalized_90_interval_width"],
            ]
            for row in rows
        ]
    )
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "prediction_rows": int(len(row_index)),
        "metric_rows": len(rows),
        "nonfinite_core_metrics": int((~np.isfinite(core)).sum()),
        "all_assertions_pass": bool(np.isfinite(core).all() and len(rows) == 12),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--thresholds", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parser().parse_args())))
