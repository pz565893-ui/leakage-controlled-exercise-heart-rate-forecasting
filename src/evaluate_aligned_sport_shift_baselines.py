from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_VERSION = "0.12.0"
HORIZONS = (60, 180, 300)
FAMILIES = (
    "outdoor_cycling",
    "indoor_virtual_cycling",
    "running",
    "walking_hiking",
    "strength_cross_training",
)


def hierarchical_metrics(
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


def causal_baseline_predictions(
    values: np.ndarray, masks: np.ndarray, ewma_alpha: float = 0.1
) -> dict[str, np.ndarray]:
    heart_rate = np.asarray(values[:, :, 0], dtype=np.float32)
    observed = np.asarray(masks[:, :, 0], dtype=bool)
    if np.any(observed.sum(axis=1) == 0):
        raise AssertionError("an evaluation context contains no heart-rate observations")

    reversed_position = np.argmax(observed[:, ::-1], axis=1)
    last_position = observed.shape[1] - 1 - reversed_position
    persistence = heart_rate[np.arange(len(heart_rate)), last_position]

    ewma = np.zeros(len(heart_rate), dtype=np.float32)
    initialized = np.zeros(len(heart_rate), dtype=bool)
    for position in range(heart_rate.shape[1]):
        present = observed[:, position]
        first = present & ~initialized
        ewma[first] = heart_rate[first, position]
        update = present & initialized
        ewma[update] = (
            ewma_alpha * heart_rate[update, position]
            + (1.0 - ewma_alpha) * ewma[update]
        )
        initialized |= present
    ewma = np.clip(ewma, 30.0, 240.0)

    time = np.arange(-(heart_rate.shape[1] - 1) * 10, 1, 10, dtype=np.float64)
    weight = observed.astype(np.float64)
    y = heart_rate.astype(np.float64)
    n = weight.sum(axis=1)
    sum_x = (weight * time[None, :]).sum(axis=1)
    sum_y = (weight * y).sum(axis=1)
    sum_xx = (weight * (time[None, :] ** 2)).sum(axis=1)
    sum_xy = (weight * time[None, :] * y).sum(axis=1)
    denominator = n * sum_xx - sum_x**2
    slope = np.zeros(len(heart_rate), dtype=np.float64)
    valid_slope = (n >= 2) & (denominator != 0)
    slope[valid_slope] = (
        n[valid_slope] * sum_xy[valid_slope]
        - sum_x[valid_slope] * sum_y[valid_slope]
    ) / denominator[valid_slope]
    mean_x = sum_x / n
    mean_y = sum_y / n
    trend = np.stack(
        [np.clip(mean_y + slope * (horizon - mean_x), 30.0, 240.0) for horizon in HORIZONS],
        axis=1,
    ).astype(np.float32)
    trend[~valid_slope] = persistence[~valid_slope, None]
    return {
        "persistence": np.repeat(persistence[:, None], len(HORIZONS), axis=1),
        "ewma_alpha_0_1": np.repeat(ewma[:, None], len(HORIZONS), axis=1),
        "linear_trend": trend,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    values = np.load(args.array_dir / "sequence_values.npy", mmap_mode="r")
    masks = np.load(args.array_dir / "sequence_masks.npy", mmap_mode="r")
    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    users = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(args.array_dir / "session_index.npy", mmap_mode="r")

    rows: list[dict[str, object]] = []
    prediction_payload: dict[str, np.ndarray] = {}
    family_audit: dict[str, object] = {}
    for family in FAMILIES:
        with np.load(args.prediction_dir / f"sport_shift_{family}_v0_12_0.npz") as sport_file:
            row_index = sport_file["row_index"]
            same_user_count = int(sport_file["same_user_rows"][0])
        baseline = causal_baseline_predictions(values[row_index], masks[row_index])
        if any(not np.isfinite(item).all() for item in baseline.values()):
            raise AssertionError(f"{family}: non-finite baseline prediction")
        prediction_payload[f"{family}__row_index"] = row_index
        prediction_payload[f"{family}__same_user_rows"] = np.asarray([same_user_count], dtype=np.int64)
        for model_name, prediction in baseline.items():
            prediction_payload[f"{family}__{model_name}"] = prediction
        regimes = {
            f"unseen_sport__{family}": slice(0, same_user_count),
            f"joint_user_sport__{family}": slice(same_user_count, len(row_index)),
        }
        for regime, selected in regimes.items():
            selected_rows = row_index[selected]
            for model_name, prediction in baseline.items():
                selected_prediction = prediction[selected]
                for position, horizon in enumerate(HORIZONS):
                    rows.append(
                        {
                            "model_version": MODEL_VERSION,
                            "held_sport_family": family,
                            "regime": regime,
                            "model": model_name,
                            "horizon_seconds": horizon,
                            **hierarchical_metrics(
                                selected_prediction[:, position],
                                targets[selected_rows, position],
                                users[selected_rows],
                                sessions[selected_rows],
                            ),
                        }
                    )
        family_audit[family] = {
            "same_user_rows": same_user_count,
            "joint_shift_rows": len(row_index) - same_user_count,
            "prediction_rows": len(row_index),
            "finite_predictions": True,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.predictions, **prediction_payload)
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "ewma_alpha": 0.1,
        "context_seconds": 300,
        "baseline_rows": len(rows),
        "families": family_audit,
        "all_assertions_pass": True,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--prediction-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parser().parse_args())))
