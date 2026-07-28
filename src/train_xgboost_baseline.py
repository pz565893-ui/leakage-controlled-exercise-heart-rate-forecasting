from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


MODEL_VERSION = "0.8.0"
SEED = 20260722
HORIZONS = (60, 180, 300)
PARTITION_TRAIN = 1
PARTITION_VALIDATION = 2
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def hierarchical_training_weights(
    user_index: np.ndarray, session_index: np.ndarray
) -> np.ndarray:
    user_index = np.asarray(user_index, dtype=np.int64)
    session_index = np.asarray(session_index, dtype=np.int64)
    _, first_positions, origin_counts = np.unique(
        session_index, return_index=True, return_counts=True
    )
    session_users = user_index[first_positions]
    user_session_counts = np.bincount(session_users)
    session_count_lookup = np.zeros(int(session_index.max()) + 1, dtype=np.float64)
    session_count_lookup[np.unique(session_index)] = origin_counts
    weights = 1.0 / (
        session_count_lookup[session_index] * user_session_counts[user_index]
    )
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


def hierarchical_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, float | int]:
    frame = pd.DataFrame(
        {
            "user": users,
            "session": sessions,
            "error": predictions - targets,
        }
    )
    frame["absolute_error"] = frame["error"].abs()
    frame["squared_error"] = frame["error"] ** 2
    session = frame.groupby(["user", "session"], sort=False).agg(
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        bias=("error", "mean"),
        origins=("error", "size"),
    )
    session["rmse"] = np.sqrt(session["mse"])
    user = session.groupby(level="user", sort=False).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        bias=("bias", "mean"),
        sessions=("mae", "size"),
        origins=("origins", "sum"),
    )
    return {
        "mae_bpm": float(user["mae"].mean()),
        "rmse_bpm": float(user["rmse"].mean()),
        "bias_bpm": float(user["bias"].mean()),
        "users": int(len(user)),
        "sessions": int(user["sessions"].sum()),
        "origins": int(user["origins"].sum()),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    array_dir = args.array_dir
    features = np.load(args.tabular_features, mmap_mode="r")
    targets = np.load(array_dir / "targets.npy", mmap_mode="r")
    dataset = np.load(array_dir / "dataset_code.npy", mmap_mode="r")
    evaluation = np.load(array_dir / "evaluation_origin.npy", mmap_mode="r")
    unseen = np.load(array_dir / "unseen_user_partition.npy", mmap_mode="r")
    external = np.load(array_dir / "primary_external_partition.npy", mmap_mode="r")
    sport = np.load(array_dir / "sport_code.npy", mmap_mode="r")
    users = np.load(array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(array_dir / "session_index.npy", mmap_mode="r")
    feature_metadata = json.loads(
        args.tabular_features.with_suffix(".json").read_text(encoding="utf-8")
    )
    feature_names = feature_metadata["feature_names"]
    train_index = np.flatnonzero((dataset == 0) & (unseen == PARTITION_TRAIN))
    validation_index = np.flatnonzero(
        (dataset == 0) & (unseen == PARTITION_VALIDATION) & (evaluation == 1)
    )
    if args.max_train_rows and len(train_index) > args.max_train_rows:
        generator = np.random.default_rng(SEED)
        train_index = np.sort(
            generator.choice(train_index, size=args.max_train_rows, replace=False)
        )
    training_users = np.asarray(users[train_index], dtype=np.int64)
    validation_users = np.asarray(users[validation_index], dtype=np.int64)
    user_overlap = int(
        np.intersect1d(np.unique(training_users), np.unique(validation_users)).size
    )
    if user_overlap:
        raise AssertionError("training and unseen-user validation users overlap")
    x_train = np.asarray(features[train_index], dtype=np.float32)
    x_validation = np.asarray(features[validation_index], dtype=np.float32)
    last_hr_train = x_train[:, 0].copy()
    last_hr_validation = x_validation[:, 0].copy()
    weights = hierarchical_training_weights(
        training_users, np.asarray(sessions[train_index], dtype=np.int64)
    )
    validation_weights = hierarchical_training_weights(
        validation_users, np.asarray(sessions[validation_index], dtype=np.int64)
    )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    models: list[xgb.XGBRegressor] = []
    model_summaries: list[dict[str, object]] = []
    for target_position, horizon in enumerate(HORIZONS):
        y_train = np.asarray(targets[train_index, target_position], dtype=np.float32) - last_hr_train
        y_validation = (
            np.asarray(targets[validation_index, target_position], dtype=np.float32)
            - last_hr_validation
        )
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.01,
            reg_lambda=2.0,
            max_bin=256,
            tree_method="hist",
            device="cuda",
            early_stopping_rounds=args.early_stopping_rounds,
            random_state=SEED + horizon,
            n_jobs=args.n_jobs,
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=weights,
            eval_set=[(x_validation, y_validation)],
            sample_weight_eval_set=[validation_weights],
            verbose=50,
        )
        path = args.model_dir / f"xgboost_residual_{horizon}s_v0_8_0.json"
        model.save_model(path)
        models.append(model)
        model_summaries.append(
            {
                "horizon_seconds": horizon,
                "best_iteration": int(model.best_iteration),
                "best_validation_mae_residual_bpm": float(model.best_score),
                "model_file": str(path),
            }
        )
    del x_train, x_validation, weights, validation_weights
    evaluation_index = np.flatnonzero(
        ((dataset == 0) & (evaluation == 1))
        | ((dataset == 1) & (external == EXTERNAL_FROZEN))
    )
    x_evaluation = np.asarray(features[evaluation_index], dtype=np.float32)
    last_hr = x_evaluation[:, 0]
    predictions = np.column_stack(
        [
            np.clip(model.predict(x_evaluation) + last_hr, 30.0, 240.0)
            for model in models
        ]
    ).astype(np.float32)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        row_index=evaluation_index.astype(np.int64),
        predictions=predictions,
    )
    metric_rows: list[dict[str, object]] = []
    regime_masks: dict[str, np.ndarray] = {
        "unseen_user_validation": (dataset[evaluation_index] == 0)
        & (unseen[evaluation_index] == PARTITION_VALIDATION),
        "unseen_user_test": (dataset[evaluation_index] == 0)
        & (unseen[evaluation_index] == PARTITION_TEST),
        "goldencheetah_frozen_external": (dataset[evaluation_index] == 1)
        & (external[evaluation_index] == EXTERNAL_FROZEN),
    }
    for code, family in ((1, "outdoor_cycling"), (2, "indoor_virtual_cycling"), (3, "running")):
        regime_masks[f"goldencheetah_external__{family}"] = (
            (dataset[evaluation_index] == 1)
            & (external[evaluation_index] == EXTERNAL_FROZEN)
            & (sport[evaluation_index] == code)
        )
    for regime, subset_mask in regime_masks.items():
        subset_rows = evaluation_index[subset_mask]
        for target_position, horizon in enumerate(HORIZONS):
            metrics = hierarchical_metrics(
                predictions[subset_mask, target_position],
                np.asarray(targets[subset_rows, target_position], dtype=np.float32),
                np.asarray(users[subset_rows], dtype=np.int32),
                np.asarray(sessions[subset_rows], dtype=np.int32),
            )
            metric_rows.append(
                {
                    "model_version": MODEL_VERSION,
                    "protocol": "unseen_user_train",
                    "regime": regime,
                    "model": "xgboost",
                    "horizon_seconds": horizon,
                    **metrics,
                    "aggregation": "origin-within-session, session-within-user, equal-user mean",
                }
            )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    importance_rows: list[dict[str, object]] = []
    for horizon, model in zip(HORIZONS, models):
        scores = model.get_booster().get_score(importance_type="gain")
        for feature_index, feature_name in enumerate(feature_names):
            importance_rows.append(
                {
                    "horizon_seconds": horizon,
                    "feature": feature_name,
                    "gain": float(scores.get(f"f{feature_index}", 0.0)),
                }
            )
    with (args.model_dir / "xgboost_feature_gain_v0_8_0.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(importance_rows[0]))
        writer.writeheader()
        writer.writerows(importance_rows)
    finite_failures = int((~np.isfinite(predictions)).sum())
    range_failures = int(((predictions < 30) | (predictions > 240)).sum())
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "model_version": MODEL_VERSION,
        "protocol": "unseen_user_train",
        "seed": SEED,
        "device": "cuda",
        "xgboost_version": xgb.__version__,
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "evaluation_prediction_rows": int(len(evaluation_index)),
        "training_users": int(len(np.unique(training_users))),
        "validation_users": int(len(np.unique(validation_users))),
        "train_validation_user_overlap": user_overlap,
        "hierarchical_training_weights": True,
        "hierarchical_validation_weights": True,
        "residual_target": "future_hr_minus_last_context_hr",
        "model_summaries": model_summaries,
        "metric_rows": len(metric_rows),
        "prediction_nonfinite_values": finite_failures,
        "prediction_range_failures": range_failures,
        "predictions_file": str(args.predictions),
        "metrics_file": str(args.metrics),
    }
    payload["all_assertions_pass"] = (
        user_overlap == 0 and finite_failures == 0 and range_failures == 0
    )
    atomic_json(args.audit, payload)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the causal XGBoost baseline.")
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--tabular-features", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
