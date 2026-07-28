from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_neural_baselines import (
    SEED,
    atomic_json,
    compute_normalization,
    prepare_batch,
    set_seed,
    utc_now,
)
from train_uncertainty_model import (
    HistoryQuantileTCN,
    conformal_thresholds,
    compute_history_normalization,
    pinball_loss,
    predict_quantiles,
    prepare_history_batch,
    uncertainty_rows,
)
from train_xgboost_baseline import (
    HORIZONS,
    PARTITION_TRAIN,
    PARTITION_TEST,
    PARTITION_VALIDATION,
    hierarchical_metrics,
    hierarchical_training_weights,
)


MODEL_VERSION = "0.12.0"
PARTITION_CALIBRATION = 3
SPORTS = {
    1: "outdoor_cycling",
    2: "indoor_virtual_cycling",
    3: "running",
    4: "walking_hiking",
    7: "strength_cross_training",
}


@torch.inference_mode()
def validation_modes(
    model: torch.nn.Module,
    index: np.ndarray,
    arrays: dict[str, np.ndarray],
    history_values: np.ndarray,
    history_mask: np.ndarray,
    input_normalization: dict[str, object],
    history_normalization: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = predict_quantiles(
        model,
        index,
        arrays["values"],
        arrays["masks"],
        arrays["elapsed"],
        arrays["model_sport"],
        arrays["sessions"],
        history_values,
        history_mask,
        input_normalization,
        history_normalization,
        device,
        batch_size,
        False,
    )
    zero = predict_quantiles(
        model,
        index,
        arrays["values"],
        arrays["masks"],
        arrays["elapsed"],
        arrays["model_sport"],
        arrays["sessions"],
        history_values,
        history_mask,
        input_normalization,
        history_normalization,
        device,
        batch_size,
        True,
    )
    return history, zero


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    held = args.held_sport_code
    family = SPORTS[held]
    device = torch.device("cuda")
    array_dir = args.array_dir
    arrays = {
        "values": np.load(array_dir / "sequence_values.npy", mmap_mode="r"),
        "masks": np.load(array_dir / "sequence_masks.npy", mmap_mode="r"),
        "targets": np.load(array_dir / "targets.npy", mmap_mode="r"),
        "elapsed": np.load(array_dir / "origin_offset_seconds.npy", mmap_mode="r"),
        "sport": np.load(array_dir / "sport_code.npy", mmap_mode="r"),
        "dataset": np.load(array_dir / "dataset_code.npy", mmap_mode="r"),
        "evaluation": np.load(array_dir / "evaluation_origin.npy", mmap_mode="r"),
        "unseen": np.load(array_dir / "unseen_user_partition.npy", mmap_mode="r"),
        "users": np.load(array_dir / "user_index.npy", mmap_mode="r"),
        "sessions": np.load(array_dir / "session_index.npy", mmap_mode="r"),
    }
    model_sport = np.asarray(arrays["sport"], dtype=np.uint8).copy()
    model_sport[model_sport == held] = 0
    arrays["model_sport"] = model_sport
    history_values = np.load(args.history_dir / "session_history_values.npy", mmap_mode="r")
    history_mask = np.load(args.history_dir / "session_history_mask.npy", mmap_mode="r")
    history_metadata = json.loads(
        (args.history_dir / "history_metadata.json").read_text(encoding="utf-8")
    )
    if history_metadata.get("strict_rule") != (
        "only sessions completed at or before the current session_start_time"
    ):
        raise AssertionError("sport-excluded history does not enforce completed-session timing")
    train_index = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["unseen"] == PARTITION_TRAIN)
        & (arrays["sport"] != held)
    )
    validation_index = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["unseen"] == PARTITION_VALIDATION)
        & (arrays["sport"] != held)
        & (arrays["evaluation"] == 1)
    )
    calibration_index = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["unseen"] == PARTITION_CALIBRATION)
        & (arrays["sport"] != held)
        & (arrays["evaluation"] == 1)
    )
    train_users = np.unique(arrays["users"][train_index])
    validation_users = np.unique(arrays["users"][validation_index])
    calibration_users = np.unique(arrays["users"][calibration_index])
    overlaps = {
        "train_validation": int(np.intersect1d(train_users, validation_users).size),
        "train_calibration": int(np.intersect1d(train_users, calibration_users).size),
        "validation_calibration": int(
            np.intersect1d(validation_users, calibration_users).size
        ),
    }
    held_training_rows = int((arrays["sport"][train_index] == held).sum())
    if any(overlaps.values()) or held_training_rows:
        raise AssertionError({"overlaps": overlaps, "held_training_rows": held_training_rows})
    same_user_index = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["unseen"] == PARTITION_TRAIN)
        & (arrays["sport"] == held)
        & (arrays["evaluation"] == 1)
        & np.isin(arrays["users"], train_users)
    )
    joint_index = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["unseen"] == PARTITION_TEST)
        & (arrays["sport"] == held)
        & (arrays["evaluation"] == 1)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_normalization = compute_normalization(
        arrays["values"], arrays["masks"], arrays["elapsed"], train_index
    )
    history_normalization = compute_history_normalization(
        history_values, history_mask, np.asarray(arrays["sessions"][train_index])
    )
    atomic_json(args.output_dir / "input_normalization.json", input_normalization)
    atomic_json(args.output_dir / "history_normalization.json", history_normalization)
    weights = hierarchical_training_weights(
        np.asarray(arrays["users"][train_index]),
        np.asarray(arrays["sessions"][train_index]),
    ).astype(np.float64)
    probabilities = weights / weights.sum()
    model = HistoryQuantileTCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = np.random.default_rng(args.seed)
    checkpoint = args.output_dir / f"{family}_best.pt"
    records: list[dict[str, float | int]] = []
    best_score = math.inf
    best_epoch = -1
    no_improvement = 0
    for epoch in range(args.max_epochs):
        model.train()
        positions = generator.choice(
            len(train_index), size=args.epoch_samples, replace=True, p=probabilities
        )
        epoch_index = train_index[positions]
        loss_sum = 0.0
        examples = 0
        for start in range(0, len(epoch_index), args.batch_size):
            index = np.sort(epoch_index[start : start + args.batch_size])
            sequence, sport_batch, elapsed_batch, last_hr = prepare_batch(
                arrays["values"],
                arrays["masks"],
                arrays["elapsed"],
                arrays["model_sport"],
                index,
                input_normalization,
                device,
            )
            sport_dropout = torch.rand_like(sport_batch.float()) < args.sport_dropout
            sport_batch = torch.where(sport_dropout, torch.zeros_like(sport_batch), sport_batch)
            history_batch, history_batch_mask = prepare_history_batch(
                history_values,
                history_mask,
                arrays["sessions"],
                index,
                history_normalization,
                device,
            )
            history_dropout = torch.rand_like(history_batch_mask) < args.history_dropout
            history_batch_mask = history_batch_mask * (~history_dropout)
            history_batch = history_batch * history_batch_mask
            target = torch.from_numpy(
                np.asarray(arrays["targets"][index], dtype=np.float32)
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(
                    sequence,
                    sport_batch,
                    elapsed_batch,
                    history_batch,
                    history_batch_mask,
                )
                loss = pinball_loss(prediction, target - last_hr[:, None])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(index)
            examples += len(index)
        validation_history, validation_zero = validation_modes(
            model,
            validation_index,
            arrays,
            history_values,
            history_mask,
            input_normalization,
            history_normalization,
            device,
            args.inference_batch_size,
        )
        mode_scores = []
        for prediction in (validation_history, validation_zero):
            mode_scores.append(
                sum(
                    float(
                        hierarchical_metrics(
                            prediction[:, position, 3],
                            np.asarray(arrays["targets"][validation_index, position]),
                            np.asarray(arrays["users"][validation_index]),
                            np.asarray(arrays["sessions"][validation_index]),
                        )["mae_bpm"]
                    )
                    for position in range(3)
                )
                / 3
            )
        score = sum(mode_scores) / 2
        scheduler.step(score)
        record = {
            "epoch": epoch + 1,
            "training_pinball_loss": loss_sum / examples,
            "validation_composite_mae": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        records.append(record)
        print(json.dumps({"family": family, **record}), flush=True)
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch + 1
            no_improvement = 0
            torch.save({"model": model.state_dict()}, checkpoint)
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    calibration_history, calibration_zero = validation_modes(
        model,
        calibration_index,
        arrays,
        history_values,
        history_mask,
        input_normalization,
        history_normalization,
        device,
        args.inference_batch_size,
    )
    calibration_targets = np.asarray(arrays["targets"][calibration_index])
    thresholds = {
        "history_informed": conformal_thresholds(calibration_history, calibration_targets),
        "zero_history": conformal_thresholds(calibration_zero, calibration_targets),
    }
    atomic_json(args.output_dir / "conformal_thresholds.json", thresholds)
    evaluation_index = np.concatenate([same_user_index, joint_index])
    evaluation_history, evaluation_zero = validation_modes(
        model,
        evaluation_index,
        arrays,
        history_values,
        history_mask,
        input_normalization,
        history_normalization,
        device,
        args.inference_batch_size,
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        row_index=evaluation_index.astype(np.int64),
        same_user_rows=np.asarray([len(same_user_index)], dtype=np.int64),
        history_quantiles=evaluation_history,
        zero_history_quantiles=evaluation_zero,
    )
    regime_slices = {
        f"unseen_sport__{family}": slice(0, len(same_user_index)),
        f"joint_user_sport__{family}": slice(len(same_user_index), len(evaluation_index)),
    }
    point_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    for regime, selected_slice in regime_slices.items():
        selected_rows = evaluation_index[selected_slice]
        for mode, prediction in {
            "history_informed": evaluation_history[selected_slice],
            "zero_history": evaluation_zero[selected_slice],
        }.items():
            selected_targets = np.asarray(arrays["targets"][selected_rows])
            for position, horizon in enumerate(HORIZONS):
                point_rows.append(
                    {
                        "model_version": MODEL_VERSION,
                        "held_sport_family": family,
                        "regime": regime,
                        "mode": mode,
                        "horizon_seconds": horizon,
                        **hierarchical_metrics(
                            prediction[:, position, 3],
                            selected_targets[:, position],
                            np.asarray(arrays["users"][selected_rows]),
                            np.asarray(arrays["sessions"][selected_rows]),
                        ),
                    }
                )
            interval_rows.extend(
                uncertainty_rows(
                    regime,
                    mode,
                    prediction,
                    selected_targets,
                    np.asarray(arrays["users"][selected_rows]),
                    np.asarray(arrays["sessions"][selected_rows]),
                    thresholds[mode],
                    model_version=MODEL_VERSION,
                )
            )
    args.point_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.point_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    args.interval_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.interval_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0]))
        writer.writeheader()
        writer.writerows(interval_rows)
    crossing = int(
        (np.diff(evaluation_history, axis=2) < -1e-6).sum()
        + (np.diff(evaluation_zero, axis=2) < -1e-6).sum()
    )
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "seed": args.seed,
        "held_sport_code": held,
        "held_sport_family": family,
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "calibration_rows": int(len(calibration_index)),
        "same_user_sport_shift_rows": int(len(same_user_index)),
        "joint_user_sport_shift_rows": int(len(joint_index)),
        "train_validation_calibration_user_overlaps": overlaps,
        "held_family_training_rows": held_training_rows,
        "held_family_token_mapping": "unknown sport code 0",
        "sport_token_dropout": args.sport_dropout,
        "excluded_family_history_dir": str(args.history_dir),
        "causal_history_version": history_metadata["history_version"],
        "causal_history_strict_rule": history_metadata["strict_rule"],
        "best_epoch": best_epoch,
        "best_validation_composite_mae": best_score,
        "epoch_samples": args.epoch_samples,
        "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "history_dropout": args.history_dropout,
        "history": records,
        "quantile_crossing_failures": crossing,
        "point_metric_rows": len(point_rows),
        "interval_metric_rows": len(interval_rows),
    }
    payload["all_assertions_pass"] = (
        not any(overlaps.values())
        and held_training_rows == 0
        and len(same_user_index) > 0
        and len(joint_index) > 0
        and crossing == 0
    )
    atomic_json(args.audit, payload)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a leave-one-sport-out quantile TCN.")
    parser.add_argument("--held-sport-code", type=int, choices=sorted(SPORTS), required=True)
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--point-metrics", type=Path, required=True)
    parser.add_argument("--interval-metrics", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--epoch-samples", type=int, default=250_000)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--history-dropout", type=float, default=0.2)
    parser.add_argument("--sport-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
