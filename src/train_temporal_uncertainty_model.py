from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

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
    TRAINING_HISTORY_MODES,
    conformal_thresholds,
    compute_history_normalization,
    pinball_loss,
    predict_quantiles,
    prepare_history_batch,
    selection_metric_name,
    uncertainty_rows,
    validation_selection_score,
)
from train_xgboost_baseline import (
    HORIZONS,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    hierarchical_metrics,
    hierarchical_training_weights,
)


MODEL_VERSION = "0.13.0"
PARTITION_CALIBRATION = 3


@torch.inference_mode()
def predict_modes(
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
    common = (
        model,
        index,
        arrays["values"],
        arrays["masks"],
        arrays["elapsed"],
        arrays["sport"],
        arrays["sessions"],
        history_values,
        history_mask,
        input_normalization,
        history_normalization,
        device,
        batch_size,
    )
    return predict_quantiles(*common, False), predict_quantiles(*common, True)


@torch.inference_mode()
def predict_zero_mode(
    model: torch.nn.Module,
    index: np.ndarray,
    arrays: dict[str, np.ndarray],
    history_values: np.ndarray,
    history_mask: np.ndarray,
    input_normalization: dict[str, object],
    history_normalization: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    return predict_quantiles(
        model,
        index,
        arrays["values"],
        arrays["masks"],
        arrays["elapsed"],
        arrays["sport"],
        arrays["sessions"],
        history_values,
        history_mask,
        input_normalization,
        history_normalization,
        device,
        batch_size,
        True,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    training_history_mode = getattr(args, "training_history_mode", "mixed")
    selection_metric = selection_metric_name(training_history_mode)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    arrays = {
        "values": np.load(args.array_dir / "sequence_values.npy", mmap_mode="r"),
        "masks": np.load(args.array_dir / "sequence_masks.npy", mmap_mode="r"),
        "targets": np.load(args.array_dir / "targets.npy", mmap_mode="r"),
        "elapsed": np.load(args.array_dir / "origin_offset_seconds.npy", mmap_mode="r"),
        "sport": np.load(args.array_dir / "sport_code.npy", mmap_mode="r"),
        "dataset": np.load(args.array_dir / "dataset_code.npy", mmap_mode="r"),
        "evaluation": np.load(args.array_dir / "evaluation_origin.npy", mmap_mode="r"),
        "temporal": np.load(args.temporal_partition, mmap_mode="r"),
        "users": np.load(args.array_dir / "user_index.npy", mmap_mode="r"),
        "sessions": np.load(args.array_dir / "session_index.npy", mmap_mode="r"),
    }
    history_values = np.load(
        args.array_dir / "session_history_values.npy", mmap_mode="r"
    )
    history_mask = np.load(
        args.array_dir / "session_history_mask.npy", mmap_mode="r"
    )
    history_metadata = json.loads(
        (args.array_dir / "history_metadata.json").read_text(encoding="utf-8")
    )
    strict_audit = json.loads(args.temporal_audit.read_text(encoding="utf-8"))
    if history_metadata.get("strict_rule") != (
        "only sessions completed at or before the current session_start_time"
    ):
        raise AssertionError("history does not enforce completed-session timing")
    if not strict_audit.get("all_assertions_pass") or strict_audit.get(
        "strict_order_failures"
    ):
        raise AssertionError("strict temporal partition audit failed")

    train_index = np.flatnonzero(
        (arrays["dataset"] == 0) & (arrays["temporal"] == PARTITION_TRAIN)
    )
    validation_index = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["temporal"] == PARTITION_VALIDATION)
        & (arrays["evaluation"] == 1)
    )
    calibration_index = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["temporal"] == PARTITION_CALIBRATION)
        & (arrays["evaluation"] == 1)
    )
    test_index = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["temporal"] == PARTITION_TEST)
        & (arrays["evaluation"] == 1)
    )
    split_indices = {
        "train": train_index,
        "validation": validation_index,
        "calibration": calibration_index,
        "test": test_index,
    }
    session_sets = {
        name: np.unique(arrays["sessions"][index])
        for name, index in split_indices.items()
    }
    user_sets = {
        name: np.unique(arrays["users"][index])
        for name, index in split_indices.items()
    }
    session_overlaps: dict[str, int] = {}
    user_overlaps: dict[str, int] = {}
    names = list(split_indices)
    for left_position, left in enumerate(names):
        for right in names[left_position + 1 :]:
            key = f"{left}_{right}"
            session_overlaps[key] = int(
                np.intersect1d(session_sets[left], session_sets[right]).size
            )
            user_overlaps[key] = int(
                np.intersect1d(user_sets[left], user_sets[right]).size
            )
    if any(session_overlaps.values()):
        raise AssertionError(f"temporal session overlap: {session_overlaps}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_normalization = compute_normalization(
        arrays["values"], arrays["masks"], arrays["elapsed"], train_index
    )
    history_normalization = compute_history_normalization(
        history_values, history_mask, np.asarray(arrays["sessions"][train_index])
    )
    atomic_json(args.output_dir / "input_normalization.json", input_normalization)
    atomic_json(
        args.output_dir / "history_normalization.json", history_normalization
    )
    weights = hierarchical_training_weights(
        np.asarray(arrays["users"][train_index]),
        np.asarray(arrays["sessions"][train_index]),
    ).astype(np.float64)
    probabilities = weights / weights.sum()
    model = HistoryQuantileTCN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = np.random.default_rng(args.seed)
    checkpoint = args.output_dir / "temporal_history_quantile_tcn_best.pt"
    records: list[dict[str, object]] = []
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
                arrays["sport"],
                index,
                input_normalization,
                device,
            )
            history_batch, history_batch_mask = prepare_history_batch(
                history_values,
                history_mask,
                arrays["sessions"],
                index,
                history_normalization,
                device,
                force_zero_history=training_history_mode == "always_zero",
            )
            if training_history_mode == "mixed":
                dropout = torch.rand_like(history_batch_mask) < args.history_dropout
                history_batch_mask = history_batch_mask * (~dropout)
                history_batch = history_batch * history_batch_mask
            target = torch.from_numpy(
                np.asarray(arrays["targets"][index], dtype=np.float32)
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                residual = model(
                    sequence,
                    sport_batch,
                    elapsed_batch,
                    history_batch,
                    history_batch_mask,
                )
                loss = pinball_loss(residual, target - last_hr[:, None])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(index)
            examples += len(index)

        validation_history = None
        if training_history_mode == "mixed":
            validation_history, validation_zero = predict_modes(
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
        else:
            validation_zero = predict_zero_mode(
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

        def validation_mae(prediction: np.ndarray) -> float:
            return float(
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

        validation_history_mae = (
            validation_mae(validation_history)
            if validation_history is not None
            else None
        )
        validation_zero_history_mae = validation_mae(validation_zero)
        score = validation_selection_score(
            training_history_mode=training_history_mode,
            history_mae=validation_history_mae,
            zero_history_mae=validation_zero_history_mae,
        )
        scheduler.step(score)
        record = {
            "epoch": epoch + 1,
            "training_pinball_loss": loss_sum / examples,
            "validation_composite_mae": score,
            "validation_history_mae": validation_history_mae,
            "validation_zero_history_mae": validation_zero_history_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch + 1
            no_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "validation_score": best_score,
                    "validation_history_mae": validation_history_mae,
                    "validation_zero_history_mae": validation_zero_history_mae,
                    "training_history_mode": training_history_mode,
                    "selection_metric": selection_metric,
                },
                checkpoint,
            )
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    calibration_history = None
    if training_history_mode == "mixed":
        calibration_history, calibration_zero = predict_modes(
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
    else:
        calibration_zero = predict_zero_mode(
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
        "zero_history": conformal_thresholds(calibration_zero, calibration_targets),
    }
    if calibration_history is not None:
        thresholds["history_informed"] = conformal_thresholds(
            calibration_history, calibration_targets
        )
    atomic_json(args.output_dir / "conformal_thresholds.json", thresholds)
    resolved_config_path = args.output_dir / "resolved_config.json"
    atomic_json(
        resolved_config_path,
        {
            "analysis_version": "0.23.0" if training_history_mode == "always_zero" else MODEL_VERSION,
            "model_version": MODEL_VERSION,
            "seed": args.seed,
            "protocol": "strict_temporal",
            "training_history_mode": training_history_mode,
            "selection_metric": selection_metric,
            "epoch_samples": args.epoch_samples,
            "batch_size": args.batch_size,
            "inference_batch_size": args.inference_batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "history_dropout": args.history_dropout,
        },
    )
    test_history = None
    if training_history_mode == "mixed":
        test_history, test_zero = predict_modes(
            model,
            test_index,
            arrays,
            history_values,
            history_mask,
            input_normalization,
            history_normalization,
            device,
            args.inference_batch_size,
        )
    else:
        test_zero = predict_zero_mode(
            model,
            test_index,
            arrays,
            history_values,
            history_mask,
            input_normalization,
            history_normalization,
            device,
            args.inference_batch_size,
        )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    if test_history is None:
        np.savez_compressed(
            args.predictions,
            row_index=test_index.astype(np.int64),
            zero_history_quantiles=test_zero,
        )
    else:
        np.savez_compressed(
            args.predictions,
            row_index=test_index.astype(np.int64),
            history_quantiles=test_history,
            zero_history_quantiles=test_zero,
        )

    point_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    test_targets = np.asarray(arrays["targets"][test_index])
    mode_predictions = {"zero_history": test_zero}
    if test_history is not None:
        mode_predictions["history_informed"] = test_history
    for mode, prediction in mode_predictions.items():
        for position, horizon in enumerate(HORIZONS):
            point_rows.append(
                {
                    "model_version": MODEL_VERSION,
                    "regime": "within_user_temporal_test",
                    "mode": mode,
                    "horizon_seconds": horizon,
                    **hierarchical_metrics(
                        prediction[:, position, 3],
                        test_targets[:, position],
                        np.asarray(arrays["users"][test_index]),
                        np.asarray(arrays["sessions"][test_index]),
                    ),
                }
            )
        interval_rows.extend(
            uncertainty_rows(
                "within_user_temporal_test",
                mode,
                prediction,
                test_targets,
                np.asarray(arrays["users"][test_index]),
                np.asarray(arrays["sessions"][test_index]),
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

    integrity_predictions = [test_zero]
    if test_history is not None:
        integrity_predictions.append(test_history)
    crossing = int(
        sum((np.diff(prediction, axis=2) < -1e-6).sum() for prediction in integrity_predictions)
    )
    nonfinite = int(
        sum((~np.isfinite(prediction)).sum() for prediction in integrity_predictions)
    )
    range_failures = int(
        sum(
            ((prediction < 30) | (prediction > 240)).sum()
            for prediction in integrity_predictions
        )
    )
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "model_version": MODEL_VERSION,
        "protocol": "strict_temporal",
        "seed": args.seed,
        "training_history_mode": training_history_mode,
        "selection_metric": selection_metric,
        "training_and_validation_history_forced_zero": (
            training_history_mode == "always_zero"
        ),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "calibration_rows": int(len(calibration_index)),
        "test_rows": int(len(test_index)),
        "users": {name: int(len(item)) for name, item in user_sets.items()},
        "sessions": {name: int(len(item)) for name, item in session_sets.items()},
        "session_overlaps": session_overlaps,
        "user_overlaps_expected_and_allowed": user_overlaps,
        "strict_temporal_partition_version": strict_audit["version"],
        "strict_temporal_order_failures": strict_audit["strict_order_failures"],
        "boundary_sessions_excluded": strict_audit["excluded_boundary_sessions"],
        "causal_history_version": history_metadata["history_version"],
        "causal_history_strict_rule": history_metadata["strict_rule"],
        "best_epoch": best_epoch,
        "best_validation_composite_mae": best_score,
        "best_validation_history_mae": saved.get("validation_history_mae"),
        "best_validation_zero_history_mae": saved.get(
            "validation_zero_history_mae"
        ),
        "epoch_samples": args.epoch_samples,
        "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "history_dropout": args.history_dropout,
        "history": records,
        "prediction_rows": int(len(test_index)),
        "quantile_crossing_failures": crossing,
        "nonfinite_prediction_values": nonfinite,
        "prediction_range_failures": range_failures,
        "point_metric_rows": len(point_rows),
        "interval_metric_rows": len(interval_rows),
        "checkpoint": str(checkpoint),
        "thresholds_file": str(args.output_dir / "conformal_thresholds.json"),
        "resolved_config": str(resolved_config_path),
    }
    payload["all_assertions_pass"] = (
        not any(session_overlaps.values())
        and strict_audit["strict_order_failures"] == 0
        and crossing == 0
        and nonfinite == 0
        and range_failures == 0
    )
    atomic_json(args.audit, payload)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--temporal-partition", type=Path, required=True)
    result.add_argument("--temporal-audit", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--point-metrics", type=Path, required=True)
    result.add_argument("--interval-metrics", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=2048)
    result.add_argument("--inference-batch-size", type=int, default=4096)
    result.add_argument("--epoch-samples", type=int, default=500_000)
    result.add_argument("--max-epochs", type=int, default=20)
    result.add_argument("--patience", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--history-dropout", type=float, default=0.2)
    result.add_argument(
        "--training-history-mode",
        choices=TRAINING_HISTORY_MODES,
        default="mixed",
        help=(
            "mixed preserves the dual-mode rule; always_zero forces zero history "
            "throughout training and validation and selects only on zero-history MAE."
        ),
    )
    result.add_argument("--seed", type=int, default=SEED + 13)
    return result


if __name__ == "__main__":
    print(json.dumps(train(parser().parse_args())))
