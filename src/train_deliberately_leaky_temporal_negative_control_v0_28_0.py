from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from build_deliberately_leaky_temporal_partition_v0_28_0 import (
    ANALYSIS_VERSION,
    PARTITION_CALIBRATION,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    index_sha256,
    pairwise_overlap_counts,
    require,
)
from run_q1_multiseed_queue import atomic_json, sha256_file, utc_now
from train_neural_baselines import compute_normalization, prepare_batch, set_seed
from train_uncertainty_model import (
    HistoryQuantileTCN,
    conformal_thresholds,
    compute_history_normalization,
    pinball_loss,
    predict_quantiles,
    prepare_history_batch,
    uncertainty_rows,
)
from train_xgboost_baseline import HORIZONS, hierarchical_metrics, hierarchical_training_weights


PROTOCOL = "deliberately_leaky_strict_temporal_negative_control"
TRAINING_HISTORY_MODE = "always_zero"
SELECTION_METRIC = "mean 1/3/5-min zero-history validation hierarchical MAE"


def validate_configuration(config: dict[str, object]) -> None:
    require(config.get("analysis_version") == ANALYSIS_VERSION, "config version")
    require(config.get("valid_for_generalization") is False, "validity flag")
    require(
        config.get("required_acknowledgement_flag")
        == "--acknowledge-invalid-generalization",
        "acknowledgement flag",
    )
    seeds = config.get("seeds")
    require(seeds == [20260722, 20260723, 20260724], "locked seed set")
    training = config.get("training")
    require(isinstance(training, dict), "missing training config")
    expected = {
        "formal_budget_locked": True,
        "training_history_mode": TRAINING_HISTORY_MODE,
        "signal_input": "multimodal",
        "epoch_samples": 500_000,
        "batch_size": 2_048,
        "inference_batch_size": 4_096,
        "max_epochs": 40,
        "patience": 4,
        "learning_rate": 0.001,
        "history_dropout": 0.0,
        "selection_metric": SELECTION_METRIC,
        "initialize_from_clean_checkpoint": False,
    }
    require(
        all(training.get(key) == value for key, value in expected.items()),
        "training configuration differs from the locked negative-control budget",
    )


def validate_runtime_budget(args: argparse.Namespace, training: dict[str, object]) -> None:
    for name in (
        "epoch_samples",
        "batch_size",
        "inference_batch_size",
        "max_epochs",
        "patience",
        "learning_rate",
    ):
        require(getattr(args, name) == training[name], f"runtime budget changed: {name}")
    require(args.history_dropout == 0.0, "history dropout must be zero")
    require(args.seed in (20260722, 20260723, 20260724), "seed outside lock")


def load_clean_control_row_index(path: Path) -> np.ndarray:
    require(path.is_file(), f"missing clean-control predictions: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(
            set(archive.files) == {"row_index", "zero_history_quantiles"},
            "clean-control archive schema mismatch",
        )
        row_index = np.asarray(archive["row_index"], dtype=np.int64)
        quantiles = archive["zero_history_quantiles"]
        require(
            quantiles.ndim == 3
            and quantiles.shape[0] == len(row_index)
            and quantiles.shape[1:] == (3, 7),
            "clean-control quantile shape mismatch",
        )
    require(len(row_index) > 0, "empty clean-control row index")
    require(len(np.unique(row_index)) == len(row_index), "duplicate clean-control rows")
    return row_index


def validate_test_alignment(
    test_index: np.ndarray, clean_row_index: np.ndarray, expected_hash: str
) -> str:
    test_index = np.asarray(test_index, dtype=np.int64)
    clean_row_index = np.asarray(clean_row_index, dtype=np.int64)
    require(test_index.ndim == clean_row_index.ndim == 1, "test index dimension")
    require(
        np.array_equal(test_index, clean_row_index),
        "leaky and clean test row indices are not identical and ordered",
    )
    observed_hash = index_sha256(test_index)
    require(observed_hash == expected_hash, "fixed-test hash mismatch")
    return observed_hash


@torch.inference_mode()
def predict_zero(
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
        "multimodal",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict[str, object]:
    require(
        args.acknowledge_invalid_generalization,
        "refusing to run without --acknowledge-invalid-generalization",
    )
    config = json.loads(args.configuration.read_text(encoding="utf-8"))
    validate_configuration(config)
    training = config["training"]
    validate_runtime_budget(args, training)
    set_seed(args.seed)
    require(torch.cuda.is_available(), "CUDA is required for formal training")
    device = torch.device("cuda")

    arrays = {
        "values": np.load(args.array_dir / "sequence_values.npy", mmap_mode="r"),
        "masks": np.load(args.array_dir / "sequence_masks.npy", mmap_mode="r"),
        "targets": np.load(args.array_dir / "targets.npy", mmap_mode="r"),
        "elapsed": np.load(args.array_dir / "origin_offset_seconds.npy", mmap_mode="r"),
        "sport": np.load(args.array_dir / "sport_code.npy", mmap_mode="r"),
        "dataset": np.load(args.array_dir / "dataset_code.npy", mmap_mode="r"),
        "evaluation": np.load(args.array_dir / "evaluation_origin.npy", mmap_mode="r"),
        "users": np.load(args.array_dir / "user_index.npy", mmap_mode="r"),
        "sessions": np.load(args.array_dir / "session_index.npy", mmap_mode="r"),
        "origin_time": np.load(args.array_dir / "origin_time.npy", mmap_mode="r"),
        "partition": np.load(args.leaky_partition, mmap_mode="r"),
    }
    require(len({len(value) for value in arrays.values()}) == 1, "array length mismatch")
    history_values = np.load(
        args.array_dir / "session_history_values.npy", mmap_mode="r"
    )
    history_mask = np.load(
        args.array_dir / "session_history_mask.npy", mmap_mode="r"
    )
    history_metadata = json.loads(
        (args.array_dir / "history_metadata.json").read_text(encoding="utf-8")
    )
    require(
        history_metadata.get("strict_rule")
        == "only sessions completed at or before the current session_start_time",
        "causal-history rule mismatch",
    )
    partition_audit = json.loads(args.leaky_partition_audit.read_text(encoding="utf-8"))
    require(partition_audit.get("all_assertions_pass") is True, "partition audit failed")
    require(partition_audit.get("valid_for_generalization") is False, "partition validity")
    require(
        partition_audit.get("output_sha256") == sha256_file(args.leaky_partition),
        "partition hash differs from audited artifact",
    )

    partition = arrays["partition"]
    train_index = np.flatnonzero(partition == PARTITION_TRAIN)
    validation_index = np.flatnonzero(partition == PARTITION_VALIDATION)
    calibration_index = np.flatnonzero(partition == PARTITION_CALIBRATION)
    test_index = np.flatnonzero(partition == PARTITION_TEST)
    split_indices = {
        "train": train_index,
        "validation": validation_index,
        "calibration": calibration_index,
        "test": test_index,
    }
    require(all(len(index) > 0 for index in split_indices.values()), "empty partition")
    row_overlaps = pairwise_overlap_counts(
        np.arange(len(partition), dtype=np.int64), partition
    )
    require(not any(row_overlaps.values()), "exact rows overlap across partitions")
    session_overlaps = pairwise_overlap_counts(arrays["sessions"], partition)
    require(session_overlaps["train_test"] > 0, "deliberate session leakage absent")
    user_overlaps = pairwise_overlap_counts(arrays["users"], partition)
    expected_fixed = config["fixed_test"]
    clean_row_index = load_clean_control_row_index(args.clean_control_predictions)
    test_hash = validate_test_alignment(
        test_index,
        clean_row_index,
        str(expected_fixed["row_index_sha256_int64_little_endian"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_normalization = compute_normalization(
        arrays["values"], arrays["masks"], arrays["elapsed"], train_index
    )
    history_normalization = compute_history_normalization(
        history_values, history_mask, np.asarray(arrays["sessions"][train_index])
    )
    input_normalization_path = args.output_dir / "input_normalization.json"
    history_normalization_path = args.output_dir / "history_normalization.json"
    atomic_json(input_normalization_path, input_normalization)
    atomic_json(history_normalization_path, history_normalization)

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
    # Keep the leaf short because the Windows project path is already close to
    # the legacy MAX_PATH boundary.
    checkpoint = args.output_dir / "best.pt"
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
                force_zero_history=True,
            )
            if epoch == 0 and start == 0:
                require(
                    not bool(torch.count_nonzero(history_batch_mask).item()),
                    "history mask was not forced to zero",
                )
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

        validation_prediction = predict_zero(
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
        horizon_mae = []
        for position in range(3):
            horizon_mae.append(
                float(
                    hierarchical_metrics(
                        validation_prediction[:, position, 3],
                        np.asarray(arrays["targets"][validation_index, position]),
                        np.asarray(arrays["users"][validation_index]),
                        np.asarray(arrays["sessions"][validation_index]),
                    )["mae_bpm"]
                )
            )
        score = float(np.mean(horizon_mae))
        scheduler.step(score)
        record = {
            "epoch": epoch + 1,
            "training_pinball_loss": loss_sum / examples,
            "validation_zero_history_mae": score,
            "validation_horizon_mae_bpm": horizon_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch + 1
            no_improvement = 0
            # Pass an already-open binary stream so PyTorch does not route the
            # non-ASCII Windows project path through its ASCII filename layer.
            with checkpoint.open("wb") as checkpoint_handle:
                torch.save(
                    {
                        "analysis_version": ANALYSIS_VERSION,
                        "protocol": PROTOCOL,
                        "valid_for_generalization": False,
                        "model": model.state_dict(),
                        "epoch": best_epoch,
                        "validation_score": best_score,
                        "training_history_mode": TRAINING_HISTORY_MODE,
                        "selection_metric": SELECTION_METRIC,
                        "seed": args.seed,
                        "fresh_random_initialization": True,
                    },
                    checkpoint_handle,
                )
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                break

    with checkpoint.open("rb") as checkpoint_handle:
        saved = torch.load(
            checkpoint_handle, map_location=device, weights_only=False
        )
    require(saved.get("fresh_random_initialization") is True, "checkpoint provenance")
    model.load_state_dict(saved["model"])
    calibration_prediction = predict_zero(
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
    thresholds = conformal_thresholds(calibration_prediction, calibration_targets)
    thresholds_path = args.output_dir / "conformal_thresholds.json"
    atomic_json(
        thresholds_path,
        {
            "analysis_version": ANALYSIS_VERSION,
            "valid_for_generalization": False,
            "coverage_guarantee_valid": False,
            "invalid_reason": "calibration shares sessions and overlapping windows with training and fixed test",
            "calibration_rows": int(len(calibration_index)),
            "thresholds": thresholds,
        },
    )

    resolved_config_path = args.output_dir / "resolved_config.json"
    atomic_json(
        resolved_config_path,
        {
            "analysis_version": ANALYSIS_VERSION,
            "protocol": PROTOCOL,
            "valid_for_generalization": False,
            "leaderboard_eligible": False,
            "seed": args.seed,
            "formal_budget_locked": True,
            "training_history_mode": TRAINING_HISTORY_MODE,
            "training_and_validation_history_forced_zero": True,
            "selection_metric": SELECTION_METRIC,
            "epoch_samples": args.epoch_samples,
            "batch_size": args.batch_size,
            "inference_batch_size": args.inference_batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "history_dropout": args.history_dropout,
            "fresh_random_initialization": True,
            "acknowledge_invalid_generalization": True,
        },
    )

    test_prediction = predict_zero(
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
    temporary_predictions = args.predictions.with_suffix(args.predictions.suffix + ".tmp")
    with temporary_predictions.open("wb") as handle:
        np.savez_compressed(
            handle,
            row_index=test_index.astype(np.int64),
            zero_history_quantiles=test_prediction,
        )
    temporary_predictions.replace(args.predictions)

    point_rows: list[dict[str, object]] = []
    test_targets = np.asarray(arrays["targets"][test_index])
    for position, horizon in enumerate(HORIZONS):
        point_rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "regime": "deliberately_leaky_within_user_temporal_test",
                "mode": "zero_history_trained",
                "horizon_seconds": horizon,
                "valid_for_generalization": False,
                "leaderboard_eligible": False,
                **hierarchical_metrics(
                    test_prediction[:, position, 3],
                    test_targets[:, position],
                    np.asarray(arrays["users"][test_index]),
                    np.asarray(arrays["sessions"][test_index]),
                ),
            }
        )
    interval_rows = uncertainty_rows(
        "deliberately_leaky_within_user_temporal_test",
        "zero_history_trained",
        test_prediction,
        test_targets,
        np.asarray(arrays["users"][test_index]),
        np.asarray(arrays["sessions"][test_index]),
        thresholds,
        model_version=ANALYSIS_VERSION,
    )
    for row in interval_rows:
        row.update(
            {
                "analysis_version": ANALYSIS_VERSION,
                "valid_for_generalization": False,
                "leaderboard_eligible": False,
                "coverage_guarantee_valid": False,
            }
        )
    write_csv(args.point_metrics, point_rows)
    write_csv(args.interval_metrics, interval_rows)

    crossing = int((np.diff(test_prediction, axis=2) < -1e-6).sum())
    nonfinite = int((~np.isfinite(test_prediction)).sum())
    range_failures = int(((test_prediction < 30) | (test_prediction > 240)).sum())
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "protocol": PROTOCOL,
        "run_purpose": "deliberately_leaky_negative_control",
        "valid_for_generalization": False,
        "leaderboard_eligible": False,
        "invalid_reason": "same-session overlapping windows deliberately contaminate training, validation, and calibration",
        "acknowledge_invalid_generalization": True,
        "seed": args.seed,
        "formal_budget_locked": True,
        "fresh_random_initialization": True,
        "clean_checkpoint_reused_or_warm_started": False,
        "training_history_mode": TRAINING_HISTORY_MODE,
        "training_and_validation_history_forced_zero": True,
        "selection_metric": SELECTION_METRIC,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "calibration_rows": int(len(calibration_index)),
        "test_rows": int(len(test_index)),
        "prediction_rows": int(len(test_index)),
        "test_row_index_sha256_int64_little_endian": test_hash,
        "clean_test_row_index_exact_order_match": True,
        "exact_row_overlap_counts": row_overlaps,
        "session_overlap_counts": session_overlaps,
        "user_overlap_counts": user_overlaps,
        "deliberate_train_test_session_overlap_present": session_overlaps["train_test"] > 0,
        "normalization_fit_partition": "deliberately_leaky_train_only",
        "cqr_fit_partition": "deliberately_leaky_calibration_only",
        "cqr_coverage_guarantee_valid": False,
        "best_epoch": best_epoch,
        "best_validation_zero_history_mae": best_score,
        "epoch_samples": args.epoch_samples,
        "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "history_dropout": args.history_dropout,
        "history": records,
        "quantile_crossing_failures": crossing,
        "nonfinite_prediction_values": nonfinite,
        "prediction_range_failures": range_failures,
        "point_metric_rows": len(point_rows),
        "interval_metric_rows": len(interval_rows),
        "configuration": str(args.configuration.resolve()),
        "configuration_sha256": sha256_file(args.configuration),
        "partition": str(args.leaky_partition.resolve()),
        "partition_sha256": sha256_file(args.leaky_partition),
        "partition_audit": str(args.leaky_partition_audit.resolve()),
        "partition_audit_sha256": sha256_file(args.leaky_partition_audit),
        "clean_control_predictions": str(args.clean_control_predictions.resolve()),
        "clean_control_predictions_sha256": sha256_file(args.clean_control_predictions),
        "checkpoint": str(checkpoint.resolve()),
        "input_normalization": str(input_normalization_path.resolve()),
        "history_normalization": str(history_normalization_path.resolve()),
        "thresholds_file": str(thresholds_path.resolve()),
        "resolved_config": str(resolved_config_path.resolve()),
        "predictions": str(args.predictions.resolve()),
    }
    payload["all_assertions_pass"] = (
        not any(row_overlaps.values())
        and session_overlaps["train_test"] > 0
        and test_hash == expected_fixed["row_index_sha256_int64_little_endian"]
        and crossing == 0
        and nonfinite == 0
        and range_failures == 0
        and bool(np.array_equal(test_index, clean_row_index))
    )
    atomic_json(args.audit, payload)
    require(bool(payload["all_assertions_pass"]), "training audit failed")
    return payload


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(
        description="Train the deliberately invalid v0.28 zero-history negative control."
    )
    result.add_argument(
        "--acknowledge-invalid-generalization",
        action="store_true",
        required=True,
        help="Required acknowledgement that this run is invalid for generalization.",
    )
    result.add_argument(
        "--configuration",
        type=Path,
        default=root / "configs" / "leaky_negative_control_v0_28_0.json",
    )
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--leaky-partition", type=Path, required=True)
    result.add_argument("--leaky-partition-audit", type=Path, required=True)
    result.add_argument("--clean-control-predictions", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--point-metrics", type=Path, required=True)
    result.add_argument("--interval-metrics", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    result.add_argument("--epoch-samples", type=int, default=500_000)
    result.add_argument("--batch-size", type=int, default=2_048)
    result.add_argument("--inference-batch-size", type=int, default=4_096)
    result.add_argument("--max-epochs", type=int, default=40)
    result.add_argument("--patience", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--history-dropout", type=float, default=0.0)
    result.add_argument("--seed", type=int, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(train(parser().parse_args()), ensure_ascii=False))
