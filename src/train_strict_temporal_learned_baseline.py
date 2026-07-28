from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from build_strict_temporal_partition import validate_order
from train_neural_baselines import (
    MODEL_VERSION as ARCHITECTURE_VERSION,
    build_model,
    compute_normalization,
    predict_indices,
    prepare_batch,
    set_seed,
    utc_now,
)
from train_xgboost_baseline import (
    HORIZONS,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    hierarchical_metrics,
    hierarchical_training_weights,
)


ANALYSIS_VERSION = "0.22.0"
STRICT_PARTITION_VERSION = "0.13.0"
STRICT_PARTITION_RULE = (
    "exclude all sessions touching or overlapping a boundary between "
    "ordered temporal partitions"
)
PARTITION_CALIBRATION = 3
PARTITION_EXCLUDED = 5
ORDERED_PARTITIONS = (1, 2, 3, 4)
FORMAL_EPOCH_SAMPLES = 500_000
FORMAL_MAX_EPOCHS = 40
FORMAL_PATIENCE = 4
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    resolved_config: Path
    normalization: Path
    checkpoint: Path
    predictions: Path
    metrics: Path
    audit: Path


def resolve_run_paths(
    output_root: Path,
    run_purpose: str,
    model: str,
    seed: int,
) -> RunPaths:
    if run_purpose not in {"formal", "smoke"}:
        raise ValueError(f"unsupported run purpose: {run_purpose}")
    if model not in {"gru", "tcn"}:
        raise ValueError(f"unsupported model: {model}")
    run_dir = output_root / run_purpose / model / f"seed_{seed}"
    return RunPaths(
        run_dir=run_dir,
        resolved_config=run_dir / "resolved_config.json",
        normalization=run_dir / "normalization_temporal_train.json",
        checkpoint=run_dir / f"{model}_best.pt",
        predictions=run_dir / "strict_temporal_test_predictions.npz",
        metrics=run_dir / "strict_temporal_test_metrics.csv",
        audit=run_dir / "audit.json",
    )


def resolve_argument_paths(args: argparse.Namespace) -> RunPaths:
    explicit_names = ("output_dir", "predictions", "metrics", "audit")
    explicit_values = {
        name: getattr(args, name, None) for name in explicit_names
    }
    if any(value is not None for value in explicit_values.values()):
        if not all(value is not None for value in explicit_values.values()):
            raise ValueError(
                "--output-dir, --predictions, --metrics, and --audit must be "
                "provided together"
            )
        run_dir = Path(explicit_values["output_dir"])
        return RunPaths(
            run_dir=run_dir,
            resolved_config=run_dir / "resolved_config.json",
            normalization=run_dir / "normalization_temporal_train.json",
            checkpoint=run_dir / f"{args.model}_best.pt",
            predictions=Path(explicit_values["predictions"]),
            metrics=Path(explicit_values["metrics"]),
            audit=Path(explicit_values["audit"]),
        )
    return resolve_run_paths(
        args.output_root, args.run_purpose, args.model, args.seed
    )


def validate_budget(
    run_purpose: str,
    epoch_samples: int,
    max_epochs: int,
    patience: int,
) -> bool:
    if epoch_samples < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("epoch_samples, max_epochs, and patience must be positive")
    formal_locked = (
        epoch_samples == FORMAL_EPOCH_SAMPLES
        and max_epochs == FORMAL_MAX_EPOCHS
        and patience == FORMAL_PATIENCE
    )
    if run_purpose == "formal" and not formal_locked:
        raise ValueError(
            "formal runs require epoch_samples=500000, max_epochs=40, patience=4"
        )
    if run_purpose == "smoke" and formal_locked:
        raise ValueError(
            "smoke runs must use a reduced budget to remain distinguishable"
        )
    return formal_locked


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no metric rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"external_path_redacted/{resolved.name}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def split_indices(
    dataset: np.ndarray,
    evaluation: np.ndarray,
    temporal: np.ndarray,
) -> dict[str, np.ndarray]:
    if not (len(dataset) == len(evaluation) == len(temporal)):
        raise ValueError("partition arrays have different lengths")
    return {
        "train": np.flatnonzero(
            (dataset == 0) & (temporal == PARTITION_TRAIN)
        ),
        "validation": np.flatnonzero(
            (dataset == 0)
            & (temporal == PARTITION_VALIDATION)
            & (evaluation == 1)
        ),
        "calibration": np.flatnonzero(
            (dataset == 0)
            & (temporal == PARTITION_CALIBRATION)
            & (evaluation == 1)
        ),
        "test": np.flatnonzero(
            (dataset == 0)
            & (temporal == PARTITION_TEST)
            & (evaluation == 1)
        ),
    }


def pairwise_overlap_counts(
    index_sets: dict[str, np.ndarray],
) -> dict[str, int]:
    names = list(index_sets)
    result: dict[str, int] = {}
    for left_position, left in enumerate(names):
        for right in names[left_position + 1 :]:
            result[f"{left}_{right}"] = int(
                np.intersect1d(index_sets[left], index_sets[right]).size
            )
    return result


def session_partition_codes(
    dataset: np.ndarray,
    temporal: np.ndarray,
    row_sessions: np.ndarray,
    n_sessions: int,
) -> tuple[np.ndarray, int]:
    codes = np.zeros(n_sessions, dtype=np.uint8)
    conflicts = 0
    for code in (*ORDERED_PARTITIONS, PARTITION_EXCLUDED):
        selected_sessions = np.unique(
            row_sessions[(dataset == 0) & (temporal == code)]
        )
        conflicts += int(np.count_nonzero(codes[selected_sessions]))
        codes[selected_sessions] = code
    return codes, conflicts


def validate_existing_outputs(
    paths: RunPaths, allow_overwrite: bool
) -> list[str]:
    outputs = [
        paths.resolved_config,
        paths.normalization,
        paths.checkpoint,
        paths.predictions,
        paths.metrics,
        paths.audit,
    ]
    existing = [str(path) for path in outputs if path.exists()]
    audit_passed = False
    if paths.audit.exists():
        try:
            audit_passed = (
                json.loads(paths.audit.read_text(encoding="utf-8")).get(
                    "all_assertions_pass"
                )
                is True
            )
        except (OSError, json.JSONDecodeError):
            audit_passed = False
    if audit_passed and not allow_overwrite:
        raise FileExistsError(
            "a completed audited run already exists; choose another seed/root "
            "or pass --allow-overwrite explicitly"
        )
    return existing


def source_file_record(path: Path) -> dict[str, object]:
    return {
        "path": portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    run_started_at_utc = utc_now()
    run_start = time.perf_counter()
    formal_locked = validate_budget(
        args.run_purpose,
        args.epoch_samples,
        args.max_epochs,
        args.patience,
    )
    if args.batch_size < 1 or args.inference_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer hyperparameters")
    if args.model not in {"gru", "tcn"}:
        raise ValueError("strict-temporal learned baselines support GRU and TCN")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict-temporal learned baselines")

    paths = resolve_argument_paths(args)
    preexisting_outputs = validate_existing_outputs(
        paths, args.allow_overwrite
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    array_paths = {
        "values": args.array_dir / "sequence_values.npy",
        "masks": args.array_dir / "sequence_masks.npy",
        "targets": args.array_dir / "targets.npy",
        "elapsed": args.array_dir / "origin_offset_seconds.npy",
        "sport": args.array_dir / "sport_code.npy",
        "dataset": args.array_dir / "dataset_code.npy",
        "evaluation": args.array_dir / "evaluation_origin.npy",
        "users": args.array_dir / "user_index.npy",
        "sessions": args.array_dir / "session_index.npy",
        "original_temporal": args.array_dir / "temporal_partition.npy",
        "strict_temporal": args.temporal_partition,
        "session_table": args.array_dir / "sessions.csv",
        "strict_temporal_audit": args.temporal_audit,
    }
    for path in array_paths.values():
        require(path.exists(), f"missing input file {path}")

    arrays = {
        name: np.load(path, mmap_mode="r")
        for name, path in array_paths.items()
        if path.suffix == ".npy"
    }
    expected_rows = len(arrays["dataset"])
    row_counts = {name: int(len(value)) for name, value in arrays.items()}
    require(
        len(set(row_counts.values())) == 1,
        f"model-array length mismatch: {row_counts}",
    )
    require(
        arrays["values"].shape == (expected_rows, 30, 3),
        f"unexpected sequence-value shape: {arrays['values'].shape}",
    )
    require(
        arrays["masks"].shape == (expected_rows, 30, 3),
        f"unexpected sequence-mask shape: {arrays['masks'].shape}",
    )
    require(
        arrays["targets"].shape == (expected_rows, 3),
        f"unexpected target shape: {arrays['targets'].shape}",
    )

    strict_audit = json.loads(args.temporal_audit.read_text(encoding="utf-8"))
    require(
        bool(strict_audit.get("all_assertions_pass")),
        "strict temporal partition audit did not pass",
    )
    require(
        str(strict_audit.get("version")) == STRICT_PARTITION_VERSION,
        "strict temporal partition version mismatch",
    )
    require(
        strict_audit.get("rule") == STRICT_PARTITION_RULE,
        "strict temporal partition rule mismatch",
    )
    require(
        strict_audit.get("ordered_partition_codes") == list(ORDERED_PARTITIONS),
        "strict temporal ordered partition codes mismatch",
    )
    require(
        int(strict_audit.get("excluded_partition_code", -1))
        == PARTITION_EXCLUDED,
        "strict temporal excluded partition code mismatch",
    )
    require(
        int(strict_audit.get("strict_order_failures", -1)) == 0,
        "source strict temporal audit has ordering failures",
    )
    require(
        int(strict_audit.get("session_partition_consistency_failures", -1))
        == 0,
        "source strict temporal audit has session-partition conflicts",
    )
    require(
        int(strict_audit.get("external_rows_modified", -1)) == 0,
        "source strict temporal audit modified external rows",
    )

    strict_temporal = arrays["strict_temporal"]
    original_temporal = arrays["original_temporal"]
    dataset = arrays["dataset"]
    evaluation = arrays["evaluation"]
    users = arrays["users"]
    sessions = arrays["sessions"]
    unknown_endomondo_rows = int(
        np.count_nonzero(
            (dataset == 0)
            & ~np.isin(strict_temporal, (*ORDERED_PARTITIONS, PARTITION_EXCLUDED))
        )
    )
    require(
        unknown_endomondo_rows == 0,
        f"Endomondo rows without strict temporal assignment: {unknown_endomondo_rows}",
    )
    strict_counts = {
        str(code): int(np.count_nonzero((dataset == 0) & (strict_temporal == code)))
        for code in (*ORDERED_PARTITIONS, PARTITION_EXCLUDED)
    }
    require(
        strict_counts == strict_audit.get("counts_after"),
        f"strict partition counts disagree with audit: {strict_counts}",
    )
    changed_rows = int(np.count_nonzero(strict_temporal != original_temporal))
    external_rows_modified = int(
        np.count_nonzero(
            strict_temporal[dataset == 1] != original_temporal[dataset == 1]
        )
    )
    require(
        changed_rows == int(strict_audit.get("changed_rows", -1)),
        "strict partition changed-row count mismatch",
    )
    require(external_rows_modified == 0, "strict partition changed external rows")

    selected = split_indices(dataset, evaluation, strict_temporal)
    require(all(len(index) > 0 for index in selected.values()), "empty temporal split")
    expected_codes = {
        "train": PARTITION_TRAIN,
        "validation": PARTITION_VALIDATION,
        "calibration": PARTITION_CALIBRATION,
        "test": PARTITION_TEST,
    }
    filter_failures: dict[str, int] = {}
    for name, index in selected.items():
        failure = int(np.count_nonzero(dataset[index] != 0))
        failure += int(np.count_nonzero(strict_temporal[index] != expected_codes[name]))
        if name != "train":
            failure += int(np.count_nonzero(evaluation[index] != 1))
        filter_failures[name] = failure
    require(
        not any(filter_failures.values()),
        f"strict temporal split-filter failures: {filter_failures}",
    )

    session_sets = {
        name: np.unique(sessions[index]) for name, index in selected.items()
    }
    user_sets = {
        name: np.unique(users[index]) for name, index in selected.items()
    }
    session_overlaps = pairwise_overlap_counts(session_sets)
    user_overlaps = pairwise_overlap_counts(user_sets)
    require(
        not any(session_overlaps.values()),
        f"strict temporal session overlap: {session_overlaps}",
    )

    session_table = pd.read_csv(
        args.array_dir / "sessions.csv",
        dtype={"session_key": str},
        low_memory=False,
    )
    n_sessions = int(session_table["session_index"].max()) + 1
    session_codes, session_partition_conflicts = session_partition_codes(
        dataset, strict_temporal, sessions, n_sessions
    )
    require(
        session_partition_conflicts == 0,
        f"session partition conflicts: {session_partition_conflicts}",
    )
    independent_order_failures = validate_order(session_table, session_codes)
    require(
        independent_order_failures == 0,
        f"independent strict temporal ordering failures: {independent_order_failures}",
    )

    source_files = {
        name: source_file_record(path) for name, path in array_paths.items()
    }
    model = build_model(args.model).to(device)
    total_parameters = int(sum(item.numel() for item in model.parameters()))
    trainable_parameters = int(
        sum(item.numel() for item in model.parameters() if item.requires_grad)
    )
    resolved_config: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "model_version": ANALYSIS_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "protocol": "strict within-user temporal learned baseline",
        "run_purpose": args.run_purpose,
        "eligible_for_manuscript_results": args.run_purpose == "formal",
        "formal_budget_locked": formal_locked,
        "model": args.model,
        "model_parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "seed": args.seed,
        "output_paths": {
            name: portable_path(Path(path))
            for name, path in asdict(paths).items()
        },
        "data": {
            "array_dir": portable_path(args.array_dir),
            "temporal_partition": portable_path(args.temporal_partition),
            "temporal_audit": portable_path(args.temporal_audit),
            "source_files": source_files,
        },
        "hyperparameters": {
            "epoch_samples": args.epoch_samples,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "inference_batch_size": args.inference_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "smooth_l1_beta": args.smooth_l1_beta,
            "gradient_clip_norm": args.gradient_clip_norm,
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.5,
                "patience": 1,
                "min_lr": 1e-5,
            },
            "early_stopping_min_delta": 1e-4,
            "sampling": (
                "with replacement; equal users and sessions through "
                "hierarchical training weights"
            ),
            "loss": "SmoothL1 on residual from last observed heart rate",
        },
        "determinism": {
            "python_numpy_torch_seed": args.seed,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ[
                "CUBLAS_WORKSPACE_CONFIG"
            ],
        },
        "restart": {
            "preexisting_known_outputs": [
                portable_path(Path(item)) for item in preexisting_outputs
            ],
            "restarted_from_scratch": bool(preexisting_outputs),
            "completed_audit_overwrite_explicitly_allowed": (
                args.allow_overwrite
            ),
        },
    }
    atomic_json(paths.resolved_config, resolved_config)

    train_index = selected["train"]
    validation_index = selected["validation"]
    test_index = selected["test"]
    normalization = compute_normalization(
        arrays["values"], arrays["masks"], arrays["elapsed"], train_index
    )
    atomic_json(paths.normalization, normalization)
    training_weights = hierarchical_training_weights(
        np.asarray(users[train_index]), np.asarray(sessions[train_index])
    ).astype(np.float64)
    sampling_probabilities = training_weights / training_weights.sum()
    probability_sum_error = abs(float(sampling_probabilities.sum()) - 1.0)
    require(probability_sum_error < 1e-12, "invalid training probabilities")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    best_score = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    training_start = time.perf_counter()

    for epoch in range(args.max_epochs):
        epoch_start = time.perf_counter()
        model.train()
        sampled_positions = generator.choice(
            len(train_index),
            size=args.epoch_samples,
            replace=True,
            p=sampling_probabilities,
        )
        epoch_index = train_index[sampled_positions]
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
                normalization,
                device,
            )
            target = torch.from_numpy(
                np.asarray(arrays["targets"][index], dtype=np.float32)
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction_residual = model(
                    sequence, sport_batch, elapsed_batch
                )
                loss = F.smooth_l1_loss(
                    prediction_residual,
                    target - last_hr[:, None],
                    beta=args.smooth_l1_beta,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(index)
            examples += len(index)

        validation_predictions = predict_indices(
            model,
            validation_index,
            arrays["values"],
            arrays["masks"],
            arrays["elapsed"],
            arrays["sport"],
            normalization,
            device,
            args.inference_batch_size,
        )
        validation_horizon_mae: list[float] = []
        for position in range(3):
            metrics = hierarchical_metrics(
                validation_predictions[:, position],
                np.asarray(
                    arrays["targets"][validation_index, position],
                    dtype=np.float32,
                ),
                np.asarray(users[validation_index]),
                np.asarray(sessions[validation_index]),
            )
            validation_horizon_mae.append(float(metrics["mae_bpm"]))
        validation_score = float(np.mean(validation_horizon_mae))
        scheduler.step(validation_score)
        record: dict[str, float | int] = {
            "epoch": epoch + 1,
            "training_smooth_l1": loss_sum / examples,
            "validation_hierarchical_mae": validation_score,
            "validation_mae_60": validation_horizon_mae[0],
            "validation_mae_180": validation_horizon_mae[1],
            "validation_mae_300": validation_horizon_mae[2],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_wall_seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_score < best_score - 1e-4:
            best_score = validation_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            atomic_torch_save(
                paths.checkpoint,
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "architecture_version": ARCHITECTURE_VERSION,
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "seed": args.seed,
                    "epoch": best_epoch,
                    "validation_score": best_score,
                    "normalization": normalization,
                    "resolved_config_sha256": sha256_file(
                        paths.resolved_config
                    ),
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    training_wall_seconds = time.perf_counter() - training_start

    require(best_epoch > 0 and paths.checkpoint.exists(), "no checkpoint saved")
    saved = torch.load(paths.checkpoint, map_location=device, weights_only=False)
    require(saved["model_name"] == args.model, "checkpoint model mismatch")
    require(int(saved["seed"]) == args.seed, "checkpoint seed mismatch")
    model.load_state_dict(saved["model"])

    test_predictions = predict_indices(
        model,
        test_index,
        arrays["values"],
        arrays["masks"],
        arrays["elapsed"],
        arrays["sport"],
        normalization,
        device,
        args.inference_batch_size,
    )
    require(
        test_predictions.shape == (len(test_index), 3),
        f"test prediction shape mismatch: {test_predictions.shape}",
    )
    prediction_nonfinite = int((~np.isfinite(test_predictions)).sum())
    prediction_range_failures = int(
        ((test_predictions < 30.0) | (test_predictions > 240.0)).sum()
    )
    require(prediction_nonfinite == 0, "non-finite test predictions")
    require(prediction_range_failures == 0, "out-of-range test predictions")
    atomic_npz(
        paths.predictions,
        row_index=test_index.astype(np.int64),
        predictions=test_predictions,
    )

    metric_rows: list[dict[str, object]] = []
    for position, horizon in enumerate(HORIZONS):
        metrics = hierarchical_metrics(
            test_predictions[:, position],
            np.asarray(
                arrays["targets"][test_index, position], dtype=np.float32
            ),
            np.asarray(users[test_index]),
            np.asarray(sessions[test_index]),
        )
        metric_rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "model_version": ANALYSIS_VERSION,
                "architecture_version": ARCHITECTURE_VERSION,
                "protocol": "strict_temporal_train",
                "regime": "within_user_temporal_test",
                "run_purpose": args.run_purpose,
                "eligible_for_manuscript_results": (
                    args.run_purpose == "formal"
                ),
                "model": args.model,
                "seed": args.seed,
                "horizon_seconds": horizon,
                **metrics,
                "aggregation": (
                    "origin-within-session, session-within-user, "
                    "equal-user mean"
                ),
            }
        )
    atomic_csv(paths.metrics, metric_rows)
    numeric_metrics = np.asarray(
        [
            [row["mae_bpm"], row["rmse_bpm"], row["bias_bpm"]]
            for row in metric_rows
        ],
        dtype=np.float64,
    )
    metric_nonfinite = int((~np.isfinite(numeric_metrics)).sum())
    require(metric_nonfinite == 0, "non-finite test metrics")
    expected_test_support = {
        "users": int(len(user_sets["test"])),
        "sessions": int(len(session_sets["test"])),
        "origins": int(len(test_index)),
    }
    support_failures = int(
        sum(
            any(int(row[key]) != value for key, value in expected_test_support.items())
            for row in metric_rows
        )
    )
    require(support_failures == 0, "test metric support mismatch")

    total_wall_seconds = time.perf_counter() - run_start
    audit: dict[str, object] = {
        "run_started_at_utc": run_started_at_utc,
        "generated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "model_version": ANALYSIS_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "protocol": "strict within-user temporal learned baseline",
        "run_purpose": args.run_purpose,
        "eligible_for_manuscript_results": args.run_purpose == "formal",
        "formal_budget_locked": formal_locked,
        "model": args.model,
        "model_parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "restart": resolved_config["restart"],
        "input_integrity": {
            "source_files": source_files,
            "row_counts": row_counts,
        },
        "strict_temporal_control": {
            "partition_version": STRICT_PARTITION_VERSION,
            "audit_assertions_pass": True,
            "audit_counts_after": strict_audit["counts_after"],
            "recomputed_counts_after": strict_counts,
            "audit_changed_rows": strict_audit["changed_rows"],
            "recomputed_changed_rows": changed_rows,
            "audit_boundary_sessions_excluded": strict_audit[
                "excluded_boundary_sessions"
            ],
            "unknown_endomondo_partition_rows": unknown_endomondo_rows,
            "external_rows_modified": external_rows_modified,
            "session_partition_conflicts": session_partition_conflicts,
            "audit_order_failures": strict_audit["strict_order_failures"],
            "independent_order_failures": independent_order_failures,
            "split_filter_failures": filter_failures,
            "session_overlaps": session_overlaps,
            "user_overlaps_expected_and_allowed": user_overlaps,
            "training_uses_only_temporal_train": True,
            "model_selection_uses_only_temporal_validation": True,
            "calibration_partition_used_for_training_or_selection": False,
            "test_used_during_training_or_selection": False,
            "predictions_contain_only_temporal_test": True,
        },
        "split_support": {
            name: {
                "origins": int(len(index)),
                "sessions": int(len(session_sets[name])),
                "users": int(len(user_sets[name])),
            }
            for name, index in selected.items()
        },
        "resolved_hyperparameters": resolved_config["hyperparameters"],
        "normalization": {
            "fit_partition": "strict temporal train only",
            "path": portable_path(paths.normalization),
            "sha256": sha256_file(paths.normalization),
        },
        "training": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_hierarchical_mae": best_score,
            "history": history,
            "sampling_probability_sum_error": probability_sum_error,
            "training_wall_seconds": training_wall_seconds,
            "total_run_wall_seconds": total_wall_seconds,
            "peak_cuda_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
        },
        "outputs": {
            "run_dir": portable_path(paths.run_dir),
            "resolved_config": {
                "path": portable_path(paths.resolved_config),
                "sha256": sha256_file(paths.resolved_config),
            },
            "checkpoint": {
                "path": portable_path(paths.checkpoint),
                "sha256": sha256_file(paths.checkpoint),
            },
            "predictions": {
                "path": portable_path(paths.predictions),
                "sha256": sha256_file(paths.predictions),
                "rows": int(len(test_index)),
            },
            "metrics": {
                "path": portable_path(paths.metrics),
                "sha256": sha256_file(paths.metrics),
                "rows": len(metric_rows),
            },
        },
        "output_checks": {
            "prediction_nonfinite_values": prediction_nonfinite,
            "prediction_range_failures": prediction_range_failures,
            "metric_nonfinite_values": metric_nonfinite,
            "metric_support_failures": support_failures,
        },
        "limitations": [
            (
                "A smoke run validates code paths only and is explicitly "
                "ineligible for manuscript results."
            ),
            (
                "A formal single-seed run remains conditional on one stochastic "
                "optimization path; multi-seed aggregation is required for final "
                "model-comparison claims."
            ),
            (
                "The strict temporal protocol permits the same user in different "
                "partitions but prohibits session overlap and enforces ordered, "
                "non-overlapping session time boundaries."
            ),
        ],
        "all_assertions_pass": True,
    }
    atomic_json(paths.audit, audit)
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Train a GRU or point TCN under the leakage-controlled strict "
            "within-user temporal protocol."
        )
    )
    result.add_argument("--model", choices=("gru", "tcn"), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument(
        "--run-purpose", choices=("formal", "smoke"), default="formal"
    )
    result.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    result.add_argument(
        "--temporal-partition",
        type=Path,
        default=Path(
            "outputs/features/model_arrays_v0_6_0/temporal_partition_strict.npy"
        ),
    )
    result.add_argument(
        "--temporal-audit",
        type=Path,
        default=Path("outputs/audit/strict_temporal_partition_v0_13_0.json"),
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/strict_temporal_learned_baselines_v0_22_0"
        ),
    )
    result.add_argument("--batch-size", type=int, default=2048)
    result.add_argument("--inference-batch-size", type=int, default=8192)
    result.add_argument(
        "--epoch-samples", type=int, default=FORMAL_EPOCH_SAMPLES
    )
    result.add_argument("--max-epochs", type=int, default=FORMAL_MAX_EPOCHS)
    result.add_argument("--patience", type=int, default=FORMAL_PATIENCE)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--smooth-l1-beta", type=float, default=5.0)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--allow-overwrite", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(train(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
