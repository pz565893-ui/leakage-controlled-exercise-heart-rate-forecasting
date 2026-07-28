from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_neural_baselines import (
    MODEL_VERSION as NEURAL_VERSION,
    SEED,
    TemporalBlock,
    atomic_json,
    compute_normalization,
    last_observed_hr,
    prepare_batch,
    set_seed,
    utc_now,
)
from train_xgboost_baseline import (
    EXTERNAL_FROZEN,
    HORIZONS,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    hierarchical_metrics,
    hierarchical_training_weights,
)


MODEL_VERSION = "0.11.0"
PARTITION_CALIBRATION = 3
QUANTILES = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float32)
INTERVALS = {
    0.50: (2, 4),
    0.80: (1, 5),
    0.90: (0, 6),
}
TRAINING_HISTORY_MODES = ("mixed", "always_zero")
MIXED_SELECTION_METRIC = (
    "mean history-informed and zero-history validation hierarchical MAE"
)
ALWAYS_ZERO_SELECTION_METRIC = (
    "mean 1/3/5-min zero-history validation hierarchical MAE"
)


def selection_metric_name(training_history_mode: str) -> str:
    if training_history_mode == "mixed":
        return MIXED_SELECTION_METRIC
    if training_history_mode == "always_zero":
        return ALWAYS_ZERO_SELECTION_METRIC
    raise ValueError(f"unsupported training history mode: {training_history_mode}")


def validation_selection_score(
    *,
    training_history_mode: str,
    history_mae: float | None,
    zero_history_mae: float,
) -> float:
    if training_history_mode == "always_zero":
        return float(zero_history_mae)
    if training_history_mode == "mixed" and history_mae is not None:
        return float((history_mae + zero_history_mae) / 2.0)
    raise ValueError("mixed selection requires a history-informed validation score")


class HistoryQuantileTCN(nn.Module):
    def __init__(self, history_features: int = 13) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(6, 64, 1)
        self.blocks = nn.Sequential(
            TemporalBlock(64, 1),
            TemporalBlock(64, 2),
            TemporalBlock(64, 4),
            TemporalBlock(64, 8),
        )
        self.sport_embedding = nn.Embedding(8, 8)
        self.history_encoder = nn.Sequential(
            nn.Linear(history_features, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.no_history_embedding = nn.Parameter(torch.zeros(32))
        self.head = nn.Sequential(
            nn.Linear(64 + 8 + 1 + 32 + 1, 96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, 3 * len(QUANTILES)),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        sequence: torch.Tensor,
        sport: torch.Tensor,
        elapsed: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        current = self.blocks(self.input_projection(sequence.transpose(1, 2)))[:, :, -1]
        encoded_history = self.history_encoder(history)
        history_representation = (
            encoded_history * history_mask
            + self.no_history_embedding[None, :] * (1.0 - history_mask)
        )
        fused = torch.cat(
            [
                current,
                self.sport_embedding(sport),
                elapsed,
                history_representation,
                history_mask,
            ],
            dim=1,
        )
        raw = self.head(fused).view(-1, 3, len(QUANTILES))
        return torch.sort(raw, dim=2).values


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.as_tensor(QUANTILES, device=prediction.device)[None, None, :]
    error = target[:, :, None] - prediction
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


def apply_signal_input(
    sequence: torch.Tensor, signal_input: str
) -> torch.Tensor:
    if signal_input == "multimodal":
        return sequence
    if signal_input != "heart_rate_only":
        raise ValueError(f"unsupported signal input: {signal_input}")
    channel_mask = torch.as_tensor(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        dtype=sequence.dtype,
        device=sequence.device,
    )
    return sequence * channel_mask[None, None, :]


def compute_history_normalization(
    history_values: np.ndarray,
    history_mask: np.ndarray,
    train_session_index: np.ndarray,
) -> dict[str, list[float]]:
    sessions = np.unique(train_session_index)
    sessions = sessions[history_mask[sessions] == 1]
    selected = np.asarray(history_values[sessions], dtype=np.float64)
    mean = selected.mean(axis=0)
    standard_deviation = selected.std(axis=0)
    standard_deviation = np.maximum(standard_deviation, 1e-6)
    return {
        "history_mean": mean.tolist(),
        "history_std": standard_deviation.tolist(),
        "training_history_sessions": int(len(sessions)),
    }


def prepare_history_batch(
    history_values: np.ndarray,
    history_mask: np.ndarray,
    session_index: np.ndarray,
    row_index: np.ndarray,
    normalization: dict[str, list[float] | int],
    device: torch.device,
    force_zero_history: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    sessions = np.asarray(session_index[row_index], dtype=np.int64)
    values = np.asarray(history_values[sessions], dtype=np.float32)
    mask = np.asarray(history_mask[sessions], dtype=np.float32)
    if force_zero_history:
        mask.fill(0.0)
    mean = np.asarray(normalization["history_mean"], dtype=np.float32)
    standard_deviation = np.asarray(normalization["history_std"], dtype=np.float32)
    values = ((values - mean) / standard_deviation) * mask[:, None]
    return (
        torch.from_numpy(values).to(device, non_blocking=True),
        torch.from_numpy(mask[:, None]).to(device, non_blocking=True),
    )


@torch.inference_mode()
def predict_quantiles(
    model: nn.Module,
    indices: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    elapsed: np.ndarray,
    sport: np.ndarray,
    session_index: np.ndarray,
    history_values: np.ndarray,
    history_mask: np.ndarray,
    input_normalization: dict[str, list[float] | float],
    history_normalization: dict[str, list[float] | int],
    device: torch.device,
    batch_size: int,
    force_zero_history: bool,
    signal_input: str = "multimodal",
) -> np.ndarray:
    model.eval()
    output = np.empty((len(indices), 3, len(QUANTILES)), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        end = min(len(indices), start + batch_size)
        index = indices[start:end]
        sequence, sport_batch, elapsed_batch, last_hr = prepare_batch(
            values, masks, elapsed, sport, index, input_normalization, device
        )
        sequence = apply_signal_input(sequence, signal_input)
        history, history_batch_mask = prepare_history_batch(
            history_values,
            history_mask,
            session_index,
            index,
            history_normalization,
            device,
            force_zero_history,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            residual = model(
                sequence,
                sport_batch,
                elapsed_batch,
                history,
                history_batch_mask,
            )
        prediction = torch.clamp(
            residual.float() + last_hr[:, None, None], 30.0, 240.0
        )
        output[start:end] = torch.sort(prediction, dim=2).values.cpu().numpy()
    return output


def conformal_thresholds(
    quantile_predictions: np.ndarray, targets: np.ndarray
) -> dict[str, list[float]]:
    thresholds: dict[str, list[float]] = {}
    n = len(targets)
    for coverage, (lower_position, upper_position) in INTERVALS.items():
        alpha = 1.0 - coverage
        probability = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
        horizon_thresholds = []
        for horizon_position in range(3):
            lower = quantile_predictions[:, horizon_position, lower_position]
            upper = quantile_predictions[:, horizon_position, upper_position]
            target = targets[:, horizon_position]
            score = np.maximum(lower - target, target - upper)
            qhat = float(np.quantile(score, probability, method="higher"))
            horizon_thresholds.append(max(0.0, qhat))
        thresholds[str(coverage)] = horizon_thresholds
    return thresholds


def hierarchical_average(
    values: np.ndarray, users: np.ndarray, sessions: np.ndarray
) -> tuple[float, int, int, int]:
    import pandas as pd

    frame = pd.DataFrame(
        {"value": values, "user": users, "session": sessions}
    )
    session = frame.groupby(["user", "session"], sort=False)["value"].mean()
    user = session.groupby(level="user", sort=False).mean()
    return float(user.mean()), int(len(user)), int(len(session)), int(len(values))


def uncertainty_rows(
    regime: str,
    mode: str,
    quantile_predictions: np.ndarray,
    targets: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
    thresholds: dict[str, list[float]],
    model_version: str = MODEL_VERSION,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for coverage, (lower_position, upper_position) in INTERVALS.items():
        for calibrated in (False, True):
            for horizon_position, horizon in enumerate(HORIZONS):
                adjustment = thresholds[str(coverage)][horizon_position] if calibrated else 0.0
                lower = np.clip(
                    quantile_predictions[:, horizon_position, lower_position] - adjustment,
                    30.0,
                    240.0,
                )
                upper = np.clip(
                    quantile_predictions[:, horizon_position, upper_position] + adjustment,
                    30.0,
                    240.0,
                )
                covered = ((targets[:, horizon_position] >= lower) & (targets[:, horizon_position] <= upper)).astype(np.float32)
                width = upper - lower
                picp, n_users, n_sessions, n_origins = hierarchical_average(
                    covered, users, sessions
                )
                mean_width, _, _, _ = hierarchical_average(width, users, sessions)
                rows.append(
                    {
                        "model_version": model_version,
                        "regime": regime,
                        "mode": mode,
                        "horizon_seconds": horizon,
                        "nominal_coverage": coverage,
                        "calibrated": calibrated,
                        "picp": picp,
                        "absolute_coverage_error": abs(picp - coverage),
                        "mean_interval_width_bpm": mean_width,
                        "conformal_adjustment_bpm": adjustment,
                        "users": n_users,
                        "sessions": n_sessions,
                        "origins": n_origins,
                    }
                )
    return rows


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    training_history_mode = getattr(args, "training_history_mode", "mixed")
    selection_metric = selection_metric_name(training_history_mode)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    array_dir = args.array_dir
    values = np.load(array_dir / "sequence_values.npy", mmap_mode="r")
    masks = np.load(array_dir / "sequence_masks.npy", mmap_mode="r")
    targets = np.load(array_dir / "targets.npy", mmap_mode="r")
    elapsed = np.load(array_dir / "origin_offset_seconds.npy", mmap_mode="r")
    sport = np.load(array_dir / "sport_code.npy", mmap_mode="r")
    dataset = np.load(array_dir / "dataset_code.npy", mmap_mode="r")
    evaluation = np.load(array_dir / "evaluation_origin.npy", mmap_mode="r")
    unseen = np.load(array_dir / "unseen_user_partition.npy", mmap_mode="r")
    external = np.load(array_dir / "primary_external_partition.npy", mmap_mode="r")
    users = np.load(array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(array_dir / "session_index.npy", mmap_mode="r")
    history_values = np.load(array_dir / "session_history_values.npy", mmap_mode="r")
    history_mask = np.load(array_dir / "session_history_mask.npy", mmap_mode="r")
    history_metadata = json.loads(
        (array_dir / "history_metadata.json").read_text(encoding="utf-8")
    )
    if history_metadata.get("strict_rule") != (
        "only sessions completed at or before the current session_start_time"
    ):
        raise AssertionError("causal history does not enforce completed-session timing")
    train_index = np.flatnonzero((dataset == 0) & (unseen == PARTITION_TRAIN))
    validation_index = np.flatnonzero(
        (dataset == 0) & (unseen == PARTITION_VALIDATION) & (evaluation == 1)
    )
    calibration_index = np.flatnonzero(
        (dataset == 0) & (unseen == PARTITION_CALIBRATION) & (evaluation == 1)
    )
    train_users = np.unique(users[train_index])
    validation_users = np.unique(users[validation_index])
    calibration_users = np.unique(users[calibration_index])
    overlaps = {
        "train_validation": int(np.intersect1d(train_users, validation_users).size),
        "train_calibration": int(np.intersect1d(train_users, calibration_users).size),
        "validation_calibration": int(
            np.intersect1d(validation_users, calibration_users).size
        ),
    }
    if any(overlaps.values()):
        raise AssertionError(f"user overlap: {overlaps}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_norm_path = args.output_dir / "normalization_unseen_user_train.json"
    if input_norm_path.exists():
        input_normalization = json.loads(input_norm_path.read_text(encoding="utf-8"))
    else:
        input_normalization = compute_normalization(values, masks, elapsed, train_index)
        atomic_json(input_norm_path, input_normalization)
    history_normalization = compute_history_normalization(
        history_values, history_mask, np.asarray(sessions[train_index])
    )
    history_norm_path = args.output_dir / "history_normalization_unseen_user_train.json"
    atomic_json(history_norm_path, history_normalization)
    weights = hierarchical_training_weights(
        np.asarray(users[train_index]), np.asarray(sessions[train_index])
    ).astype(np.float64)
    probabilities = weights / weights.sum()
    model = HistoryQuantileTCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = np.random.default_rng(args.seed)
    checkpoint = args.output_dir / "history_quantile_tcn_best_v0_11_0.pt"
    history_records: list[dict[str, object]] = []
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
                values,
                masks,
                elapsed,
                sport,
                index,
                input_normalization,
                device,
            )
            sequence = apply_signal_input(sequence, args.signal_input)
            history_batch, history_batch_mask = prepare_history_batch(
                history_values,
                history_mask,
                sessions,
                index,
                history_normalization,
                device,
                force_zero_history=training_history_mode == "always_zero",
            )
            if training_history_mode == "mixed":
                dropout = torch.rand_like(history_batch_mask) < args.history_dropout
                history_batch_mask = history_batch_mask * (~dropout)
                history_batch = history_batch * history_batch_mask
            target = torch.from_numpy(np.asarray(targets[index], dtype=np.float32)).to(
                device, non_blocking=True
            )
            target_residual = target - last_hr[:, None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(
                    sequence,
                    sport_batch,
                    elapsed_batch,
                    history_batch,
                    history_batch_mask,
                )
                loss = pinball_loss(prediction, target_residual)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(index)
            examples += len(index)
        validation_history = None
        if training_history_mode == "mixed":
            validation_history = predict_quantiles(
                model,
                validation_index,
                values,
                masks,
                elapsed,
                sport,
                sessions,
                history_values,
                history_mask,
                input_normalization,
                history_normalization,
                device,
                args.inference_batch_size,
                False,
                args.signal_input,
            )
        validation_zero = predict_quantiles(
            model,
            validation_index,
            values,
            masks,
            elapsed,
            sport,
            sessions,
            history_values,
            history_mask,
            input_normalization,
            history_normalization,
            device,
            args.inference_batch_size,
            True,
            args.signal_input,
        )
        def validation_mae(prediction: np.ndarray) -> float:
            horizon_scores = [
                float(
                    hierarchical_metrics(
                        prediction[:, position, 3],
                        np.asarray(targets[validation_index, position]),
                        np.asarray(users[validation_index]),
                        np.asarray(sessions[validation_index]),
                    )["mae_bpm"]
                )
                for position in range(3)
            ]
            return float(sum(horizon_scores) / 3)

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
        history_records.append(record)
        print(json.dumps(record), flush=True)
        if score < best_score - 1e-4:
            best_score = score
            best_epoch = epoch + 1
            no_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "input_normalization": input_normalization,
                    "history_normalization": history_normalization,
                    "epoch": best_epoch,
                    "validation_score": best_score,
                    "validation_history_mae": validation_history_mae,
                    "validation_zero_history_mae": validation_zero_history_mae,
                    "training_history_mode": training_history_mode,
                    "selection_metric": selection_metric,
                    "quantiles": QUANTILES.tolist(),
                },
                checkpoint,
            )
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    development_evaluation = (dataset == 0) & (evaluation == 1)
    if args.development_only:
        evaluation_index = np.flatnonzero(development_evaluation)
    else:
        evaluation_index = np.flatnonzero(
            development_evaluation
            | ((dataset == 1) & (external == EXTERNAL_FROZEN))
        )
    history_prediction = None
    if training_history_mode == "mixed":
        history_prediction = predict_quantiles(
            model,
            evaluation_index,
            values,
            masks,
            elapsed,
            sport,
            sessions,
            history_values,
            history_mask,
            input_normalization,
            history_normalization,
            device,
            args.inference_batch_size,
            False,
            args.signal_input,
        )
    zero_prediction = predict_quantiles(
        model,
        evaluation_index,
        values,
        masks,
        elapsed,
        sport,
        sessions,
        history_values,
        history_mask,
        input_normalization,
        history_normalization,
        device,
        args.inference_batch_size,
        True,
        args.signal_input,
    )
    calibration_mask = (dataset[evaluation_index] == 0) & (
        unseen[evaluation_index] == PARTITION_CALIBRATION
    )
    calibration_targets = np.asarray(targets[evaluation_index[calibration_mask]])
    thresholds = {
        "zero_history": conformal_thresholds(
            zero_prediction[calibration_mask], calibration_targets
        ),
    }
    if history_prediction is not None:
        thresholds["history_informed"] = conformal_thresholds(
            history_prediction[calibration_mask], calibration_targets
        )
    thresholds_path = args.output_dir / "conformal_thresholds_v0_11_0.json"
    atomic_json(
        thresholds_path,
        {
            "generated_at_utc": utc_now(),
            "calibration_rows": int(calibration_mask.sum()),
            "calibration_users": int(len(calibration_users)),
            "rule": "finite-sample higher quantile; nonnegative expansion only",
            "training_history_mode": training_history_mode,
            "selection_metric": selection_metric,
            "thresholds": thresholds,
        },
    )
    resolved_config_path = args.output_dir / "resolved_config.json"
    atomic_json(
        resolved_config_path,
        {
            "analysis_version": "0.23.0" if training_history_mode == "always_zero" else MODEL_VERSION,
            "model_version": MODEL_VERSION,
            "seed": args.seed,
            "protocol": "unseen_user",
            "training_history_mode": training_history_mode,
            "selection_metric": selection_metric,
            "epoch_samples": args.epoch_samples,
            "batch_size": args.batch_size,
            "inference_batch_size": args.inference_batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "history_dropout": args.history_dropout,
            "signal_input": args.signal_input,
            "development_only": args.development_only,
        },
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    if training_history_mode == "always_zero":
        saved_prediction_mask = (dataset[evaluation_index] == 0) & (
            unseen[evaluation_index] == PARTITION_TEST
        )
        saved_prediction_rows = evaluation_index[saved_prediction_mask]
        np.savez_compressed(
            args.predictions,
            row_index=saved_prediction_rows.astype(np.int64),
            zero_history_quantiles=zero_prediction[saved_prediction_mask],
        )
    else:
        saved_prediction_rows = evaluation_index
        np.savez_compressed(
            args.predictions,
            row_index=evaluation_index.astype(np.int64),
            history_quantiles=history_prediction,
            zero_history_quantiles=zero_prediction,
        )
    regimes: dict[str, np.ndarray] = {
        "unseen_user_validation": (dataset[evaluation_index] == 0)
        & (unseen[evaluation_index] == PARTITION_VALIDATION),
        "unseen_user_test": (dataset[evaluation_index] == 0)
        & (unseen[evaluation_index] == PARTITION_TEST),
    }
    if not args.development_only:
        regimes["goldencheetah_frozen_external"] = (
            (dataset[evaluation_index] == 1)
            & (external[evaluation_index] == EXTERNAL_FROZEN)
        )
        for code, family in (
            (1, "outdoor_cycling"),
            (2, "indoor_virtual_cycling"),
            (3, "running"),
        ):
            regimes[f"goldencheetah_external__{family}"] = (
                (dataset[evaluation_index] == 1)
                & (external[evaluation_index] == EXTERNAL_FROZEN)
                & (sport[evaluation_index] == code)
            )
    point_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    for regime, regime_mask in regimes.items():
        subset = evaluation_index[regime_mask]
        mode_predictions = {
            "zero_history": zero_prediction[regime_mask],
        }
        if history_prediction is not None and not regime.startswith("goldencheetah"):
            mode_predictions["history_informed"] = history_prediction[regime_mask]
        for mode, prediction in mode_predictions.items():
            subset_targets = np.asarray(targets[subset])
            for position, horizon in enumerate(HORIZONS):
                point_rows.append(
                    {
                        "model_version": MODEL_VERSION,
                        "regime": regime,
                        "mode": mode,
                        "model": "history_quantile_tcn",
                        "horizon_seconds": horizon,
                        **hierarchical_metrics(
                            prediction[:, position, 3],
                            subset_targets[:, position],
                            np.asarray(users[subset]),
                            np.asarray(sessions[subset]),
                        ),
                        "aggregation": "origin-within-session, session-within-user, equal-user mean",
                    }
                )
            interval_rows.extend(
                uncertainty_rows(
                    regime,
                    mode,
                    prediction,
                    subset_targets,
                    np.asarray(users[subset]),
                    np.asarray(sessions[subset]),
                    thresholds[mode],
                )
            )
    args.point_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.point_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    args.uncertainty_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.uncertainty_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0]))
        writer.writeheader()
        writer.writerows(interval_rows)
    integrity_predictions = [zero_prediction]
    if history_prediction is not None:
        integrity_predictions.append(history_prediction)
    crossing_failures = int(
        sum((np.diff(prediction, axis=2) < -1e-6).sum() for prediction in integrity_predictions)
    )
    finite_failures = int(
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
        "neural_components_version": NEURAL_VERSION,
        "model": "history_quantile_tcn",
        "protocol": "unseen_user",
        "quantiles": QUANTILES.tolist(),
        "seed": args.seed,
        "training_history_mode": training_history_mode,
        "selection_metric": selection_metric,
        "training_and_validation_history_forced_zero": (
            training_history_mode == "always_zero"
        ),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "training_rows_available": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "calibration_rows": int(len(calibration_index)),
        "user_overlap_counts": overlaps,
        "history_dropout": args.history_dropout,
        "signal_input": args.signal_input,
        "development_only": args.development_only,
        "external_inference_performed": not args.development_only,
        "causal_history_version": history_metadata["history_version"],
        "causal_history_strict_rule": history_metadata["strict_rule"],
        "epoch_samples": args.epoch_samples,
        "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "best_epoch": best_epoch,
        "best_validation_composite_mae": best_score,
        "best_validation_history_mae": saved.get("validation_history_mae"),
        "best_validation_zero_history_mae": saved.get(
            "validation_zero_history_mae"
        ),
        "history": history_records,
        "prediction_rows": int(len(saved_prediction_rows)),
        "quantile_crossing_failures": crossing_failures,
        "prediction_nonfinite_values": finite_failures,
        "prediction_range_failures": range_failures,
        "point_metric_rows": len(point_rows),
        "uncertainty_metric_rows": len(interval_rows),
        "checkpoint": str(checkpoint),
        "thresholds_file": str(thresholds_path),
        "resolved_config": str(resolved_config_path),
    }
    payload["all_assertions_pass"] = (
        not any(overlaps.values())
        and crossing_failures == 0
        and finite_failures == 0
        and range_failures == 0
    )
    atomic_json(args.audit, payload)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the history-conditioned uncertainty-aware TCN."
    )
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--point-metrics", type=Path, required=True)
    parser.add_argument("--uncertainty-metrics", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--epoch-samples", type=int, default=500_000)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--history-dropout", type=float, default=0.2)
    parser.add_argument(
        "--training-history-mode",
        choices=TRAINING_HISTORY_MODES,
        default="mixed",
        help=(
            "mixed preserves the dual-mode training/selection rule; always_zero "
            "forces a zero history mask in every training and validation batch and "
            "selects only on zero-history validation MAE."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help=(
            "Train, calibrate, and evaluate on Endomondo only. Use the separate "
            "frozen-external inference program after writing a freeze record."
        ),
    )
    parser.add_argument(
        "--signal-input",
        choices=("multimodal", "heart_rate_only"),
        default="multimodal",
    )
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
