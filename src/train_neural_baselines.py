from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_xgboost_baseline import (
    EXTERNAL_FROZEN,
    HORIZONS,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    hierarchical_metrics,
    hierarchical_training_weights,
)


MODEL_VERSION = "0.9.0"
SEED = 20260722


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_normalization(
    values: np.ndarray,
    masks: np.ndarray,
    elapsed: np.ndarray,
    train_index: np.ndarray,
    chunk_size: int = 100_000,
) -> dict[str, list[float] | float]:
    value_sum = np.zeros(3, dtype=np.float64)
    value_sum_sq = np.zeros(3, dtype=np.float64)
    value_count = np.zeros(3, dtype=np.float64)
    elapsed_sum = 0.0
    elapsed_sum_sq = 0.0
    elapsed_count = 0
    for start in range(0, len(train_index), chunk_size):
        index = train_index[start : start + chunk_size]
        value_chunk = np.asarray(values[index], dtype=np.float32)
        mask_chunk = np.asarray(masks[index], dtype=np.float32)
        value_sum += (value_chunk * mask_chunk).sum(axis=(0, 1), dtype=np.float64)
        value_sum_sq += (
            value_chunk * value_chunk * mask_chunk
        ).sum(axis=(0, 1), dtype=np.float64)
        value_count += mask_chunk.sum(axis=(0, 1), dtype=np.float64)
        log_elapsed = np.log1p(np.asarray(elapsed[index], dtype=np.float64))
        elapsed_sum += float(log_elapsed.sum())
        elapsed_sum_sq += float((log_elapsed * log_elapsed).sum())
        elapsed_count += len(index)
    value_mean = value_sum / value_count
    value_std = np.sqrt(np.maximum(value_sum_sq / value_count - value_mean**2, 1e-6))
    elapsed_mean = elapsed_sum / elapsed_count
    elapsed_std = math.sqrt(max(elapsed_sum_sq / elapsed_count - elapsed_mean**2, 1e-6))
    return {
        "value_mean": value_mean.tolist(),
        "value_std": value_std.tolist(),
        "log_elapsed_mean": elapsed_mean,
        "log_elapsed_std": elapsed_std,
        "observed_value_counts": value_count.astype(np.int64).tolist(),
    }


def last_observed_hr(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    hr_mask = masks[:, :, 0].astype(bool, copy=False)
    last_index = hr_mask.shape[1] - 1 - hr_mask[:, ::-1].argmax(axis=1)
    return np.take_along_axis(values[:, :, 0], last_index[:, None], axis=1)[:, 0]


def prepare_batch(
    values: np.ndarray,
    masks: np.ndarray,
    elapsed: np.ndarray,
    sport: np.ndarray,
    index: np.ndarray,
    normalization: dict[str, list[float] | float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    value_chunk = np.asarray(values[index], dtype=np.float32)
    mask_chunk = np.asarray(masks[index], dtype=np.float32)
    mean = np.asarray(normalization["value_mean"], dtype=np.float32)
    standard_deviation = np.asarray(normalization["value_std"], dtype=np.float32)
    standardized = ((value_chunk - mean) / standard_deviation) * mask_chunk
    sequence = np.concatenate([standardized, mask_chunk], axis=2)
    log_elapsed = np.log1p(np.asarray(elapsed[index], dtype=np.float32))
    log_elapsed = (
        log_elapsed - float(normalization["log_elapsed_mean"])
    ) / float(normalization["log_elapsed_std"])
    return (
        torch.from_numpy(sequence).to(device, non_blocking=True),
        torch.from_numpy(np.asarray(sport[index], dtype=np.int64)).to(
            device, non_blocking=True
        ),
        torch.from_numpy(log_elapsed[:, None]).to(device, non_blocking=True),
        torch.from_numpy(last_observed_hr(value_chunk, mask_chunk)).to(
            device, non_blocking=True
        ),
    )


class FusionHead(nn.Module):
    def __init__(self, representation_size: int, sport_size: int = 8) -> None:
        super().__init__()
        self.sport_embedding = nn.Embedding(8, sport_size)
        self.layers = nn.Sequential(
            nn.Linear(representation_size + sport_size + 1, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(
        self, representation: torch.Tensor, sport: torch.Tensor, elapsed: torch.Tensor
    ) -> torch.Tensor:
        return self.layers(
            torch.cat([representation, self.sport_embedding(sport), elapsed], dim=1)
        )


class GRUForecast(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )
        self.head = FusionHead(64)

    def forward(
        self, sequence: torch.Tensor, sport: torch.Tensor, elapsed: torch.Tensor
    ) -> torch.Tensor:
        _, hidden = self.encoder(sequence)
        return self.head(hidden[-1], sport, elapsed)


class CausalConv1d(nn.Conv1d):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        left_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(input_tensor, (left_padding, 0)))


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.normalization(input_tensor.transpose(1, 2)).transpose(1, 2)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, 3, dilation=dilation)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = ChannelLayerNorm(channels)
        self.norm2 = ChannelLayerNorm(channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.dropout(F.gelu(self.norm1(self.conv1(input_tensor))))
        output = self.dropout(self.norm2(self.conv2(output)))
        return F.gelu(input_tensor + output)


class TCNForecast(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(6, 64, 1)
        self.blocks = nn.Sequential(
            TemporalBlock(64, 1),
            TemporalBlock(64, 2),
            TemporalBlock(64, 4),
            TemporalBlock(64, 8),
        )
        self.head = FusionHead(64)

    def forward(
        self, sequence: torch.Tensor, sport: torch.Tensor, elapsed: torch.Tensor
    ) -> torch.Tensor:
        representation = self.blocks(self.input_projection(sequence.transpose(1, 2)))[:, :, -1]
        return self.head(representation, sport, elapsed)


class TransformerForecast(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(6, 64)
        self.position = nn.Parameter(torch.zeros(1, 30, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(64)
        self.head = FusionHead(64)

    def forward(
        self, sequence: torch.Tensor, sport: torch.Tensor, elapsed: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(sequence) + self.position)
        return self.head(self.norm(encoded[:, -1]), sport, elapsed)


def build_model(name: str) -> nn.Module:
    if name == "gru":
        return GRUForecast()
    if name == "tcn":
        return TCNForecast()
    if name == "transformer":
        return TransformerForecast()
    raise ValueError(name)


@torch.inference_mode()
def predict_indices(
    model: nn.Module,
    indices: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    elapsed: np.ndarray,
    sport: np.ndarray,
    normalization: dict[str, list[float] | float],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = np.empty((len(indices), 3), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        end = min(len(indices), start + batch_size)
        index = indices[start:end]
        sequence, sport_batch, elapsed_batch, last_hr = prepare_batch(
            values, masks, elapsed, sport, index, normalization, device
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            residual = model(sequence, sport_batch, elapsed_batch)
        prediction = torch.clamp(residual.float() + last_hr[:, None], 30.0, 240.0)
        output[start:end] = prediction.cpu().numpy()
    return output


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked neural-baseline run")
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
    train_index = np.flatnonzero((dataset == 0) & (unseen == PARTITION_TRAIN))
    validation_index = np.flatnonzero(
        (dataset == 0) & (unseen == PARTITION_VALIDATION) & (evaluation == 1)
    )
    train_users = np.unique(users[train_index])
    validation_users = np.unique(users[validation_index])
    overlap = int(np.intersect1d(train_users, validation_users).size)
    if overlap:
        raise AssertionError("train-validation user overlap")
    normalization_path = args.output_dir / "normalization_unseen_user_train.json"
    if normalization_path.exists():
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    else:
        normalization = compute_normalization(values, masks, elapsed, train_index)
        atomic_json(normalization_path, normalization)
    sampling_weights = hierarchical_training_weights(
        np.asarray(users[train_index]), np.asarray(sessions[train_index])
    ).astype(np.float64)
    sampling_probabilities = sampling_weights / sampling_weights.sum()
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{args.model}_best_v0_9_0.pt"
    history: list[dict[str, float | int]] = []
    best_score = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in range(args.max_epochs):
        model.train()
        epoch_positions = generator.choice(
            len(train_index),
            size=args.epoch_samples,
            replace=True,
            p=sampling_probabilities,
        )
        epoch_indices = train_index[epoch_positions]
        loss_sum = 0.0
        examples = 0
        for start in range(0, len(epoch_indices), args.batch_size):
            index = np.sort(epoch_indices[start : start + args.batch_size])
            sequence, sport_batch, elapsed_batch, last_hr = prepare_batch(
                values, masks, elapsed, sport, index, normalization, device
            )
            target = torch.from_numpy(
                np.asarray(targets[index], dtype=np.float32)
            ).to(device, non_blocking=True)
            target_residual = target - last_hr[:, None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction_residual = model(sequence, sport_batch, elapsed_batch)
                loss = F.smooth_l1_loss(
                    prediction_residual, target_residual, beta=5.0
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(index)
            examples += len(index)
        validation_predictions = predict_indices(
            model,
            validation_index,
            values,
            masks,
            elapsed,
            sport,
            normalization,
            device,
            args.inference_batch_size,
        )
        horizon_mae = []
        for position in range(3):
            metric = hierarchical_metrics(
                validation_predictions[:, position],
                np.asarray(targets[validation_index, position], dtype=np.float32),
                np.asarray(users[validation_index]),
                np.asarray(sessions[validation_index]),
            )
            horizon_mae.append(float(metric["mae_bpm"]))
        validation_score = sum(horizon_mae) / len(horizon_mae)
        scheduler.step(validation_score)
        record = {
            "epoch": epoch + 1,
            "training_smooth_l1": loss_sum / examples,
            "validation_hierarchical_mae": validation_score,
            "validation_mae_60": horizon_mae[0],
            "validation_mae_180": horizon_mae[1],
            "validation_mae_300": horizon_mae[2],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_score < best_score - 1e-4:
            best_score = validation_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "normalization": normalization,
                    "epoch": best_epoch,
                    "validation_score": best_score,
                },
                checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    evaluation_index = np.flatnonzero(
        ((dataset == 0) & (evaluation == 1))
        | ((dataset == 1) & (external == EXTERNAL_FROZEN))
    )
    predictions = predict_indices(
        model,
        evaluation_index,
        values,
        masks,
        elapsed,
        sport,
        normalization,
        device,
        args.inference_batch_size,
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        row_index=evaluation_index.astype(np.int64),
        predictions=predictions,
    )
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
    metric_rows: list[dict[str, object]] = []
    for regime, subset_mask in regime_masks.items():
        subset_rows = evaluation_index[subset_mask]
        for position, horizon in enumerate(HORIZONS):
            metric_rows.append(
                {
                    "model_version": MODEL_VERSION,
                    "protocol": "unseen_user_train",
                    "regime": regime,
                    "model": args.model,
                    "horizon_seconds": horizon,
                    **hierarchical_metrics(
                        predictions[subset_mask, position],
                        np.asarray(targets[subset_rows, position], dtype=np.float32),
                        np.asarray(users[subset_rows]),
                        np.asarray(sessions[subset_rows]),
                    ),
                    "aggregation": "origin-within-session, session-within-user, equal-user mean",
                }
            )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    finite_failures = int((~np.isfinite(predictions)).sum())
    range_failures = int(((predictions < 30) | (predictions > 240)).sum())
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "model_version": MODEL_VERSION,
        "model": args.model,
        "protocol": "unseen_user_train",
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "training_rows_available": int(len(train_index)),
        "training_users": int(len(train_users)),
        "validation_rows": int(len(validation_index)),
        "validation_users": int(len(validation_users)),
        "train_validation_user_overlap": overlap,
        "epoch_samples": args.epoch_samples,
        "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "best_epoch": best_epoch,
        "best_validation_hierarchical_mae": best_score,
        "history": history,
        "checkpoint": str(checkpoint),
        "prediction_rows": int(len(evaluation_index)),
        "prediction_nonfinite_values": finite_failures,
        "prediction_range_failures": range_failures,
        "metric_rows": len(metric_rows),
        "normalization_file": str(normalization_path),
    }
    payload["all_assertions_pass"] = (
        overlap == 0 and finite_failures == 0 and range_failures == 0
    )
    atomic_json(args.audit, payload)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Train common-input neural baselines.")
    parser.add_argument("--model", choices=("gru", "tcn", "transformer"), required=True)
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--epoch-samples", type=int, default=1_000_000)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
