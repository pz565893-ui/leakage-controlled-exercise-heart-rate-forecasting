from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_uncertainty_model import (
    HistoryQuantileTCN,
    predict_quantiles,
    uncertainty_rows,
)
from train_xgboost_baseline import HORIZONS, PARTITION_TEST, hierarchical_metrics


MODEL_VERSION = "0.15.0"


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    values = np.load(args.array_dir / "sequence_values.npy", mmap_mode="r")
    masks = np.load(args.array_dir / "sequence_masks.npy", mmap_mode="r")
    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    elapsed = np.load(args.array_dir / "origin_offset_seconds.npy", mmap_mode="r")
    sport = np.load(args.array_dir / "sport_code.npy", mmap_mode="r")
    dataset = np.load(args.array_dir / "dataset_code.npy", mmap_mode="r")
    evaluation = np.load(args.array_dir / "evaluation_origin.npy", mmap_mode="r")
    unseen = np.load(args.array_dir / "unseen_user_partition.npy", mmap_mode="r")
    users = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(args.array_dir / "session_index.npy", mmap_mode="r")
    history_values = np.load(
        args.array_dir / "session_history_values.npy", mmap_mode="r"
    )
    history_mask = np.load(
        args.array_dir / "session_history_mask.npy", mmap_mode="r"
    )
    dense_index = np.flatnonzero((dataset == 0) & (unseen == PARTITION_TEST))
    standard_index = np.flatnonzero(
        (dataset == 0) & (unseen == PARTITION_TEST) & (evaluation == 1)
    )
    if not np.all(np.isin(standard_index, dense_index)):
        raise AssertionError("standard evaluation rows are not a subset of dense rows")

    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HistoryQuantileTCN().to(device)
    model.load_state_dict(saved["model"])
    input_normalization = saved["input_normalization"]
    history_normalization = saved["history_normalization"]
    common = (
        model,
        dense_index,
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
        args.batch_size,
    )
    history_prediction = predict_quantiles(*common, False, "multimodal")
    zero_prediction = predict_quantiles(*common, True, "multimodal")
    threshold_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
    thresholds = threshold_payload["thresholds"]
    dense_targets = np.asarray(targets[dense_index])
    point_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    for mode, prediction in {
        "history_informed": history_prediction,
        "zero_history": zero_prediction,
    }.items():
        for position, horizon in enumerate(HORIZONS):
            point_rows.append(
                {
                    "model_version": MODEL_VERSION,
                    "regime": "unseen_user_test_dense_60s_origins",
                    "mode": mode,
                    "horizon_seconds": horizon,
                    **hierarchical_metrics(
                        prediction[:, position, 3],
                        dense_targets[:, position],
                        np.asarray(users[dense_index]),
                        np.asarray(sessions[dense_index]),
                    ),
                }
            )
        interval_rows.extend(
            uncertainty_rows(
                "unseen_user_test_dense_60s_origins",
                mode,
                prediction,
                dense_targets,
                np.asarray(users[dense_index]),
                np.asarray(sessions[dense_index]),
                thresholds[mode],
                model_version=MODEL_VERSION,
            )
        )
    args.point_output.parent.mkdir(parents=True, exist_ok=True)
    with args.point_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    args.interval_output.parent.mkdir(parents=True, exist_ok=True)
    with args.interval_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0]))
        writer.writeheader()
        writer.writerows(interval_rows)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        row_index=dense_index.astype(np.int64),
        history_quantiles=history_prediction,
        zero_history_quantiles=zero_prediction,
    )
    crossing = int(
        (np.diff(history_prediction, axis=2) < -1e-6).sum()
        + (np.diff(zero_prediction, axis=2) < -1e-6).sum()
    )
    nonfinite = int(
        (~np.isfinite(history_prediction)).sum()
        + (~np.isfinite(zero_prediction)).sum()
    )
    payload: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "source_checkpoint": str(args.checkpoint),
        "sensitivity": "evaluation origin stride 60 seconds instead of primary 300 seconds",
        "dense_rows": int(len(dense_index)),
        "standard_rows": int(len(standard_index)),
        "row_multiplier": float(len(dense_index) / len(standard_index)),
        "users": int(len(np.unique(users[dense_index]))),
        "sessions": int(len(np.unique(sessions[dense_index]))),
        "standard_rows_subset_of_dense": True,
        "quantile_crossing_failures": crossing,
        "nonfinite_prediction_values": nonfinite,
        "point_rows": len(point_rows),
        "interval_rows": len(interval_rows),
        "all_assertions_pass": crossing == 0 and nonfinite == 0,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--thresholds", type=Path, required=True)
    result.add_argument("--point-output", type=Path, required=True)
    result.add_argument("--interval-output", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=4096)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parser().parse_args())))
