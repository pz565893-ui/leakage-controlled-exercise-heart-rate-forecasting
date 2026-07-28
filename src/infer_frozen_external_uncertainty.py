from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from train_uncertainty_model import (
    HistoryQuantileTCN,
    MODEL_VERSION,
    predict_quantiles,
    uncertainty_rows,
)
from train_xgboost_baseline import EXTERNAL_FROZEN, HORIZONS, hierarchical_metrics


ANALYSIS_VERSION = "0.21.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no output rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_freeze_record(
    path: Path,
    checkpoint: Path,
    thresholds: Path,
    input_normalization: Path,
    history_normalization: Path,
) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "frozen_before_external_inference":
        raise AssertionError("freeze record is not finalized")
    if record.get("external_outcomes_used_for_selection") is not False:
        raise AssertionError("freeze record does not exclude external outcome selection")
    expected = {
        "checkpoint": checkpoint,
        "thresholds": thresholds,
        "input_normalization": input_normalization,
        "history_normalization": history_normalization,
    }
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise AssertionError("freeze record has no artifact manifest")
    for name, artifact_path in expected.items():
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise AssertionError(f"freeze record is missing {name}")
        if Path(str(entry.get("path"))).resolve() != artifact_path.resolve():
            raise AssertionError(f"freeze record path mismatch for {name}")
        if entry.get("sha256") != sha256_file(artifact_path):
            raise AssertionError(f"freeze record hash mismatch for {name}")
    source_code = record.get("source_code")
    if not isinstance(source_code, dict):
        raise AssertionError("freeze record has no source-code manifest")
    current_sources = {
        "training_script": Path(__file__).resolve().with_name(
            "train_uncertainty_model.py"
        ),
        "external_inference_script": Path(__file__).resolve(),
    }
    for name, source_path in current_sources.items():
        entry = source_code.get(name)
        if not isinstance(entry, dict):
            raise AssertionError(f"freeze record is missing source code {name}")
        if Path(str(entry.get("path"))).resolve() != source_path:
            raise AssertionError(f"freeze source path mismatch for {name}")
        if entry.get("sha256") != sha256_file(source_path):
            raise AssertionError(f"source code changed after freeze for {name}")
    return record


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    freeze = validate_freeze_record(
        args.freeze_record,
        args.checkpoint,
        args.thresholds,
        args.input_normalization,
        args.history_normalization,
    )
    arrays = {
        "values": np.load(args.array_dir / "sequence_values.npy", mmap_mode="r"),
        "masks": np.load(args.array_dir / "sequence_masks.npy", mmap_mode="r"),
        "targets": np.load(args.array_dir / "targets.npy", mmap_mode="r"),
        "elapsed": np.load(
            args.array_dir / "origin_offset_seconds.npy", mmap_mode="r"
        ),
        "sport": np.load(args.array_dir / "sport_code.npy", mmap_mode="r"),
        "dataset": np.load(args.array_dir / "dataset_code.npy", mmap_mode="r"),
        "external": np.load(
            args.array_dir / "primary_external_partition.npy", mmap_mode="r"
        ),
        "users": np.load(args.array_dir / "user_index.npy", mmap_mode="r"),
        "sessions": np.load(args.array_dir / "session_index.npy", mmap_mode="r"),
        "history_values": np.load(
            args.array_dir / "session_history_values.npy", mmap_mode="r"
        ),
        "history_mask": np.load(
            args.array_dir / "session_history_mask.npy", mmap_mode="r"
        ),
    }
    lengths = {name: len(value) for name, value in arrays.items() if name != "history_values" and name != "history_mask"}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"row-array length mismatch: {lengths}")
    index = np.flatnonzero(
        (arrays["dataset"] == 1) & (arrays["external"] == EXTERNAL_FROZEN)
    )
    if len(index) != 531_725:
        raise AssertionError(f"unexpected frozen external support: {len(index)}")

    device = torch.device("cuda")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HistoryQuantileTCN().to(device)
    model.load_state_dict(saved["model"])
    input_normalization = json.loads(
        args.input_normalization.read_text(encoding="utf-8")
    )
    history_normalization = json.loads(
        args.history_normalization.read_text(encoding="utf-8")
    )
    thresholds_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
    thresholds = thresholds_payload["thresholds"]["zero_history"]

    prediction = predict_quantiles(
        model,
        index,
        arrays["values"],
        arrays["masks"],
        arrays["elapsed"],
        arrays["sport"],
        arrays["sessions"],
        arrays["history_values"],
        arrays["history_mask"],
        input_normalization,
        history_normalization,
        device,
        args.inference_batch_size,
        True,
        args.signal_input,
    )
    regimes = {
        "goldencheetah_frozen_external": np.ones(len(index), dtype=bool),
        "goldencheetah_external__outdoor_cycling": arrays["sport"][index] == 1,
        "goldencheetah_external__indoor_virtual_cycling": arrays["sport"][index] == 2,
        "goldencheetah_external__running": arrays["sport"][index] == 3,
    }
    point_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    for regime, mask in regimes.items():
        subset = index[mask]
        subset_prediction = prediction[mask]
        subset_targets = np.asarray(arrays["targets"][subset])
        if len(subset) == 0:
            raise AssertionError(f"empty external regime: {regime}")
        for position, horizon in enumerate(HORIZONS):
            point_rows.append(
                {
                    "model_version": MODEL_VERSION,
                    "analysis_version": ANALYSIS_VERSION,
                    "seed": int(freeze["seed"]),
                    "regime": regime,
                    "mode": "zero_history",
                    "model": "history_quantile_tcn",
                    "horizon_seconds": horizon,
                    **hierarchical_metrics(
                        subset_prediction[:, position, 3],
                        subset_targets[:, position],
                        np.asarray(arrays["users"][subset]),
                        np.asarray(arrays["sessions"][subset]),
                    ),
                    "aggregation": (
                        "origin-within-session, session-within-user, equal-user mean"
                    ),
                }
            )
        interval_rows.extend(
            {
                **row,
                "analysis_version": ANALYSIS_VERSION,
                "seed": int(freeze["seed"]),
            }
            for row in uncertainty_rows(
                regime,
                "zero_history",
                subset_prediction,
                subset_targets,
                np.asarray(arrays["users"][subset]),
                np.asarray(arrays["sessions"][subset]),
                thresholds,
            )
        )

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.predictions,
        row_index=index.astype(np.int64),
        zero_history_quantiles=prediction,
    )
    atomic_csv(args.point_metrics, point_rows)
    atomic_csv(args.interval_metrics, interval_rows)
    crossing = int((np.diff(prediction, axis=2) < -1e-6).sum())
    nonfinite = int((~np.isfinite(prediction)).sum())
    range_failures = int(((prediction < 30) | (prediction > 240)).sum())
    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": ANALYSIS_VERSION,
        "model_version": MODEL_VERSION,
        "protocol": "frozen GoldenCheetah external inference after development freeze",
        "seed": int(freeze["seed"]),
        "freeze_record": str(args.freeze_record),
        "freeze_record_sha256": sha256_file(args.freeze_record),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "thresholds_sha256": sha256_file(args.thresholds),
        "external_rows": int(len(index)),
        "external_users": int(len(np.unique(arrays["users"][index]))),
        "external_sessions": int(len(np.unique(arrays["sessions"][index]))),
        "signal_input": args.signal_input,
        "inference_batch_size": args.inference_batch_size,
        "device": torch.cuda.get_device_name(0),
        "external_adaptation_or_recalibration": False,
        "prediction_nonfinite_values": nonfinite,
        "prediction_range_failures": range_failures,
        "quantile_crossing_failures": crossing,
        "outputs": {
            "predictions": str(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
            "point_metrics": str(args.point_metrics),
            "point_metrics_sha256": sha256_file(args.point_metrics),
            "interval_metrics": str(args.interval_metrics),
            "interval_metrics_sha256": sha256_file(args.interval_metrics),
        },
        "all_assertions_pass": (
            crossing == 0 and nonfinite == 0 and range_failures == 0
        ),
    }
    atomic_json(args.audit, payload)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen GoldenCheetah inference after a signed freeze record."
    )
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--input-normalization", type=Path, required=True)
    parser.add_argument("--history-normalization", type=Path, required=True)
    parser.add_argument("--freeze-record", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--point-metrics", type=Path, required=True)
    parser.add_argument("--interval-metrics", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument(
        "--signal-input",
        choices=("multimodal", "heart_rate_only"),
        default="multimodal",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
