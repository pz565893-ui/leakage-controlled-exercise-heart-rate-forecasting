from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.26.0"
HORIZONS = (60, 180, 300)
COVERAGES = (0.50, 0.80, 0.90)
PARTITION_CALIBRATION = 3
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1
HR_MIN = 30.0
HR_MAX = 240.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_sample_radius(scores: np.ndarray, coverage: float) -> float:
    """Return the split-conformal higher order statistic for absolute residuals."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("scores must be finite and nonnegative")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0, 1)")
    rank = int(math.ceil((values.size + 1) * coverage))
    rank = min(max(rank, 1), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


def last_observed_hr(
    sequence_values: np.ndarray,
    sequence_masks: np.ndarray,
    row_index: np.ndarray,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """Extract the most recent observed HR from each past-only 30-bin context."""
    selected = np.asarray(row_index, dtype=np.int64)
    result = np.empty(selected.size, dtype=np.float32)
    for start in range(0, selected.size, chunk_size):
        end = min(start + chunk_size, selected.size)
        index = selected[start:end]
        values = np.asarray(sequence_values[index, :, 0], dtype=np.float32)
        masks = np.asarray(sequence_masks[index, :, 0], dtype=bool)
        if np.any(~masks.any(axis=1)):
            raise AssertionError("eligible context without an observed HR value")
        reverse_position = np.argmax(masks[:, ::-1], axis=1)
        position = masks.shape[1] - 1 - reverse_position
        result[start:end] = values[np.arange(index.size), position]
    if not np.isfinite(result).all() or np.any((result < HR_MIN) | (result > HR_MAX)):
        raise AssertionError("invalid persistence prediction")
    return result


def hierarchical_average(values: np.ndarray, users: np.ndarray, sessions: np.ndarray) -> float:
    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=np.float64),
            "user": np.asarray(users),
            "session": np.asarray(sessions),
        }
    )
    session_values = frame.groupby(["user", "session"], sort=False)["value"].mean()
    user_values = session_values.groupby(level="user", sort=False).mean()
    return float(user_values.mean())


def interval_score(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: float,
) -> np.ndarray:
    alpha = 1.0 - coverage
    return (
        upper
        - lower
        + (2.0 / alpha) * (lower - target) * (target < lower)
        + (2.0 / alpha) * (target - upper) * (target > upper)
    )


def weighted_interval_score(
    point: np.ndarray,
    target: np.ndarray,
    bounds: dict[float, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    total = 0.5 * np.abs(target - point)
    for coverage, (lower, upper) in bounds.items():
        total = total + ((1.0 - coverage) / 2.0) * interval_score(
            target, lower, upper, coverage
        )
    return total / (len(bounds) + 0.5)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    array_dir = args.array_dir
    sequence_values = np.load(array_dir / "sequence_values.npy", mmap_mode="r")
    sequence_masks = np.load(array_dir / "sequence_masks.npy", mmap_mode="r")
    targets = np.load(array_dir / "targets.npy", mmap_mode="r")
    dataset = np.load(array_dir / "dataset_code.npy", mmap_mode="r")
    evaluation = np.load(array_dir / "evaluation_origin.npy", mmap_mode="r")
    strict = np.load(array_dir / "temporal_partition_strict.npy", mmap_mode="r")
    unseen = np.load(array_dir / "unseen_user_partition.npy", mmap_mode="r")
    external = np.load(array_dir / "primary_external_partition.npy", mmap_mode="r")
    users = np.load(array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(array_dir / "session_index.npy", mmap_mode="r")

    calibration_indices = {
        "strict_temporal": np.flatnonzero(
            (dataset == 0) & (evaluation == 1) & (strict == PARTITION_CALIBRATION)
        ),
        "unseen_user": np.flatnonzero(
            (dataset == 0) & (evaluation == 1) & (unseen == PARTITION_CALIBRATION)
        ),
    }
    regime_indices = {
        "strict_temporal_test": np.flatnonzero(
            (dataset == 0) & (evaluation == 1) & (strict == PARTITION_TEST)
        ),
        "unseen_user_test": np.flatnonzero(
            (dataset == 0) & (evaluation == 1) & (unseen == PARTITION_TEST)
        ),
        "goldencheetah_frozen_external": np.flatnonzero(
            (dataset == 1) & (evaluation == 1) & (external == EXTERNAL_FROZEN)
        ),
    }
    if any(index.size == 0 for index in (*calibration_indices.values(), *regime_indices.values())):
        raise AssertionError("empty calibration or evaluation regime")

    radii: dict[str, dict[str, dict[str, float]]] = {}
    calibration_predictions: dict[str, np.ndarray] = {}
    for protocol, index in calibration_indices.items():
        prediction = last_observed_hr(sequence_values, sequence_masks, index, args.chunk_size)
        calibration_predictions[protocol] = prediction
        radii[protocol] = {}
        for horizon_position, horizon in enumerate(HORIZONS):
            score = np.abs(
                np.asarray(targets[index, horizon_position], dtype=np.float32) - prediction
            )
            radii[protocol][str(horizon)] = {
                str(coverage): finite_sample_radius(score, coverage)
                for coverage in COVERAGES
            }

    rows: list[dict[str, object]] = []
    regime_protocol = {
        "strict_temporal_test": "strict_temporal",
        "unseen_user_test": "unseen_user",
        "goldencheetah_frozen_external": "unseen_user",
    }
    for regime, index in regime_indices.items():
        protocol = regime_protocol[regime]
        prediction = last_observed_hr(sequence_values, sequence_masks, index, args.chunk_size)
        selected_users = np.asarray(users[index])
        selected_sessions = np.asarray(sessions[index])
        for horizon_position, horizon in enumerate(HORIZONS):
            target = np.asarray(targets[index, horizon_position], dtype=np.float32)
            bounds: dict[float, tuple[np.ndarray, np.ndarray]] = {}
            row: dict[str, object] = {
                "analysis_version": ANALYSIS_VERSION,
                "model": "persistence_symmetric_split_conformal",
                "regime": regime,
                "calibration_source": f"{protocol}_calibration",
                "horizon_seconds": horizon,
                "mae_bpm": hierarchical_average(
                    np.abs(target - prediction), selected_users, selected_sessions
                ),
                "users": int(np.unique(selected_users).size),
                "sessions": int(np.unique(selected_sessions).size),
                "origins": int(index.size),
            }
            for coverage in COVERAGES:
                radius = radii[protocol][str(horizon)][str(coverage)]
                lower = np.clip(prediction - radius, HR_MIN, HR_MAX)
                upper = np.clip(prediction + radius, HR_MIN, HR_MAX)
                bounds[coverage] = (lower, upper)
                label = int(round(coverage * 100))
                row[f"radius_{label}_bpm"] = radius
                row[f"picp_{label}"] = hierarchical_average(
                    ((target >= lower) & (target <= upper)).astype(np.float32),
                    selected_users,
                    selected_sessions,
                )
                row[f"width_{label}_bpm"] = hierarchical_average(
                    upper - lower, selected_users, selected_sessions
                )
            row["weighted_interval_score"] = hierarchical_average(
                weighted_interval_score(prediction, target, bounds),
                selected_users,
                selected_sessions,
            )
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    strict_session_overlap = int(
        np.intersect1d(
            sessions[calibration_indices["strict_temporal"]],
            sessions[regime_indices["strict_temporal_test"]],
        ).size
    )
    unseen_user_overlap = int(
        np.intersect1d(
            users[calibration_indices["unseen_user"]],
            users[regime_indices["unseen_user_test"]],
        ).size
    )
    threshold_monotonic_failures = 0
    for protocol_values in radii.values():
        for horizon_values in protocol_values.values():
            ordered = [horizon_values[str(coverage)] for coverage in COVERAGES]
            threshold_monotonic_failures += int(any(a > b for a, b in zip(ordered, ordered[1:])))
    numeric = np.asarray(
        [
            [
                row["mae_bpm"],
                row["picp_50"],
                row["picp_80"],
                row["picp_90"],
                row["width_50_bpm"],
                row["width_80_bpm"],
                row["width_90_bpm"],
                row["weighted_interval_score"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    external_calibration_rows = int(
        sum(np.count_nonzero(dataset[index] == 1) for index in calibration_indices.values())
    )
    audit: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "method": (
            "Deterministic persistence point forecast with origin-pooled finite-sample "
            "symmetric split-conformal absolute-residual radii; intervals clipped to 30--240 bpm."
        ),
        "radii": radii,
        "calibration_rows": {
            key: int(value.size) for key, value in calibration_indices.items()
        },
        "evaluation_rows": {key: int(value.size) for key, value in regime_indices.items()},
        "strict_calibration_test_session_overlap": strict_session_overlap,
        "unseen_calibration_test_user_overlap": unseen_user_overlap,
        "external_calibration_rows": external_calibration_rows,
        "goldencheetah_used_for_calibration": external_calibration_rows > 0,
        "threshold_monotonic_failures": threshold_monotonic_failures,
        "nonfinite_metric_values": int((~np.isfinite(numeric)).sum()),
        "coverage_range_failures": int(
            np.count_nonzero((numeric[:, 1:4] < 0.0) | (numeric[:, 1:4] > 1.0))
        ),
        "metric_rows": len(rows),
        "output": "outputs/results/persistence_conformal_baseline_v0_26_0.csv",
    }
    audit["all_assertions_pass"] = (
        strict_session_overlap == 0
        and unseen_user_overlap == 0
        and external_calibration_rows == 0
        and threshold_monotonic_failures == 0
        and audit["nonfinite_metric_values"] == 0
        and audit["coverage_range_failures"] == 0
    )
    atomic_json(args.audit, audit)
    if not audit["all_assertions_pass"]:
        raise AssertionError(json.dumps(audit, indent=2))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an independent split-conformal persistence interval baseline."
    )
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
