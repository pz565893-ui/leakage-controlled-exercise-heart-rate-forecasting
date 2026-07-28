from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.27.0"
SEEDS = (20260722, 20260723, 20260724)
HORIZONS = (60, 180, 300)
SPORTS = {
    "outdoor_cycling": 1,
    "indoor_virtual_cycling": 2,
    "running": 3,
    "walking_hiking": 4,
    "strength_cross_training": 7,
}
PARTITION_TEST = 4
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260723


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def aligned_positions(reference_rows: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_rows, dtype=np.int64)
    query = np.asarray(query_rows, dtype=np.int64)
    if reference.ndim != 1 or query.ndim != 1:
        raise ValueError("row-index arrays must be one-dimensional")
    if np.any(reference[1:] <= reference[:-1]):
        raise AssertionError("reference row indices must be strictly increasing")
    positions = np.searchsorted(reference, query)
    if np.any(positions >= reference.size) or not np.array_equal(reference[positions], query):
        raise AssertionError("query rows are not an exact subset of reference rows")
    return positions


def per_user_metrics(
    target: np.ndarray,
    full_prediction: np.ndarray,
    held_prediction: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.DataFrame:
    if not (
        target.shape == full_prediction.shape == held_prediction.shape
        and target.ndim == 2
        and target.shape[1] == len(HORIZONS)
    ):
        raise ValueError("target and prediction arrays must share shape (n, 3)")
    frames = []
    for position, horizon in enumerate(HORIZONS):
        frame = pd.DataFrame(
            {
                "user": users,
                "session": sessions,
                "full_absolute_error": np.abs(
                    target[:, position] - full_prediction[:, position]
                ),
                "held_absolute_error": np.abs(
                    target[:, position] - held_prediction[:, position]
                ),
            }
        )
        session_values = frame.groupby(["user", "session"], sort=False)[
            ["full_absolute_error", "held_absolute_error"]
        ].mean()
        user_values = session_values.groupby(level="user", sort=False).mean().reset_index()
        user_values["horizon_seconds"] = horizon
        user_values["delta_mae_bpm"] = (
            user_values.held_absolute_error - user_values.full_absolute_error
        )
        frames.append(user_values)
    return pd.concat(frames, ignore_index=True)


def bootstrap_mean_ci(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0 or not np.isfinite(sample).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(replicates, sample.size))
    bootstrap = sample[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(sample.mean()), float(low), float(high)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    dataset = np.load(args.array_dir / "dataset_code.npy", mmap_mode="r")
    unseen = np.load(args.array_dir / "unseen_user_partition.npy", mmap_mode="r")
    sport = np.load(args.array_dir / "sport_code.npy", mmap_mode="r")
    users = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(args.array_dir / "session_index.npy", mmap_mode="r")

    output_rows: list[dict[str, object]] = []
    audit_support: dict[str, dict[str, int]] = {}
    alignment_failures = 0
    boundary_failures = 0
    prediction_failures = 0

    for family, sport_code in SPORTS.items():
        seed_user_frames = []
        expected_rows: np.ndarray | None = None
        family_support: dict[str, int] | None = None
        for seed in SEEDS:
            held_path = (
                args.multiseed_root
                / f"seed_{seed}"
                / "held_sport"
                / family
                / "predictions.npz"
            )
            full_path = (
                args.multiseed_root
                / f"seed_{seed}"
                / "unseen_main"
                / "development_predictions.npz"
            )
            with np.load(held_path) as held, np.load(full_path) as full:
                held_rows = np.asarray(held["row_index"], dtype=np.int64)
                same_user_rows = int(np.asarray(held["same_user_rows"]).item())
                joint_rows = held_rows[same_user_rows:]
                full_rows = np.asarray(full["row_index"], dtype=np.int64)
                try:
                    full_positions = aligned_positions(full_rows, joint_rows)
                except AssertionError:
                    alignment_failures += 1
                    raise
                if expected_rows is None:
                    expected_rows = joint_rows.copy()
                elif not np.array_equal(expected_rows, joint_rows):
                    alignment_failures += 1
                    raise AssertionError(f"joint rows changed across seeds for {family}")

                held_prediction = np.asarray(
                    held["zero_history_quantiles"][same_user_rows:, :, 3],
                    dtype=np.float32,
                )
                full_prediction = np.asarray(
                    full["zero_history_quantiles"][full_positions, :, 3],
                    dtype=np.float32,
                )
                selected_target = np.asarray(targets[joint_rows], dtype=np.float32)
                selected_users = np.asarray(users[joint_rows])
                selected_sessions = np.asarray(sessions[joint_rows])

                boundary_failure = int(
                    np.count_nonzero(dataset[joint_rows] != 0)
                    + np.count_nonzero(unseen[joint_rows] != PARTITION_TEST)
                    + np.count_nonzero(sport[joint_rows] != sport_code)
                )
                boundary_failures += boundary_failure
                prediction_failure = int(
                    np.count_nonzero(~np.isfinite(held_prediction))
                    + np.count_nonzero(~np.isfinite(full_prediction))
                    + np.count_nonzero((held_prediction < 30.0) | (held_prediction > 240.0))
                    + np.count_nonzero((full_prediction < 30.0) | (full_prediction > 240.0))
                )
                prediction_failures += prediction_failure
                if boundary_failure or prediction_failure:
                    raise AssertionError(f"prediction boundary failure for {family}/{seed}")

                support = {
                    "origins": int(joint_rows.size),
                    "sessions": int(np.unique(selected_sessions).size),
                    "users": int(np.unique(selected_users).size),
                }
                if family_support is None:
                    family_support = support
                elif family_support != support:
                    raise AssertionError(f"support changed across seeds for {family}")

                user_frame = per_user_metrics(
                    selected_target,
                    full_prediction,
                    held_prediction,
                    selected_users,
                    selected_sessions,
                )
                user_frame["seed"] = seed
                seed_user_frames.append(user_frame)

        if family_support is None:
            raise AssertionError(f"missing support for {family}")
        audit_support[family] = family_support
        combined = pd.concat(seed_user_frames, ignore_index=True)
        averaged = (
            combined.groupby(["user", "horizon_seconds"], sort=False)[
                ["full_absolute_error", "held_absolute_error", "delta_mae_bpm"]
            ]
            .mean()
            .reset_index()
        )
        for horizon in HORIZONS:
            selected = averaged[averaged.horizon_seconds == horizon]
            estimate, ci_low, ci_high = bootstrap_mean_ci(
                selected.delta_mae_bpm.to_numpy(),
                BOOTSTRAP_REPLICATES,
                BOOTSTRAP_SEED + sport_code * 1000 + horizon,
            )
            output_rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "analysis_status": "post_hoc_matched_origin_sensitivity",
                    "regime": "joint_unseen_user_sport",
                    "information_mode": "history_masked",
                    "held_sport_family": family,
                    "horizon_seconds": horizon,
                    "seeds": len(SEEDS),
                    "full_sport_mae_bpm": float(selected.full_absolute_error.mean()),
                    "held_sport_mae_bpm": float(selected.held_absolute_error.mean()),
                    "held_minus_full_delta_mae_bpm": estimate,
                    "ci_low_bpm": ci_low,
                    "ci_high_bpm": ci_high,
                    "users": family_support["users"],
                    "sessions": family_support["sessions"],
                    "origins": family_support["origins"],
                    "users_with_higher_held_error_percent": float(
                        100.0 * np.mean(selected.delta_mae_bpm.to_numpy() > 0.0)
                    ),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_unit": "user_after_within_session_aggregation_and_seed_averaging",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    numeric = np.asarray(
        [
            [
                row["full_sport_mae_bpm"],
                row["held_sport_mae_bpm"],
                row["held_minus_full_delta_mae_bpm"],
                row["ci_low_bpm"],
                row["ci_high_bpm"],
            ]
            for row in output_rows
        ],
        dtype=np.float64,
    )
    audit: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "analysis_status": "post_hoc_matched_origin_sensitivity",
        "estimand": (
            "history-masked held-family model minus history-masked full-sport model "
            "MAE on identical joint unseen-user/sport origins"
        ),
        "interpretation_limit": (
            "Operational sport-availability contrast, not a causal sport effect; the models "
            "also differ in token exposure, sport-excluded fitting data, and locked training budget."
        ),
        "seeds": list(SEEDS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "support": audit_support,
        "alignment_failures": alignment_failures,
        "boundary_failures": boundary_failures,
        "prediction_failures": prediction_failures,
        "nonfinite_result_values": int(np.count_nonzero(~np.isfinite(numeric))),
        "metric_rows": len(output_rows),
        "output": "outputs/results/matched_sport_availability_v0_27_0.csv",
    }
    audit["all_assertions_pass"] = (
        alignment_failures == 0
        and boundary_failures == 0
        and prediction_failures == 0
        and audit["nonfinite_result_values"] == 0
        and len(output_rows) == len(SPORTS) * len(HORIZONS)
    )
    atomic_json(args.audit, audit)
    if not audit["all_assertions_pass"]:
        raise AssertionError(json.dumps(audit, indent=2))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Matched-origin full-sport versus held-sport frozen-prediction sensitivity."
    )
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--multiseed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
