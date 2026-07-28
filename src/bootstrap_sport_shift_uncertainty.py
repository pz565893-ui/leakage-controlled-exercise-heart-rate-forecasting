from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.17.0"
SOURCE_MODEL_VERSION = "0.12.0"
NOMINAL_COVERAGE = 0.90
LOWER_QUANTILE_POSITION = 0
UPPER_QUANTILE_POSITION = 6
HORIZONS = (60, 180, 300)
PARTITION_TRAIN = 1
PARTITION_TEST = 4
SPORTS = {
    1: "outdoor_cycling",
    2: "indoor_virtual_cycling",
    3: "running",
    4: "walking_hiking",
    7: "strength_cross_training",
}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def user_level_metrics(
    covered: np.ndarray,
    width: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "covered": covered,
            "width": width,
            "user": users,
            "session": sessions,
        }
    )
    session_level = frame.groupby(["user", "session"], sort=False)[
        ["covered", "width"]
    ].mean()
    return session_level.groupby(level="user", sort=False).mean()


def paired_user_bootstrap(
    user_values: np.ndarray,
    replicates: int,
    seed: int,
    batch_size: int = 2_000,
) -> np.ndarray:
    if user_values.ndim != 2 or user_values.shape[1] != 2:
        raise ValueError("expected an n-user by two-metric array")
    if len(user_values) < 2:
        raise ValueError("at least two users are required for bootstrap inference")
    generator = np.random.default_rng(seed)
    result = np.empty((replicates, 2), dtype=np.float64)
    for start in range(0, replicates, batch_size):
        end = min(replicates, start + batch_size)
        indices = generator.integers(
            0, len(user_values), size=(end - start, len(user_values))
        )
        result[start:end] = user_values[indices].mean(axis=1)
    return result


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def analyze(args: argparse.Namespace) -> dict[str, object]:
    arrays = {
        "targets": np.load(args.array_dir / "targets.npy", mmap_mode="r"),
        "dataset": np.load(args.array_dir / "dataset_code.npy", mmap_mode="r"),
        "evaluation": np.load(
            args.array_dir / "evaluation_origin.npy", mmap_mode="r"
        ),
        "unseen": np.load(
            args.array_dir / "unseen_user_partition.npy", mmap_mode="r"
        ),
        "sport": np.load(args.array_dir / "sport_code.npy", mmap_mode="r"),
        "users": np.load(args.array_dir / "user_index.npy", mmap_mode="r"),
        "sessions": np.load(args.array_dir / "session_index.npy", mmap_mode="r"),
    }
    row_counts = {name: int(len(array)) for name, array in arrays.items()}
    require(len(set(row_counts.values())) == 1, f"array row mismatch: {row_counts}")
    total_rows = next(iter(row_counts.values()))

    reference = pd.read_csv(args.reference_csv)
    required_reference_columns = {
        "model_version",
        "regime",
        "mode",
        "horizon_seconds",
        "nominal_coverage",
        "calibrated",
        "picp",
        "absolute_coverage_error",
        "mean_interval_width_bpm",
        "conformal_adjustment_bpm",
        "users",
        "sessions",
        "origins",
    }
    require(
        required_reference_columns.issubset(reference.columns),
        "reference interval CSV is missing required columns",
    )
    reference["calibrated"] = (
        reference["calibrated"].astype(str).str.lower() == "true"
    )

    output_rows: list[dict[str, object]] = []
    family_audits: dict[str, object] = {}
    maximum_reference_delta = {
        "picp": 0.0,
        "absolute_coverage_error": 0.0,
        "mean_interval_width_bpm": 0.0,
        "conformal_adjustment_bpm": 0.0,
    }

    for sport_code, family in SPORTS.items():
        prediction_path = (
            args.prediction_dir / f"sport_shift_{family}_v0_12_0.npz"
        )
        threshold_path = (
            args.model_dir / family / "conformal_thresholds.json"
        )
        require(prediction_path.exists(), f"missing {prediction_path}")
        require(threshold_path.exists(), f"missing {threshold_path}")

        with np.load(prediction_path) as predictions:
            required_keys = {
                "row_index",
                "same_user_rows",
                "history_quantiles",
                "zero_history_quantiles",
            }
            require(
                required_keys.issubset(predictions.files),
                f"{family}: prediction keys {predictions.files}",
            )
            row_index = np.asarray(predictions["row_index"], dtype=np.int64)
            same_user_marker = np.asarray(predictions["same_user_rows"])
            history_quantiles = np.asarray(predictions["history_quantiles"])
            zero_quantiles = np.asarray(predictions["zero_history_quantiles"])

        require(
            same_user_marker.shape == (1,),
            f"{family}: same_user_rows must be a scalar array",
        )
        same_user_rows = int(same_user_marker[0])
        require(
            0 < same_user_rows < len(row_index),
            f"{family}: invalid same_user_rows={same_user_rows}",
        )
        require(
            row_index.ndim == 1
            and int(row_index.min()) >= 0
            and int(row_index.max()) < total_rows,
            f"{family}: row indices out of bounds",
        )
        require(
            len(np.unique(row_index)) == len(row_index),
            f"{family}: duplicate row indices",
        )
        expected_shape = (len(row_index), 3, 7)
        require(
            history_quantiles.shape == expected_shape
            and zero_quantiles.shape == expected_shape,
            f"{family}: quantile shape mismatch",
        )
        require(
            np.isfinite(history_quantiles).all()
            and np.isfinite(zero_quantiles).all(),
            f"{family}: non-finite predictions",
        )
        crossing_failures = int(
            (np.diff(history_quantiles, axis=2) < -1e-6).sum()
            + (np.diff(zero_quantiles, axis=2) < -1e-6).sum()
        )
        require(crossing_failures == 0, f"{family}: quantile crossing")

        train_index = np.flatnonzero(
            (arrays["dataset"] == 0)
            & (arrays["unseen"] == PARTITION_TRAIN)
            & (arrays["sport"] != sport_code)
        )
        train_users = np.unique(arrays["users"][train_index])
        expected_same_user = np.flatnonzero(
            (arrays["dataset"] == 0)
            & (arrays["unseen"] == PARTITION_TRAIN)
            & (arrays["sport"] == sport_code)
            & (arrays["evaluation"] == 1)
            & np.isin(arrays["users"], train_users)
        )
        expected_joint = np.flatnonzero(
            (arrays["dataset"] == 0)
            & (arrays["unseen"] == PARTITION_TEST)
            & (arrays["sport"] == sport_code)
            & (arrays["evaluation"] == 1)
        )
        same_user_exact = np.array_equal(
            row_index[:same_user_rows], expected_same_user
        )
        joint_exact = np.array_equal(row_index[same_user_rows:], expected_joint)
        require(same_user_exact, f"{family}: same-user row mapping mismatch")
        require(joint_exact, f"{family}: joint-shift row mapping mismatch")

        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        for mode in ("history_informed", "zero_history"):
            require(mode in thresholds, f"{family}: missing {mode} thresholds")
            require(
                "0.9" in thresholds[mode]
                and len(thresholds[mode]["0.9"]) == 3,
                f"{family}: invalid 90% thresholds for {mode}",
            )

        regime_slices = {
            f"unseen_sport__{family}": slice(0, same_user_rows),
            f"joint_user_sport__{family}": slice(same_user_rows, len(row_index)),
        }
        mode_predictions = {
            "history_informed": history_quantiles,
            "zero_history": zero_quantiles,
        }
        family_user_counts: dict[str, int] = {}
        family_session_counts: dict[str, int] = {}

        for regime, selected_slice in regime_slices.items():
            selected_rows = row_index[selected_slice]
            selected_users = np.asarray(arrays["users"][selected_rows])
            selected_sessions = np.asarray(arrays["sessions"][selected_rows])
            family_user_counts[regime] = int(len(np.unique(selected_users)))
            family_session_counts[regime] = int(len(np.unique(selected_sessions)))

            for mode, all_predictions in mode_predictions.items():
                predictions = all_predictions[selected_slice]
                adjustments = np.asarray(
                    thresholds[mode]["0.9"], dtype=np.float64
                )
                for horizon_position, horizon in enumerate(HORIZONS):
                    target = np.asarray(
                        arrays["targets"][selected_rows, horizon_position]
                    )
                    adjustment = float(adjustments[horizon_position])
                    lower = np.clip(
                        predictions[
                            :, horizon_position, LOWER_QUANTILE_POSITION
                        ]
                        - adjustment,
                        30.0,
                        240.0,
                    )
                    upper = np.clip(
                        predictions[
                            :, horizon_position, UPPER_QUANTILE_POSITION
                        ]
                        + adjustment,
                        30.0,
                        240.0,
                    )
                    covered = (
                        (target >= lower) & (target <= upper)
                    ).astype(np.float32)
                    width = upper - lower
                    user_metrics = user_level_metrics(
                        covered,
                        width,
                        selected_users,
                        selected_sessions,
                    )
                    picp = float(user_metrics["covered"].mean())
                    mean_width = float(user_metrics["width"].mean())
                    coverage_error = abs(picp - NOMINAL_COVERAGE)
                    n_users = int(len(user_metrics))
                    n_sessions = int(len(np.unique(selected_sessions)))
                    n_origins = int(len(selected_rows))

                    reference_row = reference[
                        (reference["regime"] == regime)
                        & (reference["mode"] == mode)
                        & (reference["horizon_seconds"] == horizon)
                        & np.isclose(
                            reference["nominal_coverage"], NOMINAL_COVERAGE
                        )
                        & reference["calibrated"]
                    ]
                    require(
                        len(reference_row) == 1,
                        f"{family}/{regime}/{mode}/{horizon}: reference row count",
                    )
                    reference_record = reference_row.iloc[0]
                    recomputed = {
                        "picp": picp,
                        "absolute_coverage_error": coverage_error,
                        "mean_interval_width_bpm": mean_width,
                        "conformal_adjustment_bpm": adjustment,
                    }
                    for metric, value in recomputed.items():
                        delta = abs(float(reference_record[metric]) - value)
                        maximum_reference_delta[metric] = max(
                            maximum_reference_delta[metric], delta
                        )
                    require(
                        int(reference_record["users"]) == n_users
                        and int(reference_record["sessions"]) == n_sessions
                        and int(reference_record["origins"]) == n_origins,
                        f"{family}/{regime}/{mode}/{horizon}: support mismatch",
                    )

                    label = f"{family}|{regime}|{mode}|{horizon}"
                    bootstrap = paired_user_bootstrap(
                        user_metrics[["covered", "width"]].to_numpy(
                            dtype=np.float64
                        ),
                        args.bootstrap_replicates,
                        stable_seed(args.seed, label),
                    )
                    picp_ci_low, picp_ci_high = percentile_interval(
                        bootstrap[:, 0]
                    )
                    width_ci_low, width_ci_high = percentile_interval(
                        bootstrap[:, 1]
                    )
                    bootstrap_coverage_error = np.abs(
                        bootstrap[:, 0] - NOMINAL_COVERAGE
                    )
                    error_ci_low, error_ci_high = percentile_interval(
                        bootstrap_coverage_error
                    )

                    output_rows.append(
                        {
                            "analysis_version": ANALYSIS_VERSION,
                            "source_model_version": SOURCE_MODEL_VERSION,
                            "held_sport_code": sport_code,
                            "held_sport_family": family,
                            "regime": regime,
                            "mode": mode,
                            "horizon_seconds": horizon,
                            "nominal_coverage": NOMINAL_COVERAGE,
                            "calibrated": True,
                            "conformal_adjustment_bpm": adjustment,
                            "picp": picp,
                            "picp_ci_low": picp_ci_low,
                            "picp_ci_high": picp_ci_high,
                            "absolute_coverage_error": coverage_error,
                            "absolute_coverage_error_ci_low": error_ci_low,
                            "absolute_coverage_error_ci_high": error_ci_high,
                            "mean_interval_width_bpm": mean_width,
                            "mean_interval_width_ci_low": width_ci_low,
                            "mean_interval_width_ci_high": width_ci_high,
                            "users": n_users,
                            "sessions": n_sessions,
                            "origins": n_origins,
                            "support_caution_lt25_users": n_users < 25,
                            "bootstrap_replicates": args.bootstrap_replicates,
                            "bootstrap_unit": "user",
                            "aggregation": (
                                "origin-within-session, session-within-user, "
                                "equal-user mean"
                            ),
                        }
                    )

        family_audits[family] = {
            "held_sport_code": sport_code,
            "prediction_file": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "threshold_file": str(threshold_path),
            "threshold_sha256": sha256_file(threshold_path),
            "prediction_rows": int(len(row_index)),
            "same_user_rows_saved": same_user_rows,
            "same_user_rows_reconstructed": int(len(expected_same_user)),
            "joint_rows_saved": int(len(row_index) - same_user_rows),
            "joint_rows_reconstructed": int(len(expected_joint)),
            "same_user_mapping_exact": same_user_exact,
            "joint_mapping_exact": joint_exact,
            "row_indices_unique": True,
            "quantile_crossing_failures": crossing_failures,
            "users_by_regime": family_user_counts,
            "sessions_by_regime": family_session_counts,
            "output_rows": 12,
            "all_assertions_pass": True,
        }

    require(len(output_rows) == 60, f"expected 60 rows, got {len(output_rows)}")
    tolerance = 5e-6
    require(
        max(maximum_reference_delta.values()) <= tolerance,
        f"reference recomputation delta exceeded tolerance: {maximum_reference_delta}",
    )

    output_frame = pd.DataFrame(output_rows)
    metric_columns = [
        "picp",
        "picp_ci_low",
        "picp_ci_high",
        "absolute_coverage_error",
        "absolute_coverage_error_ci_low",
        "absolute_coverage_error_ci_high",
        "mean_interval_width_bpm",
        "mean_interval_width_ci_low",
        "mean_interval_width_ci_high",
    ]
    output_checks = {
        "duplicate_analysis_keys": int(
            output_frame.duplicated(
                ["held_sport_family", "regime", "mode", "horizon_seconds"]
            ).sum()
        ),
        "nonfinite_metric_values": int(
            (~np.isfinite(output_frame[metric_columns].to_numpy())).sum()
        ),
        "picp_range_failures": int(
            (~output_frame["picp"].between(0.0, 1.0)).sum()
        ),
        "nonpositive_width_failures": int(
            (output_frame["mean_interval_width_bpm"] <= 0.0).sum()
        ),
        "picp_point_outside_ci": int(
            (
                (output_frame["picp"] < output_frame["picp_ci_low"])
                | (output_frame["picp"] > output_frame["picp_ci_high"])
            ).sum()
        ),
        "width_point_outside_ci": int(
            (
                (
                    output_frame["mean_interval_width_bpm"]
                    < output_frame["mean_interval_width_ci_low"]
                )
                | (
                    output_frame["mean_interval_width_bpm"]
                    > output_frame["mean_interval_width_ci_high"]
                )
            ).sum()
        ),
    }
    output_diagnostics = {
        "coverage_error_point_outside_percentile_ci": int(
            (
                (
                    output_frame["absolute_coverage_error"]
                    < output_frame["absolute_coverage_error_ci_low"]
                )
                | (
                    output_frame["absolute_coverage_error"]
                    > output_frame["absolute_coverage_error_ci_high"]
                )
            ).sum()
        ),
        "coverage_error_ci_note": (
            "A percentile interval for the absolute transformed statistic need "
            "not contain the plug-in point estimate, especially in small, "
            "discrete user samples."
        ),
    }
    require(
        not any(output_checks.values()), f"output validation failed: {output_checks}"
    )

    atomic_csv(args.output_csv, output_rows)
    payload: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "source_model_version": SOURCE_MODEL_VERSION,
        "intended_use": (
            "calibrated 90% sport-shift and joint user-sport uncertainty "
            "with user-bootstrap confidence intervals; no retraining"
        ),
        "array_rows": total_rows,
        "array_row_counts": row_counts,
        "families": family_audits,
        "joint_user_sport_feasible": True,
        "joint_user_sport_feasibility_evidence": (
            "For every family, the NPZ suffix after same_user_rows exactly "
            "matched the independently reconstructed dataset=Endomondo, "
            "unseen_user_partition=test, held-sport, evaluation-origin rows."
        ),
        "metrics": [
            "calibrated_90_picp",
            "absolute_coverage_error",
            "mean_interval_width_bpm",
        ],
        "aggregation": (
            "origins averaged within session, sessions averaged within user, "
            "users equally weighted"
        ),
        "bootstrap": {
            "unit": "user",
            "replicates": args.bootstrap_replicates,
            "base_seed": args.seed,
            "interval": "two-sided 95% percentile",
            "paired_metrics": True,
        },
        "reference_csv": str(args.reference_csv),
        "maximum_absolute_reference_recomputation_delta": maximum_reference_delta,
        "reference_tolerance": tolerance,
        "output_csv": str(args.output_csv),
        "output_csv_sha256": sha256_file(args.output_csv),
        "output_rows": len(output_rows),
        "output_checks": output_checks,
        "output_diagnostics": output_diagnostics,
        "limitations": [
            (
                "Bootstrap intervals quantify between-user sampling variation "
                "conditional on each already-fitted v0.12.0 checkpoint; they do "
                "not include model-training or seed variability."
            ),
            (
                "The conformal thresholds are the existing v0.12.0 thresholds, "
                "which were calibrated over correlated origins; this analysis "
                "does not retrofit user-clustered conformal calibration."
            ),
            (
                "Joint intersections with fewer than 25 users remain cautionary "
                "even when a bootstrap interval is reported."
            ),
            (
                "The analysis evaluates empirical average coverage and width; it "
                "does not establish conditional or per-user coverage guarantees."
            ),
        ],
        "all_assertions_pass": True,
    }
    atomic_json(args.audit_json, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit existing sport-shift predictions and bootstrap calibrated "
            "90% interval metrics at the user level without retraining."
        )
    )
    parser.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("outputs/predictions"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/models/sport_shift_v0_12_0"),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("outputs/results/sport_shift_interval_v0_12_0.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/results/sport_shift_uncertainty_bootstrap_v0_17_0.csv"
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/audit/sport_shift_uncertainty_bootstrap_v0_17_0.json"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("at least 1,000 bootstrap replicates are required")
    payload = analyze(args)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
