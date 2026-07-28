from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ANALYSIS_VERSION = "0.18.0"
NOMINAL_COVERAGE = 0.90
HORIZONS = (60, 180, 300)
QUANTILES = np.asarray(
    [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float32
)
INTERVALS = {
    0.50: (2, 4),
    0.80: (1, 5),
    0.90: (0, 6),
}
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def bootstrap_user_means(
    user_values: np.ndarray,
    replicates: int,
    seed: int,
    batch_size: int = 2_000,
) -> np.ndarray:
    if user_values.ndim != 2 or len(user_values) < 2:
        raise ValueError("expected at least two users and a two-dimensional matrix")
    generator = np.random.default_rng(seed)
    result = np.empty(
        (replicates, user_values.shape[1]), dtype=np.float64
    )
    for start in range(0, replicates, batch_size):
        end = min(replicates, start + batch_size)
        index = generator.integers(
            0, len(user_values), size=(end - start, len(user_values))
        )
        result[start:end] = user_values[index].mean(axis=1)
    return result


def hierarchical_user_metrics(
    covered: np.ndarray,
    width: np.ndarray,
    wis: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "covered": covered,
            "width": width,
            "wis": wis,
            "user": users,
            "session": sessions,
        }
    )
    session_level = frame.groupby(["user", "session"], sort=False)[
        ["covered", "width", "wis"]
    ].mean()
    return session_level.groupby(level="user", sort=False).mean()


def calibrated_bounds(
    prediction: np.ndarray,
    thresholds: dict[str, list[float]],
    horizon_position: int,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    result: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for coverage, (lower_position, upper_position) in INTERVALS.items():
        adjustment = float(thresholds[str(coverage)][horizon_position])
        result[coverage] = (
            np.clip(
                prediction[:, lower_position] - adjustment, 30.0, 240.0
            ),
            np.clip(
                prediction[:, upper_position] + adjustment, 30.0, 240.0
            ),
        )
    return result


def weighted_interval_score(
    median: np.ndarray,
    target: np.ndarray,
    bounds: dict[float, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    total = 0.5 * np.abs(target - median)
    for coverage, (lower, upper) in bounds.items():
        alpha = 1.0 - coverage
        interval_score = (
            upper
            - lower
            + (2.0 / alpha) * (lower - target) * (target < lower)
            + (2.0 / alpha) * (target - upper) * (target > upper)
        )
        total = total + (alpha / 2.0) * interval_score
    return total / (len(bounds) + 0.5)


def per_user_spearman(
    width: np.ndarray,
    absolute_error: np.ndarray,
    users: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    user_ids: list[int] = []
    correlations: list[float] = []
    for user in np.unique(users):
        selected = users == user
        if int(selected.sum()) < 3:
            continue
        if (
            np.ptp(width[selected]) == 0
            or np.ptp(absolute_error[selected]) == 0
        ):
            continue
        correlation = spearmanr(
            width[selected], absolute_error[selected]
        ).statistic
        if np.isfinite(correlation):
            user_ids.append(int(user))
            correlations.append(float(correlation))
    return np.asarray(user_ids, dtype=np.int64), np.asarray(
        correlations, dtype=np.float64
    )


def load_thresholds(path: Path, mode: str) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    threshold_root = payload.get("thresholds", payload)
    require(mode in threshold_root, f"{path}: missing threshold mode {mode}")
    thresholds = threshold_root[mode]
    for coverage in ("0.5", "0.8", "0.9"):
        require(
            coverage in thresholds and len(thresholds[coverage]) == 3,
            f"{path}: invalid {coverage} thresholds for {mode}",
        )
    return thresholds


def reference_row(
    frame: pd.DataFrame,
    regime: str,
    mode: str,
    horizon: int,
    interval: bool,
) -> pd.Series:
    selected = frame[
        (frame["regime"] == regime)
        & (frame["mode"] == mode)
        & (frame["horizon_seconds"] == horizon)
        & frame["calibrated"]
    ]
    if interval:
        selected = selected[
            np.isclose(selected["nominal_coverage"], NOMINAL_COVERAGE)
        ]
    require(
        len(selected) == 1,
        f"reference row count for {regime}/{mode}/{horizon}: {len(selected)}",
    )
    return selected.iloc[0]


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
        "external": np.load(
            args.array_dir / "primary_external_partition.npy", mmap_mode="r"
        ),
        "temporal": np.load(
            args.array_dir / "temporal_partition_strict.npy", mmap_mode="r"
        ),
        "users": np.load(args.array_dir / "user_index.npy", mmap_mode="r"),
        "sessions": np.load(args.array_dir / "session_index.npy", mmap_mode="r"),
    }
    row_counts = {name: int(len(value)) for name, value in arrays.items()}
    require(len(set(row_counts.values())) == 1, f"array mismatch: {row_counts}")
    total_rows = next(iter(row_counts.values()))

    temporal_prediction_path = (
        args.prediction_dir / "temporal_uncertainty_v0_13_0.npz"
    )
    user_prediction_path = (
        args.prediction_dir / "uncertainty_user_generalization_v0_11_0.npz"
    )
    temporal_threshold_path = (
        args.temporal_model_dir / "conformal_thresholds.json"
    )
    user_threshold_path = (
        args.user_model_dir / "conformal_thresholds_v0_11_0.json"
    )
    for path in (
        temporal_prediction_path,
        user_prediction_path,
        temporal_threshold_path,
        user_threshold_path,
    ):
        require(path.exists(), f"missing source file {path}")

    with np.load(temporal_prediction_path) as source:
        require(
            {"row_index", "history_quantiles"}.issubset(source.files),
            "temporal NPZ missing required fields",
        )
        temporal_rows = np.asarray(source["row_index"], dtype=np.int64)
        temporal_predictions = np.asarray(source["history_quantiles"])
    expected_temporal = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["temporal"] == PARTITION_TEST)
        & (arrays["evaluation"] == 1)
    )
    temporal_exact = np.array_equal(temporal_rows, expected_temporal)
    require(temporal_exact, "temporal NPZ row mapping mismatch")

    with np.load(user_prediction_path) as source:
        require(
            {
                "row_index",
                "history_quantiles",
                "zero_history_quantiles",
            }.issubset(source.files),
            "user/external NPZ missing required fields",
        )
        user_all_rows = np.asarray(source["row_index"], dtype=np.int64)
        unseen_mask = (
            (arrays["dataset"][user_all_rows] == 0)
            & (arrays["unseen"][user_all_rows] == PARTITION_TEST)
        )
        external_mask = (
            (arrays["dataset"][user_all_rows] == 1)
            & (arrays["external"][user_all_rows] == EXTERNAL_FROZEN)
        )
        unseen_predictions = np.asarray(source["history_quantiles"])[
            unseen_mask
        ]
        external_predictions = np.asarray(source["zero_history_quantiles"])[
            external_mask
        ]
    expected_user_all = np.flatnonzero(
        ((arrays["dataset"] == 0) & (arrays["evaluation"] == 1))
        | (
            (arrays["dataset"] == 1)
            & (arrays["external"] == EXTERNAL_FROZEN)
        )
    )
    user_all_exact = np.array_equal(user_all_rows, expected_user_all)
    require(user_all_exact, "user/external NPZ full row mapping mismatch")
    unseen_rows = user_all_rows[unseen_mask]
    external_rows = user_all_rows[external_mask]
    expected_unseen = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["unseen"] == PARTITION_TEST)
        & (arrays["evaluation"] == 1)
    )
    expected_external = np.flatnonzero(
        (arrays["dataset"] == 1)
        & (arrays["external"] == EXTERNAL_FROZEN)
    )
    unseen_exact = np.array_equal(unseen_rows, expected_unseen)
    external_exact = np.array_equal(external_rows, expected_external)
    require(unseen_exact, "unseen-user row mapping mismatch")
    require(external_exact, "external row mapping mismatch")

    for name, rows, predictions in (
        ("temporal", temporal_rows, temporal_predictions),
        ("unseen_user", unseen_rows, unseen_predictions),
        ("external", external_rows, external_predictions),
    ):
        require(
            rows.ndim == 1
            and len(np.unique(rows)) == len(rows)
            and int(rows.min()) >= 0
            and int(rows.max()) < total_rows,
            f"{name}: invalid row indices",
        )
        require(
            predictions.shape == (len(rows), 3, 7),
            f"{name}: prediction shape {predictions.shape}",
        )
        require(np.isfinite(predictions).all(), f"{name}: non-finite predictions")
        require(
            int((np.diff(predictions, axis=2) < -1e-6).sum()) == 0,
            f"{name}: quantile crossings",
        )

    temporal_intervals = pd.read_csv(args.temporal_interval_reference)
    temporal_probabilistic = pd.read_csv(
        args.temporal_probabilistic_reference
    )
    user_intervals = pd.read_csv(args.user_interval_reference)
    user_probabilistic = pd.read_csv(args.user_probabilistic_reference)
    for frame in (
        temporal_intervals,
        temporal_probabilistic,
        user_intervals,
        user_probabilistic,
    ):
        frame["calibrated"] = (
            frame["calibrated"].astype(str).str.lower() == "true"
        )

    regimes = [
        {
            "regime": "within_user_temporal_test",
            "mode": "history_informed",
            "source_model_version": "0.13.0",
            "rows": temporal_rows,
            "predictions": temporal_predictions,
            "thresholds": load_thresholds(
                temporal_threshold_path, "history_informed"
            ),
            "interval_reference": temporal_intervals,
            "probabilistic_reference": temporal_probabilistic,
        },
        {
            "regime": "unseen_user_test",
            "mode": "history_informed",
            "source_model_version": "0.11.0",
            "rows": unseen_rows,
            "predictions": unseen_predictions,
            "thresholds": load_thresholds(
                user_threshold_path, "history_informed"
            ),
            "interval_reference": user_intervals,
            "probabilistic_reference": user_probabilistic,
        },
        {
            "regime": "goldencheetah_frozen_external",
            "mode": "zero_history",
            "source_model_version": "0.11.0",
            "rows": external_rows,
            "predictions": external_predictions,
            "thresholds": load_thresholds(
                user_threshold_path, "zero_history"
            ),
            "interval_reference": user_intervals,
            "probabilistic_reference": user_probabilistic,
        },
    ]

    output_rows: list[dict[str, object]] = []
    maximum_reference_delta = {
        "picp": 0.0,
        "absolute_coverage_error": 0.0,
        "mean_interval_width_bpm": 0.0,
        "weighted_interval_score": 0.0,
        "mean_user_spearman_width_absolute_error": 0.0,
    }
    regime_audit: dict[str, object] = {}

    for regime_spec in regimes:
        regime = str(regime_spec["regime"])
        mode = str(regime_spec["mode"])
        rows = np.asarray(regime_spec["rows"], dtype=np.int64)
        predictions = np.asarray(regime_spec["predictions"])
        thresholds = regime_spec["thresholds"]
        interval_reference = regime_spec["interval_reference"]
        probabilistic_reference = regime_spec["probabilistic_reference"]
        selected_users = np.asarray(arrays["users"][rows])
        selected_sessions = np.asarray(arrays["sessions"][rows])
        regime_spearman_users: list[int] = []

        for horizon_position, horizon in enumerate(HORIZONS):
            target = np.asarray(arrays["targets"][rows, horizon_position])
            prediction = predictions[:, horizon_position]
            median = prediction[:, 3]
            bounds = calibrated_bounds(
                prediction, thresholds, horizon_position
            )
            lower_90, upper_90 = bounds[NOMINAL_COVERAGE]
            covered = ((target >= lower_90) & (target <= upper_90)).astype(
                np.float32
            )
            width = upper_90 - lower_90
            wis = weighted_interval_score(median, target, bounds)
            absolute_error = np.abs(median - target)

            user_metrics = hierarchical_user_metrics(
                covered,
                width,
                wis,
                selected_users,
                selected_sessions,
            )
            picp = float(user_metrics["covered"].mean())
            mean_width = float(user_metrics["width"].mean())
            mean_wis = float(user_metrics["wis"].mean())
            coverage_error = abs(picp - NOMINAL_COVERAGE)
            n_users = int(len(user_metrics))
            n_sessions = int(len(np.unique(selected_sessions)))
            n_origins = int(len(rows))

            spearman_user_ids, spearman_values = per_user_spearman(
                width, absolute_error, selected_users
            )
            mean_spearman = float(spearman_values.mean())
            regime_spearman_users.append(int(len(spearman_values)))

            interval_row = reference_row(
                interval_reference, regime, mode, horizon, interval=True
            )
            probabilistic_row = reference_row(
                probabilistic_reference,
                regime,
                mode,
                horizon,
                interval=False,
            )
            recomputed = {
                "picp": (picp, float(interval_row["picp"])),
                "absolute_coverage_error": (
                    coverage_error,
                    float(interval_row["absolute_coverage_error"]),
                ),
                "mean_interval_width_bpm": (
                    mean_width,
                    float(interval_row["mean_interval_width_bpm"]),
                ),
                "weighted_interval_score": (
                    mean_wis,
                    float(probabilistic_row["weighted_interval_score"]),
                ),
                "mean_user_spearman_width_absolute_error": (
                    mean_spearman,
                    float(
                        probabilistic_row[
                            "mean_user_spearman_width_absolute_error"
                        ]
                    ),
                ),
            }
            for metric, (value, reference_value) in recomputed.items():
                maximum_reference_delta[metric] = max(
                    maximum_reference_delta[metric],
                    abs(value - reference_value),
                )
            require(
                int(interval_row["users"]) == n_users
                and int(interval_row["sessions"]) == n_sessions
                and int(interval_row["origins"]) == n_origins,
                f"{regime}/{horizon}: interval support mismatch",
            )
            require(
                int(probabilistic_row["users"]) == n_users
                and int(probabilistic_row["sessions"]) == n_sessions
                and int(probabilistic_row["origins"]) == n_origins
                and int(probabilistic_row["users_with_defined_spearman"])
                == len(spearman_values),
                f"{regime}/{horizon}: probabilistic support mismatch",
            )

            bootstrap = bootstrap_user_means(
                user_metrics[["covered", "width", "wis"]].to_numpy(
                    dtype=np.float64
                ),
                args.bootstrap_replicates,
                stable_seed(args.seed, f"{regime}|{mode}|{horizon}|metrics"),
            )
            picp_ci_low, picp_ci_high = percentile_interval(bootstrap[:, 0])
            width_ci_low, width_ci_high = percentile_interval(bootstrap[:, 1])
            wis_ci_low, wis_ci_high = percentile_interval(bootstrap[:, 2])
            bootstrap_coverage_error = np.abs(
                bootstrap[:, 0] - NOMINAL_COVERAGE
            )
            error_ci_low, error_ci_high = percentile_interval(
                bootstrap_coverage_error
            )

            spearman_bootstrap = bootstrap_user_means(
                spearman_values[:, None],
                args.bootstrap_replicates,
                stable_seed(
                    args.seed, f"{regime}|{mode}|{horizon}|spearman"
                ),
            )[:, 0]
            spearman_ci_low, spearman_ci_high = percentile_interval(
                spearman_bootstrap
            )

            output_rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "source_model_version": regime_spec[
                        "source_model_version"
                    ],
                    "regime": regime,
                    "mode": mode,
                    "horizon_seconds": horizon,
                    "nominal_coverage": NOMINAL_COVERAGE,
                    "calibrated": True,
                    "picp": picp,
                    "picp_ci_low": picp_ci_low,
                    "picp_ci_high": picp_ci_high,
                    "absolute_coverage_error": coverage_error,
                    "absolute_coverage_error_ci_low": error_ci_low,
                    "absolute_coverage_error_ci_high": error_ci_high,
                    "mean_90_interval_width_bpm": mean_width,
                    "mean_90_interval_width_ci_low": width_ci_low,
                    "mean_90_interval_width_ci_high": width_ci_high,
                    "weighted_interval_score": mean_wis,
                    "weighted_interval_score_ci_low": wis_ci_low,
                    "weighted_interval_score_ci_high": wis_ci_high,
                    "mean_user_spearman_width_absolute_error": mean_spearman,
                    "mean_user_spearman_ci_low": spearman_ci_low,
                    "mean_user_spearman_ci_high": spearman_ci_high,
                    "users_with_defined_spearman": int(
                        len(spearman_user_ids)
                    ),
                    "users": n_users,
                    "sessions": n_sessions,
                    "origins": n_origins,
                    "bootstrap_replicates": args.bootstrap_replicates,
                    "bootstrap_unit": "user",
                    "aggregation": (
                        "origin-within-session, session-within-user, "
                        "equal-user mean; Spearman computed within user"
                    ),
                }
            )

        regime_audit[regime] = {
            "mode": mode,
            "source_model_version": regime_spec["source_model_version"],
            "rows": int(len(rows)),
            "users": int(len(np.unique(selected_users))),
            "sessions": int(len(np.unique(selected_sessions))),
            "users_with_defined_spearman_by_horizon": regime_spearman_users,
            "quantile_crossing_failures": int(
                (np.diff(predictions, axis=2) < -1e-6).sum()
            ),
            "all_assertions_pass": True,
        }

    require(len(output_rows) == 9, f"expected 9 rows, got {len(output_rows)}")
    tolerance = 5e-6
    require(
        max(maximum_reference_delta.values()) <= tolerance,
        f"reference mismatch: {maximum_reference_delta}",
    )

    output_frame = pd.DataFrame(output_rows)
    key_columns = ["regime", "mode", "horizon_seconds"]
    numeric_columns = [
        column
        for column in output_frame.columns
        if any(
            token in column
            for token in (
                "picp",
                "coverage_error",
                "interval_width",
                "weighted_interval_score",
                "spearman",
            )
        )
        and column != "users_with_defined_spearman"
    ]
    output_checks = {
        "duplicate_analysis_keys": int(
            output_frame.duplicated(key_columns).sum()
        ),
        "nonfinite_metric_values": int(
            (~np.isfinite(output_frame[numeric_columns].to_numpy())).sum()
        ),
        "picp_range_failures": int(
            (~output_frame["picp"].between(0.0, 1.0)).sum()
        ),
        "nonpositive_width_failures": int(
            (output_frame["mean_90_interval_width_bpm"] <= 0.0).sum()
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
                    output_frame["mean_90_interval_width_bpm"]
                    < output_frame["mean_90_interval_width_ci_low"]
                )
                | (
                    output_frame["mean_90_interval_width_bpm"]
                    > output_frame["mean_90_interval_width_ci_high"]
                )
            ).sum()
        ),
        "wis_point_outside_ci": int(
            (
                (
                    output_frame["weighted_interval_score"]
                    < output_frame["weighted_interval_score_ci_low"]
                )
                | (
                    output_frame["weighted_interval_score"]
                    > output_frame["weighted_interval_score_ci_high"]
                )
            ).sum()
        ),
        "spearman_point_outside_ci": int(
            (
                (
                    output_frame[
                        "mean_user_spearman_width_absolute_error"
                    ]
                    < output_frame["mean_user_spearman_ci_low"]
                )
                | (
                    output_frame[
                        "mean_user_spearman_width_absolute_error"
                    ]
                    > output_frame["mean_user_spearman_ci_high"]
                )
            ).sum()
        ),
    }
    require(
        not any(output_checks.values()), f"output validation failed: {output_checks}"
    )
    coverage_error_outside = int(
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
    )

    atomic_csv(args.output_csv, output_rows)
    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "intended_use": (
            "Figure 3 post-CQR 90% PICP, width, WIS, and within-user "
            "width-error Spearman confidence intervals without retraining"
        ),
        "array_rows": total_rows,
        "array_row_counts": row_counts,
        "source_files": {
            "temporal_predictions": {
                "path": str(temporal_prediction_path),
                "sha256": sha256_file(temporal_prediction_path),
            },
            "user_external_predictions": {
                "path": str(user_prediction_path),
                "sha256": sha256_file(user_prediction_path),
            },
            "temporal_thresholds": {
                "path": str(temporal_threshold_path),
                "sha256": sha256_file(temporal_threshold_path),
            },
            "user_external_thresholds": {
                "path": str(user_threshold_path),
                "sha256": sha256_file(user_threshold_path),
            },
        },
        "alignment": {
            "temporal_npz_rows": int(len(temporal_rows)),
            "temporal_reconstructed_rows": int(len(expected_temporal)),
            "temporal_mapping_exact": temporal_exact,
            "user_external_npz_rows": int(len(user_all_rows)),
            "user_external_reconstructed_rows": int(len(expected_user_all)),
            "user_external_full_mapping_exact": user_all_exact,
            "unseen_user_rows": int(len(unseen_rows)),
            "unseen_user_reconstructed_rows": int(len(expected_unseen)),
            "unseen_user_mapping_exact": unseen_exact,
            "external_rows": int(len(external_rows)),
            "external_reconstructed_rows": int(len(expected_external)),
            "external_mapping_exact": external_exact,
            "unused_user_external_npz_rows": int(
                len(user_all_rows) - len(unseen_rows) - len(external_rows)
            ),
        },
        "regimes": regime_audit,
        "metrics": {
            "picp_and_width": "post-CQR central 90% interval",
            "wis": (
                "post-CQR central 50%, 80%, and 90% intervals plus median; "
                "same formula as evaluate_probabilistic_metrics.py"
            ),
            "spearman": (
                "90% interval width versus absolute median error within each "
                "eligible user, then equal-user mean"
            ),
        },
        "bootstrap": {
            "unit": "user",
            "replicates": args.bootstrap_replicates,
            "base_seed": args.seed,
            "interval": "two-sided 95% percentile",
            "picp_width_wis_resampled_together": True,
            "spearman_resampled_over_users_with_defined_correlations": True,
        },
        "maximum_absolute_reference_recomputation_delta": (
            maximum_reference_delta
        ),
        "reference_tolerance": tolerance,
        "output_csv": str(args.output_csv),
        "output_csv_sha256": sha256_file(args.output_csv),
        "output_rows": len(output_rows),
        "output_checks": output_checks,
        "output_diagnostics": {
            "coverage_error_point_outside_percentile_ci": (
                coverage_error_outside
            ),
            "coverage_error_ci_note": (
                "A percentile interval for the absolute transformed statistic "
                "need not contain the plug-in estimate."
            ),
        },
        "limitations": [
            (
                "Bootstrap intervals quantify between-user sampling variation "
                "conditional on the existing v0.11.0 and v0.13.0 checkpoints; "
                "they do not include model-training or seed variability."
            ),
            (
                "Existing CQR thresholds were calibrated over correlated "
                "origins. This analysis does not retrofit user-clustered "
                "conformal calibration or establish a finite-sample user-level "
                "coverage guarantee."
            ),
            (
                "PICP, width, and WIS use session-then-user aggregation. The "
                "Spearman statistic is computed directly across origins within "
                "each eligible user to reproduce the existing metric exactly."
            ),
            (
                "The external primary subset contains 531,725 evaluated origins "
                "from 31,851 sessions and 144 users; it is not the larger "
                "537,672-origin pre-restriction origin set."
            ),
            (
                "The reported intervals describe empirical average performance, "
                "not conditional coverage for each user, sport, or forecast."
            ),
        ],
        "all_assertions_pass": True,
    }
    atomic_json(args.audit_json, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Figure 3 prediction alignment and bootstrap post-CQR "
            "probabilistic metrics at the user level without retraining."
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
        "--temporal-model-dir",
        type=Path,
        default=Path("outputs/models/temporal_uncertainty_v0_13_0"),
    )
    parser.add_argument(
        "--user-model-dir",
        type=Path,
        default=Path(
            "outputs/models/uncertainty_user_generalization_v0_11_0"
        ),
    )
    parser.add_argument(
        "--temporal-interval-reference",
        type=Path,
        default=Path(
            "outputs/results/temporal_uncertainty_interval_v0_13_0.csv"
        ),
    )
    parser.add_argument(
        "--temporal-probabilistic-reference",
        type=Path,
        default=Path(
            "outputs/results/temporal_probabilistic_metrics_v0_13_0.csv"
        ),
    )
    parser.add_argument(
        "--user-interval-reference",
        type=Path,
        default=Path("outputs/results/uncertainty_interval_metrics_v0_11_0.csv"),
    )
    parser.add_argument(
        "--user-probabilistic-reference",
        type=Path,
        default=Path("outputs/results/probabilistic_metrics_v0_11_0.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/results/figure3_uncertainty_bootstrap_v0_18_0.csv"
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/audit/figure3_uncertainty_bootstrap_v0_18_0.json"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("at least 1,000 bootstrap replicates are required")
    audit = analyze(args)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
