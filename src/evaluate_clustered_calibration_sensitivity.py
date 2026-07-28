from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.20.0"
SOURCE_MODEL_VERSION = "0.11.0"
HORIZONS = (60, 180, 300)
INTERVALS = {
    0.50: (2, 4),
    0.80: (1, 5),
    0.90: (0, 6),
}
PARTITION_CALIBRATION = 3
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1
MODES = {
    "history_informed": "history_quantiles",
    "zero_history": "zero_history_quantiles",
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def hierarchical_origin_weights(
    users: np.ndarray, sessions: np.ndarray
) -> np.ndarray:
    """Give users equal weight, sessions equal weight within user, and origins
    equal weight within session.
    """

    users = np.asarray(users)
    sessions = np.asarray(sessions)
    if users.ndim != 1 or sessions.ndim != 1 or len(users) != len(sessions):
        raise ValueError("users and sessions must be aligned one-dimensional arrays")
    if len(users) == 0:
        raise ValueError("empty hierarchy")

    unique_sessions, first, session_inverse, origins_per_session = np.unique(
        sessions, return_index=True, return_inverse=True, return_counts=True
    )
    session_users = users[first]
    if not np.array_equal(users, session_users[session_inverse]):
        raise ValueError("a session maps to more than one user")
    unique_users, session_user_inverse = np.unique(
        session_users, return_inverse=True
    )
    sessions_per_user = np.bincount(session_user_inverse)
    weights = 1.0 / (
        len(unique_users)
        * sessions_per_user[session_user_inverse[session_inverse]]
        * origins_per_session[session_inverse]
    )
    if len(unique_sessions) != len(origins_per_session):
        raise AssertionError("session indexing failure")
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"hierarchical weights sum to {weights.sum()}")
    return weights.astype(np.float64)


def weighted_quantile_higher(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or len(values) == 0:
        raise ValueError("values and weights must be aligned non-empty vectors")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("non-finite quantile input")
    if (weights <= 0).any():
        raise ValueError("weights must be strictly positive")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    target = probability * cumulative[-1]
    position = min(
        int(np.searchsorted(cumulative, target, side="left")),
        len(sorted_values) - 1,
    )
    return float(sorted_values[position])


def nonconformity_score(
    prediction: np.ndarray,
    target: np.ndarray,
    lower_position: int,
    upper_position: int,
) -> np.ndarray:
    return np.maximum(
        prediction[:, lower_position] - target,
        target - prediction[:, upper_position],
    ).astype(np.float64)


def hierarchical_metric(
    values: np.ndarray, users: np.ndarray, sessions: np.ndarray
) -> float:
    weights = hierarchical_origin_weights(users, sessions)
    return float(np.dot(np.asarray(values, dtype=np.float64), weights))


def cluster_subsample_positions(
    users: np.ndarray,
    sessions: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap calibration users, then draw one session and one origin per
    sampled user. This is a sensitivity sampler, not a conformal guarantee.
    """

    users = np.asarray(users)
    sessions = np.asarray(sessions)
    unique_users = np.unique(users)
    rows_by_user: list[list[np.ndarray]] = []
    for user in unique_users:
        user_positions = np.flatnonzero(users == user)
        user_sessions = np.unique(sessions[user_positions])
        rows_by_user.append(
            [user_positions[sessions[user_positions] == session] for session in user_sessions]
        )
    generator = np.random.default_rng(seed)
    result = np.empty((replicates, len(unique_users)), dtype=np.int32)
    for replicate in range(replicates):
        sampled_users = generator.integers(
            0, len(unique_users), size=len(unique_users)
        )
        for column, user_position in enumerate(sampled_users):
            session_groups = rows_by_user[int(user_position)]
            session_group = session_groups[
                int(generator.integers(0, len(session_groups)))
            ]
            result[replicate, column] = int(
                session_group[int(generator.integers(0, len(session_group)))]
            )
    return result


def sampled_thresholds(
    scores: np.ndarray,
    sampled_positions: np.ndarray,
    coverage: float,
) -> np.ndarray:
    n_clusters = sampled_positions.shape[1]
    rank = min(n_clusters, math.ceil((n_clusters + 1) * coverage))
    sampled_scores = np.asarray(scores)[sampled_positions]
    thresholds = np.partition(sampled_scores, rank - 1, axis=1)[:, rank - 1]
    return np.maximum(thresholds, 0.0).astype(np.float64)


def per_user_weighted_cdf(
    scores: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a replicate-by-user matrix of session-balanced coverage."""

    user_ids = np.unique(users)
    output = np.empty((len(thresholds), len(user_ids)), dtype=np.float32)
    for position, user in enumerate(user_ids):
        selected = users == user
        user_scores = np.asarray(scores[selected], dtype=np.float64)
        user_weights = hierarchical_origin_weights(
            np.zeros(int(selected.sum()), dtype=np.int8), sessions[selected]
        )
        order = np.argsort(user_scores, kind="mergesort")
        sorted_scores = user_scores[order]
        cumulative = np.cumsum(user_weights[order])
        indices = np.searchsorted(sorted_scores, thresholds, side="right")
        values = np.zeros(len(thresholds), dtype=np.float64)
        positive = indices > 0
        values[positive] = cumulative[indices[positive] - 1]
        output[:, position] = values
    return user_ids, output


def paired_replicate_user_bootstrap(
    replicate_user_values: np.ndarray, seed: int, batch_size: int = 1_000
) -> np.ndarray:
    if replicate_user_values.ndim != 2 or replicate_user_values.shape[1] < 2:
        raise ValueError("expected a replicate-by-user matrix")
    generator = np.random.default_rng(seed)
    replicates, n_users = replicate_user_values.shape
    result = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        end = min(replicates, start + batch_size)
        indices = generator.integers(
            0, n_users, size=(end - start, n_users)
        )
        rows = np.arange(start, end)[:, None]
        result[start:end] = replicate_user_values[rows, indices].mean(axis=1)
    return result


def load_thresholds(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = payload.get("thresholds", payload)
    for mode in MODES:
        require(mode in root, f"missing threshold mode {mode}")
        for coverage in ("0.5", "0.8", "0.9"):
            require(
                coverage in root[mode] and len(root[mode][coverage]) == 3,
                f"invalid {mode}/{coverage} thresholds",
            )
    return root


def reference_interval_row(
    frame: pd.DataFrame,
    regime: str,
    mode: str,
    horizon: int,
    coverage: float,
) -> pd.Series:
    selected = frame[
        (frame["regime"] == regime)
        & (frame["mode"] == mode)
        & (frame["horizon_seconds"] == horizon)
        & np.isclose(frame["nominal_coverage"], coverage)
        & frame["calibrated"]
    ]
    require(
        len(selected) == 1,
        f"reference row count {regime}/{mode}/{horizon}/{coverage}: {len(selected)}",
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
        "users": np.load(args.array_dir / "user_index.npy", mmap_mode="r"),
        "sessions": np.load(args.array_dir / "session_index.npy", mmap_mode="r"),
    }
    row_counts = {name: int(len(value)) for name, value in arrays.items()}
    require(len(set(row_counts.values())) == 1, f"array mismatch: {row_counts}")

    with np.load(args.predictions) as source:
        require(
            {"row_index", *MODES.values()}.issubset(source.files),
            "prediction archive missing required fields",
        )
        row_index = np.asarray(source["row_index"], dtype=np.int64)
        predictions = {
            mode: np.asarray(source[key]) for mode, key in MODES.items()
        }
    expected_rows = np.flatnonzero(
        ((arrays["dataset"] == 0) & (arrays["evaluation"] == 1))
        | (
            (arrays["dataset"] == 1)
            & (arrays["external"] == EXTERNAL_FROZEN)
        )
    )
    require(np.array_equal(row_index, expected_rows), "prediction row mapping mismatch")
    for mode, prediction in predictions.items():
        require(
            prediction.shape == (len(row_index), 3, 7),
            f"{mode}: prediction shape mismatch",
        )
        require(np.isfinite(prediction).all(), f"{mode}: non-finite prediction")
        require(
            int((np.diff(prediction, axis=2) < -1e-6).sum()) == 0,
            f"{mode}: quantile crossings",
        )

    local = {
        name: np.asarray(values[row_index])
        for name, values in arrays.items()
        if name != "evaluation"
    }
    masks = {
        "calibration": (local["dataset"] == 0)
        & (local["unseen"] == PARTITION_CALIBRATION),
        "unseen_user_test": (local["dataset"] == 0)
        & (local["unseen"] == PARTITION_TEST),
        "goldencheetah_frozen_external": (local["dataset"] == 1)
        & (local["external"] == EXTERNAL_FROZEN),
    }
    expected_support = {
        "calibration": (85_247, 97),
        "unseen_user_test": (101_184, 105),
        "goldencheetah_frozen_external": (531_725, 144),
    }
    for name, (expected_origins, expected_users) in expected_support.items():
        require(int(masks[name].sum()) == expected_origins, f"{name}: origin count")
        require(
            len(np.unique(local["users"][masks[name]])) == expected_users,
            f"{name}: user count",
        )

    origin_thresholds = load_thresholds(args.thresholds)
    reference = pd.read_csv(args.interval_reference)
    reference["calibrated"] = (
        reference["calibrated"].astype(str).str.lower() == "true"
    )
    calibration_weights = hierarchical_origin_weights(
        local["users"][masks["calibration"]],
        local["sessions"][masks["calibration"]],
    )
    sampled_positions = cluster_subsample_positions(
        local["users"][masks["calibration"]],
        local["sessions"][masks["calibration"]],
        args.bootstrap_replicates,
        stable_seed(args.seed, "calibration-cluster-subsample"),
    )

    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    maximum_reference_delta = {"picp": 0.0, "width": 0.0}
    balanced_thresholds: dict[str, dict[str, list[float]]] = {
        mode: {str(level): [] for level in INTERVALS} for mode in MODES
    }

    for mode, prediction in predictions.items():
        calibration_prediction = prediction[masks["calibration"]]
        calibration_targets = local["targets"][masks["calibration"]]
        for coverage, (lower_position, upper_position) in INTERVALS.items():
            coverage_key = str(coverage)
            for horizon_position, horizon in enumerate(HORIZONS):
                calibration_scores = nonconformity_score(
                    calibration_prediction[:, horizon_position],
                    calibration_targets[:, horizon_position],
                    lower_position,
                    upper_position,
                )
                balanced_adjustment = max(
                    0.0,
                    weighted_quantile_higher(
                        calibration_scores, calibration_weights, coverage
                    ),
                )
                balanced_thresholds[mode][coverage_key].append(
                    balanced_adjustment
                )
                cluster_adjustments = sampled_thresholds(
                    calibration_scores, sampled_positions, coverage
                )

                for regime in (
                    "unseen_user_test",
                    "goldencheetah_frozen_external",
                ):
                    if regime == "goldencheetah_frozen_external" and mode != "zero_history":
                        continue
                    selected = masks[regime]
                    selected_prediction = prediction[selected, horizon_position]
                    selected_target = local["targets"][selected, horizon_position]
                    selected_users = local["users"][selected]
                    selected_sessions = local["sessions"][selected]
                    methods = {
                        "origin_pooled_cqr": float(
                            origin_thresholds[mode][coverage_key][horizon_position]
                        ),
                        "user_session_balanced_empirical": balanced_adjustment,
                    }
                    for method, adjustment in methods.items():
                        lower = np.clip(
                            selected_prediction[:, lower_position] - adjustment,
                            30.0,
                            240.0,
                        )
                        upper = np.clip(
                            selected_prediction[:, upper_position] + adjustment,
                            30.0,
                            240.0,
                        )
                        covered = (
                            (selected_target >= lower) & (selected_target <= upper)
                        ).astype(np.float64)
                        picp = hierarchical_metric(
                            covered, selected_users, selected_sessions
                        )
                        width = hierarchical_metric(
                            upper - lower, selected_users, selected_sessions
                        )
                        metric_rows.append(
                            {
                                "analysis_version": ANALYSIS_VERSION,
                                "source_model_version": SOURCE_MODEL_VERSION,
                                "regime": regime,
                                "mode": mode,
                                "horizon_seconds": horizon,
                                "nominal_coverage": coverage,
                                "calibration_method": method,
                                "conformal_adjustment_bpm": adjustment,
                                "picp": picp,
                                "absolute_coverage_error": abs(picp - coverage),
                                "mean_interval_width_bpm": width,
                                "users": int(len(np.unique(selected_users))),
                                "sessions": int(len(np.unique(selected_sessions))),
                                "origins": int(selected.sum()),
                                "aggregation": (
                                    "origin-within-session, session-within-user, "
                                    "equal-user mean"
                                ),
                                "formal_coverage_guarantee_claimed": False,
                            }
                        )
                        if method == "origin_pooled_cqr":
                            reference_row = reference_interval_row(
                                reference, regime, mode, horizon, coverage
                            )
                            maximum_reference_delta["picp"] = max(
                                maximum_reference_delta["picp"],
                                abs(picp - float(reference_row["picp"])),
                            )
                            maximum_reference_delta["width"] = max(
                                maximum_reference_delta["width"],
                                abs(
                                    width
                                    - float(
                                        reference_row["mean_interval_width_bpm"]
                                    )
                                ),
                            )

                    selected_scores = nonconformity_score(
                        selected_prediction,
                        selected_target,
                        lower_position,
                        upper_position,
                    )
                    _, replicate_user_coverage = per_user_weighted_cdf(
                        selected_scores,
                        selected_users,
                        selected_sessions,
                        cluster_adjustments,
                    )
                    replicate_picp = paired_replicate_user_bootstrap(
                        replicate_user_coverage,
                        stable_seed(
                            args.seed,
                            f"{regime}|{mode}|{coverage}|{horizon}|test-users",
                        ),
                    )
                    adjustment_low, adjustment_high = percentile_interval(
                        cluster_adjustments
                    )
                    picp_low, picp_high = percentile_interval(replicate_picp)
                    bootstrap_rows.append(
                        {
                            "analysis_version": ANALYSIS_VERSION,
                            "source_model_version": SOURCE_MODEL_VERSION,
                            "regime": regime,
                            "mode": mode,
                            "horizon_seconds": horizon,
                            "nominal_coverage": coverage,
                            "calibration_sensitivity": (
                                "bootstrap calibration users; one sampled session "
                                "and origin per sampled user"
                            ),
                            "adjustment_median_bpm": float(
                                np.median(cluster_adjustments)
                            ),
                            "adjustment_ci_low_bpm": adjustment_low,
                            "adjustment_ci_high_bpm": adjustment_high,
                            "picp_median": float(np.median(replicate_picp)),
                            "picp_ci_low": picp_low,
                            "picp_ci_high": picp_high,
                            "calibration_users": 97,
                            "evaluation_users": int(
                                len(np.unique(selected_users))
                            ),
                            "bootstrap_replicates": args.bootstrap_replicates,
                            "formal_coverage_guarantee_claimed": False,
                        }
                    )

    require(len(metric_rows) == 54, f"expected 54 metric rows, got {len(metric_rows)}")
    require(
        len(bootstrap_rows) == 27,
        f"expected 27 bootstrap rows, got {len(bootstrap_rows)}",
    )
    tolerance = 5e-6
    require(
        maximum_reference_delta["picp"] <= tolerance,
        f"PICP reference mismatch: {maximum_reference_delta['picp']}",
    )
    require(
        maximum_reference_delta["width"] <= tolerance,
        f"width reference mismatch: {maximum_reference_delta['width']}",
    )
    numeric_metric = pd.DataFrame(metric_rows)[
        [
            "conformal_adjustment_bpm",
            "picp",
            "absolute_coverage_error",
            "mean_interval_width_bpm",
        ]
    ].to_numpy()
    numeric_bootstrap = pd.DataFrame(bootstrap_rows)[
        [
            "adjustment_median_bpm",
            "adjustment_ci_low_bpm",
            "adjustment_ci_high_bpm",
            "picp_median",
            "picp_ci_low",
            "picp_ci_high",
        ]
    ].to_numpy()
    require(np.isfinite(numeric_metric).all(), "non-finite metric output")
    require(np.isfinite(numeric_bootstrap).all(), "non-finite bootstrap output")

    atomic_csv(args.output_metrics, metric_rows)
    atomic_csv(args.output_bootstrap, bootstrap_rows)
    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "source_model_version": SOURCE_MODEL_VERSION,
        "intended_use": (
            "Post hoc calibration-estimand and clustered-calibration sensitivity; "
            "no model retraining and no formal user-level coverage guarantee"
        ),
        "source_files": {
            "predictions": str(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
            "thresholds": str(args.thresholds),
            "thresholds_sha256": sha256_file(args.thresholds),
            "interval_reference": str(args.interval_reference),
            "interval_reference_sha256": sha256_file(args.interval_reference),
        },
        "row_counts": row_counts,
        "support": {
            name: {
                "origins": int(mask.sum()),
                "sessions": int(len(np.unique(local["sessions"][mask]))),
                "users": int(len(np.unique(local["users"][mask]))),
            }
            for name, mask in masks.items()
        },
        "user_session_balanced_thresholds": balanced_thresholds,
        "cluster_sensitivity": {
            "bootstrap_replicates": args.bootstrap_replicates,
            "base_seed": args.seed,
            "calibration_unit": (
                "bootstrap users with replacement; sample one session and one "
                "origin within each sampled user"
            ),
            "evaluation_uncertainty": (
                "paired replicate user bootstrap of session-balanced user coverage"
            ),
        },
        "maximum_absolute_reference_delta": maximum_reference_delta,
        "reference_tolerance": tolerance,
        "outputs": {
            "metrics": str(args.output_metrics),
            "metrics_sha256": sha256_file(args.output_metrics),
            "metrics_rows": len(metric_rows),
            "bootstrap": str(args.output_bootstrap),
            "bootstrap_sha256": sha256_file(args.output_bootstrap),
            "bootstrap_rows": len(bootstrap_rows),
        },
        "limitations": [
            (
                "The user-session-balanced threshold is a descriptive empirical "
                "quantile aligned to the reported hierarchical PICP estimand; it "
                "is not asserted to inherit split-conformal finite-sample coverage."
            ),
            (
                "The clustered sensitivity combines calibration-user resampling "
                "with one random session and origin per sampled user. Its percentile "
                "ranges describe sensitivity to calibration composition and test-user "
                "sampling, not conditional coverage for every user or sport."
            ),
            (
                "All calculations are conditional on the existing single-seed fitted "
                "checkpoint and do not quantify optimization variability."
            ),
        ],
        "all_assertions_pass": True,
    }
    atomic_json(args.audit_json, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate user/session-balanced empirical calibration and a clustered "
            "calibration sensitivity from frozen predictions."
        )
    )
    parser.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "outputs/predictions/uncertainty_user_generalization_v0_11_0.npz"
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(
            "outputs/models/uncertainty_user_generalization_v0_11_0/"
            "conformal_thresholds_v0_11_0.json"
        ),
    )
    parser.add_argument(
        "--interval-reference",
        type=Path,
        default=Path("outputs/results/uncertainty_interval_metrics_v0_11_0.csv"),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path(
            "outputs/results/user_balanced_calibration_metrics_v0_20_0.csv"
        ),
    )
    parser.add_argument(
        "--output-bootstrap",
        type=Path,
        default=Path(
            "outputs/results/clustered_calibration_bootstrap_v0_20_0.csv"
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/audit/clustered_calibration_v0_20_0.json"),
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
