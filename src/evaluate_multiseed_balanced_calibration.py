from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ANALYSIS_VERSION = "0.24.0"
SOURCE_EXPERIMENT_VERSION = "0.22.0"
HORIZONS = (60, 180, 300)
INTERVALS = {
    0.50: (2, 4),
    0.80: (1, 5),
    0.90: (0, 6),
}
SEEDS = (20260722, 20260723, 20260724, 20260725, 20260726)
PARTITION_CALIBRATION = 3
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1
MODES = {
    "history_informed": "history_quantiles",
    "zero_history": "zero_history_quantiles",
}
METHOD_ORIGIN = "origin_pooled_finite_sample_cqr"
METHOD_BALANCED = "equal_user_equal_session_empirical"
AGGREGATION = "origin-within-session, session-within-user, equal-user mean"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def weighted_quantile_higher(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape != weights.shape or len(values) == 0:
        raise ValueError("values and weights must be aligned non-empty vectors")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("non-finite weighted-quantile input")
    if (weights <= 0.0).any():
        raise ValueError("weights must be strictly positive")
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    target = probability * cumulative[-1]
    position = min(
        int(np.searchsorted(cumulative, target, side="left")),
        len(values) - 1,
    )
    return float(values[order[position]])


def origin_pooled_finite_sample_threshold(
    scores: np.ndarray, coverage: float
) -> float:
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a non-empty vector")
    probability = min(1.0, math.ceil((len(scores) + 1) * coverage) / len(scores))
    return max(
        0.0,
        float(np.quantile(scores, probability, method="higher")),
    )


def nonconformity_score(
    prediction: np.ndarray,
    target: np.ndarray,
    lower_position: int,
    upper_position: int,
) -> np.ndarray:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    require(
        prediction.ndim == 2
        and prediction.shape[1] == 7
        and target.shape == (len(prediction),),
        "invalid prediction/target shapes for nonconformity score",
    )
    return np.maximum(
        prediction[:, lower_position] - target,
        target - prediction[:, upper_position],
    )


@dataclass(frozen=True)
class Hierarchy:
    user_ids: np.ndarray
    user_inverse: np.ndarray
    within_user_origin_weights: np.ndarray
    users: int
    sessions: int
    origins: int

    @classmethod
    def build(cls, users: np.ndarray, sessions: np.ndarray) -> "Hierarchy":
        users = np.asarray(users)
        sessions = np.asarray(sessions)
        if users.ndim != 1 or sessions.shape != users.shape or len(users) == 0:
            raise ValueError("users and sessions must be aligned non-empty vectors")

        (
            unique_sessions,
            first,
            session_inverse,
            origins_per_session,
        ) = np.unique(
            sessions,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        session_users = users[first]
        if not np.array_equal(users, session_users[session_inverse]):
            raise ValueError("a session maps to more than one user")
        user_ids, session_user_inverse = np.unique(
            session_users, return_inverse=True
        )
        sessions_per_user = np.bincount(session_user_inverse)
        origin_user_inverse = session_user_inverse[session_inverse]
        within_user_weights = 1.0 / (
            sessions_per_user[origin_user_inverse]
            * origins_per_session[session_inverse]
        )
        user_weight_sums = np.bincount(
            origin_user_inverse,
            weights=within_user_weights,
            minlength=len(user_ids),
        )
        require(
            np.allclose(user_weight_sums, 1.0, rtol=0.0, atol=1e-12),
            "within-user hierarchical weights do not sum to one",
        )
        return cls(
            user_ids=user_ids,
            user_inverse=origin_user_inverse.astype(np.int32, copy=False),
            within_user_origin_weights=within_user_weights.astype(
                np.float64, copy=False
            ),
            users=int(len(user_ids)),
            sessions=int(len(unique_sessions)),
            origins=int(len(users)),
        )

    @property
    def global_origin_weights(self) -> np.ndarray:
        return self.within_user_origin_weights / self.users

    def per_user_mean(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.origins,):
            raise ValueError("value vector does not match hierarchy")
        result = np.bincount(
            self.user_inverse,
            weights=values * self.within_user_origin_weights,
            minlength=self.users,
        )
        require(len(result) == self.users, "per-user aggregation length mismatch")
        return result.astype(np.float64, copy=False)


def interval_per_user(
    prediction: np.ndarray,
    target: np.ndarray,
    hierarchy: Hierarchy,
    lower_position: int,
    upper_position: int,
    adjustment: float,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    require(
        prediction.shape == (hierarchy.origins, 7)
        and target.shape == (hierarchy.origins,),
        "interval inputs do not match evaluation hierarchy",
    )
    lower = np.clip(
        prediction[:, lower_position] - adjustment,
        30.0,
        240.0,
    )
    upper = np.clip(
        prediction[:, upper_position] + adjustment,
        30.0,
        240.0,
    )
    covered = ((target >= lower) & (target <= upper)).astype(np.float64)
    return hierarchy.per_user_mean(covered), hierarchy.per_user_mean(upper - lower)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values), [0.025, 0.975])
    return float(low), float(high)


def bootstrap_method_difference(
    origin_coverage: np.ndarray,
    balanced_coverage: np.ndarray,
    origin_width: np.ndarray,
    balanced_width: np.ndarray,
    nominal_coverage: float,
    replicates: int,
    seed: int,
    batch_size: int = 1_000,
) -> dict[str, float]:
    vectors = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            origin_coverage,
            balanced_coverage,
            origin_width,
            balanced_width,
        )
    )
    n_users = len(vectors[0])
    if n_users < 2 or any(value.shape != (n_users,) for value in vectors):
        raise ValueError("bootstrap inputs must be aligned vectors with >=2 users")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    generator = np.random.default_rng(seed)
    delta_picp = np.empty(replicates, dtype=np.float64)
    delta_absolute_error = np.empty(replicates, dtype=np.float64)
    delta_width = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        end = min(replicates, start + batch_size)
        indices = generator.integers(0, n_users, size=(end - start, n_users))
        origin_picp = vectors[0][indices].mean(axis=1)
        balanced_picp = vectors[1][indices].mean(axis=1)
        delta_picp[start:end] = balanced_picp - origin_picp
        delta_absolute_error[start:end] = (
            np.abs(balanced_picp - nominal_coverage)
            - np.abs(origin_picp - nominal_coverage)
        )
        delta_width[start:end] = (
            vectors[3][indices].mean(axis=1)
            - vectors[2][indices].mean(axis=1)
        )

    picp_low, picp_high = percentile_interval(delta_picp)
    error_low, error_high = percentile_interval(delta_absolute_error)
    width_low, width_high = percentile_interval(delta_width)
    origin_picp = float(vectors[0].mean())
    balanced_picp = float(vectors[1].mean())
    return {
        "delta_picp": balanced_picp - origin_picp,
        "delta_picp_ci_low": picp_low,
        "delta_picp_ci_high": picp_high,
        "delta_absolute_coverage_error": (
            abs(balanced_picp - nominal_coverage)
            - abs(origin_picp - nominal_coverage)
        ),
        "delta_absolute_coverage_error_ci_low": error_low,
        "delta_absolute_coverage_error_ci_high": error_high,
        "delta_mean_interval_width_bpm": float(
            vectors[3].mean() - vectors[2].mean()
        ),
        "delta_mean_interval_width_ci_low_bpm": width_low,
        "delta_mean_interval_width_ci_high_bpm": width_high,
    }


def load_thresholds(path: Path) -> tuple[dict[str, dict[str, list[float]]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = payload.get("thresholds", payload)
    for mode in MODES:
        require(mode in root, f"{path}: missing threshold mode {mode}")
        for coverage in INTERVALS:
            key = str(coverage)
            require(
                key in root[mode] and len(root[mode][key]) == len(HORIZONS),
                f"{path}: invalid {mode}/{key} thresholds",
            )
    return root, payload


def load_reference_rows(paths: Iterable[Path]) -> dict[tuple, dict[str, str]]:
    rows: dict[tuple, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                calibrated = str(row.get("calibrated", "")).strip().lower()
                if calibrated not in {"true", "1"}:
                    continue
                key = (
                    row["regime"],
                    row["mode"],
                    int(row["horizon_seconds"]),
                    float(row["nominal_coverage"]),
                )
                require(key not in rows, f"duplicate reference interval row: {key}")
                rows[key] = row
    return rows


def validate_prediction_array(
    prediction: np.ndarray, expected_rows: int, label: str
) -> None:
    require(
        prediction.shape == (expected_rows, len(HORIZONS), 7),
        f"{label}: prediction shape {prediction.shape}",
    )
    require(np.isfinite(prediction).all(), f"{label}: non-finite predictions")
    require(
        int((np.diff(prediction, axis=2) < -1e-6).sum()) == 0,
        f"{label}: quantile crossings",
    )
    require(
        bool(((prediction >= 30.0) & (prediction <= 240.0)).all()),
        f"{label}: prediction range failure",
    )


def summarize_median_min_max(
    rows: list[dict[str, object]],
    group_columns: tuple[str, ...],
    metric_columns: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[column] for column in group_columns)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda value: tuple(str(item) for item in value)):
        group = groups[key]
        result: dict[str, object] = dict(zip(group_columns, key, strict=True))
        result["seed_count"] = len(group)
        for column in metric_columns:
            values = np.asarray([float(row[column]) for row in group])
            require(np.isfinite(values).all(), f"non-finite summary metric {column}")
            result[f"{column}_median"] = float(np.median(values))
            result[f"{column}_min"] = float(values.min())
            result[f"{column}_max"] = float(values.max())
        output.append(result)
    return output


def inspect_strict_temporal_artifacts(
    temporal_dir: Path, temporal_partition: np.ndarray
) -> dict[str, object]:
    archives: list[dict[str, object]] = []
    calibration_rows_present = 0
    for path in sorted(temporal_dir.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as source:
            keys = sorted(source.files)
            if "row_index" in source.files:
                rows = np.asarray(source["row_index"], dtype=np.int64)
                partitions, counts = np.unique(
                    temporal_partition[rows], return_counts=True
                )
                partition_counts = {
                    str(int(partition)): int(count)
                    for partition, count in zip(partitions, counts, strict=True)
                }
                calibration_rows_present += int(
                    (temporal_partition[rows] == PARTITION_CALIBRATION).sum()
                )
            else:
                rows = np.empty(0, dtype=np.int64)
                partition_counts = {}
        archives.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "keys": keys,
                "row_count": int(len(rows)),
                "strict_temporal_partition_counts": partition_counts,
            }
        )
    threshold_path = temporal_dir / "model" / "conformal_thresholds.json"
    require(threshold_path.exists(), f"missing strict-temporal thresholds: {threshold_path}")
    return {
        "directory": str(temporal_dir),
        "prediction_archives": archives,
        "prediction_archive_count": len(archives),
        "persisted_calibration_prediction_rows": calibration_rows_present,
        "thresholds_path": str(threshold_path),
        "thresholds_sha256": sha256_file(threshold_path),
        "balanced_recalibration_status": "unavailable_from_persisted_predictions",
        "missing_required_artifact": (
            "strict-temporal calibration quantile predictions aligned to "
            "temporal_partition_strict == 3"
        ),
        "checkpoint_regeneration_performed": False,
    }


def method_metric_row(
    *,
    seed: int,
    regime: str,
    mode: str,
    horizon: int,
    coverage: float,
    method: str,
    adjustment: float,
    per_user_coverage: np.ndarray,
    per_user_width: np.ndarray,
    calibration_hierarchy: Hierarchy,
    evaluation_hierarchy: Hierarchy,
) -> dict[str, object]:
    picp = float(per_user_coverage.mean())
    width = float(per_user_width.mean())
    return {
        "analysis_version": ANALYSIS_VERSION,
        "source_experiment_version": SOURCE_EXPERIMENT_VERSION,
        "seed": seed,
        "regime": regime,
        "mode": mode,
        "horizon_seconds": horizon,
        "nominal_coverage": coverage,
        "calibration_method": method,
        "conformal_adjustment_bpm": adjustment,
        "picp": picp,
        "absolute_coverage_error": abs(picp - coverage),
        "mean_interval_width_bpm": width,
        "calibration_users": calibration_hierarchy.users,
        "calibration_sessions": calibration_hierarchy.sessions,
        "calibration_origins": calibration_hierarchy.origins,
        "evaluation_users": evaluation_hierarchy.users,
        "evaluation_sessions": evaluation_hierarchy.sessions,
        "evaluation_origins": evaluation_hierarchy.origins,
        "evaluation_aggregation": AGGREGATION,
        "formal_equal_user_coverage_guarantee_claimed": False,
        "external_target_data_used_for_calibration": False,
    }


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
        "sessions": np.load(
            args.array_dir / "session_index.npy", mmap_mode="r"
        ),
    }
    row_counts = {name: int(len(value)) for name, value in arrays.items()}
    require(len(set(row_counts.values())) == 1, f"array length mismatch: {row_counts}")

    development_rows = np.flatnonzero(
        (arrays["dataset"] == 0) & (arrays["evaluation"] == 1)
    )
    calibration_positions = np.flatnonzero(
        arrays["unseen"][development_rows] == PARTITION_CALIBRATION
    )
    test_positions = np.flatnonzero(
        arrays["unseen"][development_rows] == PARTITION_TEST
    )
    external_rows = np.flatnonzero(
        (arrays["dataset"] == 1) & (arrays["external"] == EXTERNAL_FROZEN)
    )
    require(len(development_rows) == 1_001_128, "unexpected development row count")
    require(len(calibration_positions) == 85_247, "unexpected calibration row count")
    require(len(test_positions) == 101_184, "unexpected unseen-user test row count")
    require(len(external_rows) == 531_725, "unexpected frozen-external row count")

    calibration_rows = development_rows[calibration_positions]
    test_rows = development_rows[test_positions]
    calibration_hierarchy = Hierarchy.build(
        arrays["users"][calibration_rows], arrays["sessions"][calibration_rows]
    )
    test_hierarchy = Hierarchy.build(
        arrays["users"][test_rows], arrays["sessions"][test_rows]
    )
    external_hierarchy = Hierarchy.build(
        arrays["users"][external_rows], arrays["sessions"][external_rows]
    )
    require(calibration_hierarchy.users == 97, "unexpected calibration users")
    require(test_hierarchy.users == 105, "unexpected unseen-user test users")
    require(external_hierarchy.users == 144, "unexpected external users")

    calibration_targets = np.asarray(arrays["targets"][calibration_rows])
    test_targets = np.asarray(arrays["targets"][test_rows])
    external_targets = np.asarray(arrays["targets"][external_rows])
    require(np.isfinite(calibration_targets).all(), "non-finite calibration targets")
    require(np.isfinite(test_targets).all(), "non-finite test targets")
    require(np.isfinite(external_targets).all(), "non-finite external targets")

    metric_rows: list[dict[str, object]] = []
    conditional_difference_rows: list[dict[str, object]] = []
    per_user_fixed_seed: dict[
        tuple[str, str, int, float], dict[str, object]
    ] = {}
    source_records: dict[str, object] = {}
    strict_records: dict[str, object] = {}
    maximum_deltas = {
        "stored_vs_recomputed_origin_threshold_bpm": 0.0,
        "stored_reference_picp": 0.0,
        "stored_reference_width_bpm": 0.0,
        "stored_reference_threshold_bpm": 0.0,
    }

    for seed in args.seeds:
        seed_root = args.experiment_root / f"seed_{seed}"
        unseen_dir = seed_root / "unseen_main"
        temporal_dir = seed_root / "temporal_main"
        development_path = unseen_dir / "development_predictions.npz"
        external_path = unseen_dir / "external_predictions.npz"
        threshold_path = unseen_dir / "model" / "conformal_thresholds_v0_11_0.json"
        development_reference = unseen_dir / "development_interval_metrics.csv"
        external_reference = unseen_dir / "external_interval_metrics.csv"
        for path in (
            development_path,
            external_path,
            threshold_path,
            development_reference,
            external_reference,
        ):
            require(path.exists(), f"missing seed artifact: {path}")

        thresholds, threshold_payload = load_thresholds(threshold_path)
        require(
            int(threshold_payload.get("calibration_rows", -1))
            == calibration_hierarchy.origins,
            f"seed {seed}: threshold calibration-row count mismatch",
        )
        require(
            int(threshold_payload.get("calibration_users", -1))
            == calibration_hierarchy.users,
            f"seed {seed}: threshold calibration-user count mismatch",
        )
        references = load_reference_rows(
            (development_reference, external_reference)
        )

        with np.load(development_path, allow_pickle=False) as source:
            require(
                {"row_index", *MODES.values()}.issubset(source.files),
                f"seed {seed}: development archive missing fields",
            )
            row_index = np.asarray(source["row_index"], dtype=np.int64)
            require(
                np.array_equal(row_index, development_rows),
                f"seed {seed}: development row mapping mismatch",
            )

            with np.load(external_path, allow_pickle=False) as external_source:
                require(
                    {"row_index", "zero_history_quantiles"}.issubset(
                        external_source.files
                    ),
                    f"seed {seed}: external archive missing fields",
                )
                external_index = np.asarray(
                    external_source["row_index"], dtype=np.int64
                )
                require(
                    np.array_equal(external_index, external_rows),
                    f"seed {seed}: external row mapping mismatch",
                )
                external_prediction = np.asarray(
                    external_source["zero_history_quantiles"]
                )
                validate_prediction_array(
                    external_prediction,
                    len(external_rows),
                    f"seed {seed} external zero-history",
                )

                for mode, prediction_key in MODES.items():
                    prediction = np.asarray(source[prediction_key])
                    validate_prediction_array(
                        prediction,
                        len(development_rows),
                        f"seed {seed} development {mode}",
                    )
                    calibration_prediction = prediction[calibration_positions]
                    test_prediction = prediction[test_positions]

                    for coverage, (lower_position, upper_position) in INTERVALS.items():
                        coverage_key = str(coverage)
                        for horizon_position, horizon in enumerate(HORIZONS):
                            scores = nonconformity_score(
                                calibration_prediction[:, horizon_position],
                                calibration_targets[:, horizon_position],
                                lower_position,
                                upper_position,
                            )
                            stored_origin_threshold = float(
                                thresholds[mode][coverage_key][horizon_position]
                            )
                            recomputed_origin_threshold = (
                                origin_pooled_finite_sample_threshold(scores, coverage)
                            )
                            maximum_deltas[
                                "stored_vs_recomputed_origin_threshold_bpm"
                            ] = max(
                                maximum_deltas[
                                    "stored_vs_recomputed_origin_threshold_bpm"
                                ],
                                abs(
                                    stored_origin_threshold
                                    - recomputed_origin_threshold
                                ),
                            )
                            balanced_threshold = max(
                                0.0,
                                weighted_quantile_higher(
                                    scores,
                                    calibration_hierarchy.global_origin_weights,
                                    coverage,
                                ),
                            )

                            regimes = [
                                (
                                    "unseen_user_test",
                                    test_prediction[:, horizon_position],
                                    test_targets[:, horizon_position],
                                    test_hierarchy,
                                )
                            ]
                            if mode == "zero_history":
                                regimes.append(
                                    (
                                        "goldencheetah_frozen_external",
                                        external_prediction[:, horizon_position],
                                        external_targets[:, horizon_position],
                                        external_hierarchy,
                                    )
                                )

                            for (
                                regime,
                                regime_prediction,
                                regime_target,
                                regime_hierarchy,
                            ) in regimes:
                                origin_coverage, origin_width = interval_per_user(
                                    regime_prediction,
                                    regime_target,
                                    regime_hierarchy,
                                    lower_position,
                                    upper_position,
                                    stored_origin_threshold,
                                )
                                balanced_coverage, balanced_width = interval_per_user(
                                    regime_prediction,
                                    regime_target,
                                    regime_hierarchy,
                                    lower_position,
                                    upper_position,
                                    balanced_threshold,
                                )
                                origin_row = method_metric_row(
                                    seed=seed,
                                    regime=regime,
                                    mode=mode,
                                    horizon=horizon,
                                    coverage=coverage,
                                    method=METHOD_ORIGIN,
                                    adjustment=stored_origin_threshold,
                                    per_user_coverage=origin_coverage,
                                    per_user_width=origin_width,
                                    calibration_hierarchy=calibration_hierarchy,
                                    evaluation_hierarchy=regime_hierarchy,
                                )
                                balanced_row = method_metric_row(
                                    seed=seed,
                                    regime=regime,
                                    mode=mode,
                                    horizon=horizon,
                                    coverage=coverage,
                                    method=METHOD_BALANCED,
                                    adjustment=balanced_threshold,
                                    per_user_coverage=balanced_coverage,
                                    per_user_width=balanced_width,
                                    calibration_hierarchy=calibration_hierarchy,
                                    evaluation_hierarchy=regime_hierarchy,
                                )
                                metric_rows.extend((origin_row, balanced_row))

                                reference_key = (regime, mode, horizon, coverage)
                                require(
                                    reference_key in references,
                                    f"seed {seed}: missing reference {reference_key}",
                                )
                                reference = references[reference_key]
                                maximum_deltas["stored_reference_picp"] = max(
                                    maximum_deltas["stored_reference_picp"],
                                    abs(
                                        float(origin_row["picp"])
                                        - float(reference["picp"])
                                    ),
                                )
                                maximum_deltas[
                                    "stored_reference_width_bpm"
                                ] = max(
                                    maximum_deltas[
                                        "stored_reference_width_bpm"
                                    ],
                                    abs(
                                        float(
                                            origin_row[
                                                "mean_interval_width_bpm"
                                            ]
                                        )
                                        - float(
                                            reference["mean_interval_width_bpm"]
                                        )
                                    ),
                                )
                                maximum_deltas[
                                    "stored_reference_threshold_bpm"
                                ] = max(
                                    maximum_deltas[
                                        "stored_reference_threshold_bpm"
                                    ],
                                    abs(
                                        stored_origin_threshold
                                        - float(
                                            reference[
                                                "conformal_adjustment_bpm"
                                            ]
                                        )
                                    ),
                                )

                                comparison = bootstrap_method_difference(
                                    origin_coverage,
                                    balanced_coverage,
                                    origin_width,
                                    balanced_width,
                                    coverage,
                                    args.bootstrap_replicates,
                                    stable_seed(
                                        args.bootstrap_seed,
                                        (
                                            f"{seed}|{regime}|{mode}|{horizon}|"
                                            f"{coverage}"
                                        ),
                                    ),
                                )
                                conditional_difference_rows.append(
                                    {
                                        "analysis_version": ANALYSIS_VERSION,
                                        "scope": "conditional_on_seed",
                                        "seed": seed,
                                        "regime": regime,
                                        "mode": mode,
                                        "horizon_seconds": horizon,
                                        "nominal_coverage": coverage,
                                        "origin_pooled_adjustment_bpm": (
                                            stored_origin_threshold
                                        ),
                                        "balanced_adjustment_bpm": balanced_threshold,
                                        "delta_adjustment_bpm": (
                                            balanced_threshold
                                            - stored_origin_threshold
                                        ),
                                        **comparison,
                                        "evaluation_users": regime_hierarchy.users,
                                        "bootstrap_replicates": (
                                            args.bootstrap_replicates
                                        ),
                                        "bootstrap_unit": "evaluation user",
                                        "calibration_thresholds_fixed": True,
                                        "optimization_seed_variability_in_ci": False,
                                    }
                                )

                                fixed_key = (regime, mode, horizon, coverage)
                                bucket = per_user_fixed_seed.setdefault(
                                    fixed_key,
                                    {
                                        "user_ids": regime_hierarchy.user_ids,
                                        "origin_coverage": [],
                                        "balanced_coverage": [],
                                        "origin_width": [],
                                        "balanced_width": [],
                                    },
                                )
                                require(
                                    np.array_equal(
                                        bucket["user_ids"],
                                        regime_hierarchy.user_ids,
                                    ),
                                    f"seed {seed}: evaluation-user mapping changed",
                                )
                                for name, value in (
                                    ("origin_coverage", origin_coverage),
                                    ("balanced_coverage", balanced_coverage),
                                    ("origin_width", origin_width),
                                    ("balanced_width", balanced_width),
                                ):
                                    bucket[name].append(value)

                    del prediction
                del external_prediction

        strict_records[str(seed)] = inspect_strict_temporal_artifacts(
            temporal_dir, arrays["temporal"]
        )
        source_records[str(seed)] = {
            "development_predictions": {
                "path": str(development_path),
                "sha256": sha256_file(development_path),
            },
            "external_predictions": {
                "path": str(external_path),
                "sha256": sha256_file(external_path),
            },
            "origin_pooled_thresholds": {
                "path": str(threshold_path),
                "sha256": sha256_file(threshold_path),
            },
            "development_interval_reference": {
                "path": str(development_reference),
                "sha256": sha256_file(development_reference),
            },
            "external_interval_reference": {
                "path": str(external_reference),
                "sha256": sha256_file(external_reference),
            },
        }

    expected_metric_rows = len(args.seeds) * 27 * 2
    expected_conditional_rows = len(args.seeds) * 27
    require(
        len(metric_rows) == expected_metric_rows,
        f"expected {expected_metric_rows} metric rows, got {len(metric_rows)}",
    )
    require(
        len(conditional_difference_rows) == expected_conditional_rows,
        (
            f"expected {expected_conditional_rows} conditional difference rows, "
            f"got {len(conditional_difference_rows)}"
        ),
    )

    fixed_seed_rows: list[dict[str, object]] = []
    for (regime, mode, horizon, coverage), bucket in sorted(
        per_user_fixed_seed.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        stacks = {
            name: np.stack(bucket[name]).mean(axis=0)
            for name in (
                "origin_coverage",
                "balanced_coverage",
                "origin_width",
                "balanced_width",
            )
        }
        require(
            all(np.asarray(bucket[name]).shape[0] == len(args.seeds) for name in stacks),
            f"incomplete fixed-seed stack for {(regime, mode, horizon, coverage)}",
        )
        comparison = bootstrap_method_difference(
            stacks["origin_coverage"],
            stacks["balanced_coverage"],
            stacks["origin_width"],
            stacks["balanced_width"],
            coverage,
            args.bootstrap_replicates,
            stable_seed(
                args.bootstrap_seed,
                f"fixed-five|{regime}|{mode}|{horizon}|{coverage}",
            ),
        )
        fixed_seed_rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "scope": "user_bootstrap_after_averaging_five_fixed_seeds",
                "seed": "all_five_fixed",
                "regime": regime,
                "mode": mode,
                "horizon_seconds": horizon,
                "nominal_coverage": coverage,
                "origin_pooled_adjustment_bpm": "not_single_valued_across_seeds",
                "balanced_adjustment_bpm": "not_single_valued_across_seeds",
                "delta_adjustment_bpm": "not_single_valued_across_seeds",
                **comparison,
                "evaluation_users": len(bucket["user_ids"]),
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_unit": "evaluation user",
                "calibration_thresholds_fixed": True,
                "optimization_seed_variability_in_ci": False,
            }
        )

    difference_rows = conditional_difference_rows + fixed_seed_rows
    require(len(fixed_seed_rows) == 27, "expected 27 fixed-seed difference rows")

    metric_summary = summarize_median_min_max(
        metric_rows,
        (
            "regime",
            "mode",
            "horizon_seconds",
            "nominal_coverage",
            "calibration_method",
        ),
        (
            "conformal_adjustment_bpm",
            "picp",
            "absolute_coverage_error",
            "mean_interval_width_bpm",
        ),
    )
    difference_summary = summarize_median_min_max(
        conditional_difference_rows,
        ("regime", "mode", "horizon_seconds", "nominal_coverage"),
        (
            "delta_adjustment_bpm",
            "delta_picp",
            "delta_absolute_coverage_error",
            "delta_mean_interval_width_bpm",
        ),
    )
    require(len(metric_summary) == 54, "expected 54 method summary rows")
    require(len(difference_summary) == 27, "expected 27 difference summary rows")

    for name, value in maximum_deltas.items():
        require(value <= 5e-6, f"reference validation failed for {name}: {value}")
    numeric_metric_fields = (
        "conformal_adjustment_bpm",
        "picp",
        "absolute_coverage_error",
        "mean_interval_width_bpm",
    )
    require(
        np.isfinite(
            np.asarray(
                [
                    [float(row[field]) for field in numeric_metric_fields]
                    for row in metric_rows
                ]
            )
        ).all(),
        "non-finite metric output",
    )

    atomic_csv(args.output_metrics, metric_rows)
    atomic_csv(args.output_summary, metric_summary)
    atomic_csv(args.output_differences, difference_rows)
    atomic_csv(args.output_difference_summary, difference_summary)

    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "source_experiment_version": SOURCE_EXPERIMENT_VERSION,
        "generated_at_utc": utc_now(),
        "intended_use": (
            "Frozen-prediction comparison of the persisted origin-pooled "
            "finite-sample CQR threshold with an equal-user/equal-session "
            "empirical calibration estimand across five main-model seeds"
        ),
        "seeds": list(args.seeds),
        "no_retraining": True,
        "no_checkpoint_inference": True,
        "external_adaptation_or_recalibration": False,
        "external_threshold_source": (
            "Endomondo unseen-user calibration partition; applied unchanged "
            "to GoldenCheetah frozen external predictions"
        ),
        "array_row_counts": row_counts,
        "support": {
            "calibration": {
                "origins": calibration_hierarchy.origins,
                "sessions": calibration_hierarchy.sessions,
                "users": calibration_hierarchy.users,
            },
            "unseen_user_test": {
                "origins": test_hierarchy.origins,
                "sessions": test_hierarchy.sessions,
                "users": test_hierarchy.users,
            },
            "goldencheetah_frozen_external": {
                "origins": external_hierarchy.origins,
                "sessions": external_hierarchy.sessions,
                "users": external_hierarchy.users,
            },
        },
        "calibration_estimands": {
            METHOD_ORIGIN: (
                "finite-sample higher quantile over all calibration origins; "
                "nonnegative expansion"
            ),
            METHOD_BALANCED: (
                "weighted empirical higher quantile with equal user weight, "
                "equal session weight within user, and equal origin weight "
                "within session; nonnegative expansion"
            ),
        },
        "evaluation_aggregation": AGGREGATION,
        "source_files_by_seed": source_records,
        "strict_temporal_availability": {
            "status": "not_recomputed",
            "reason": (
                "Every persisted strict-temporal prediction archive contains "
                "test rows only (temporal_partition_strict == 4); no aligned "
                "calibration-partition quantile predictions were persisted."
            ),
            "by_seed": strict_records,
        },
        "validation": {
            "development_row_mapping_exact_for_all_seeds": True,
            "external_row_mapping_exact_for_all_seeds": True,
            "quantile_crossing_failures": 0,
            "nonfinite_prediction_values": 0,
            "prediction_range_failures": 0,
            "maximum_absolute_deltas": maximum_deltas,
            "reference_tolerance": 5e-6,
            "all_assertions_pass": True,
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "base_seed": args.bootstrap_seed,
            "inferential_unit": "evaluation user",
            "within_user_aggregation": (
                "origins averaged within session, sessions averaged within user"
            ),
            "conditional_seed_intervals": (
                "paired resampling of evaluation users with both calibration "
                "thresholds held fixed"
            ),
            "five_fixed_seed_intervals": (
                "per-user method metrics averaged across the five observed "
                "optimization seeds before paired user resampling"
            ),
            "calibration_sample_variability_in_ci": False,
            "optimization_seed_variability_in_ci": False,
        },
        "outputs": {
            "per_seed_metrics": {
                "path": str(args.output_metrics),
                "sha256": sha256_file(args.output_metrics),
                "rows": len(metric_rows),
            },
            "method_median_min_max": {
                "path": str(args.output_summary),
                "sha256": sha256_file(args.output_summary),
                "rows": len(metric_summary),
            },
            "paired_differences_and_user_bootstrap": {
                "path": str(args.output_differences),
                "sha256": sha256_file(args.output_differences),
                "rows": len(difference_rows),
            },
            "difference_median_min_max": {
                "path": str(args.output_difference_summary),
                "sha256": sha256_file(args.output_difference_summary),
                "rows": len(difference_summary),
            },
        },
        "limitations": [
            (
                "The equal-user/equal-session threshold is a descriptive "
                "weighted empirical estimand; no finite-sample equal-user or "
                "conditional coverage guarantee is claimed."
            ),
            (
                "Evaluation-user bootstrap intervals condition on the fitted "
                "checkpoint and its two already-estimated calibration thresholds; "
                "they do not include calibration-sample uncertainty."
            ),
            (
                "The fixed-five-seed bootstrap treats the five completed seeds "
                "as fixed and therefore does not quantify the population of "
                "possible optimization seeds."
            ),
            (
                "Strict-temporal balanced calibration cannot be reconstructed "
                "from persisted artifacts without new checkpoint inference; it "
                "is deliberately reported as unavailable rather than estimated "
                "from test targets."
            ),
            (
                "GoldenCheetah target values are used only for frozen evaluation; "
                "neither threshold is estimated or selected on external targets."
            ),
        ],
        "all_assertions_pass": True,
    }
    atomic_json(args.audit_json, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare origin-pooled and equal-user/equal-session calibration "
            "estimands across the five frozen main-model seeds."
        )
    )
    parser.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/q1_multiseed_v0_21_0"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path(
            "outputs/results/multiseed_balanced_calibration_per_seed_v0_24_0.csv"
        ),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path(
            "outputs/results/multiseed_balanced_calibration_summary_v0_24_0.csv"
        ),
    )
    parser.add_argument(
        "--output-differences",
        type=Path,
        default=Path(
            "outputs/results/multiseed_balanced_calibration_differences_v0_24_0.csv"
        ),
    )
    parser.add_argument(
        "--output-difference-summary",
        type=Path,
        default=Path(
            "outputs/results/multiseed_balanced_calibration_difference_summary_v0_24_0.csv"
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/audit/multiseed_balanced_calibration_v0_24_0.json"
        ),
    )
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    require(args.seeds == SEEDS, f"expected frozen seed set {SEEDS}")
    if args.bootstrap_replicates < 1_000:
        raise ValueError("at least 1,000 bootstrap replicates are required")
    audit = analyze(args)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
