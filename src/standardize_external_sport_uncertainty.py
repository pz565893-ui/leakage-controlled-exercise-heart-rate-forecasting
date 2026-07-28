from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_external_sport_uncertainty import (
    calibrated_bounds,
    load_thresholds,
    weighted_interval_score,
)


ANALYSIS_VERSION = "0.24.0"
SOURCE_MODEL_VERSION = "0.11.0"
REFERENCE_SEED = 20260722
HORIZONS = (60, 180, 300)
PARTITION_CALIBRATION = 3
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1
SPORTS = {
    1: "outdoor_cycling",
    2: "indoor_virtual_cycling",
    3: "running",
}
METRIC_UNITS = {
    "picp_90": "proportion",
    "mean_90_interval_width_bpm": "bpm",
    "weighted_interval_score": "bpm",
}
PUBLIC_RESULT_COLUMNS = (
    "analysis_version",
    "source_model_version",
    "reference_seed",
    "comparison_scope",
    "sport_family",
    "horizon_seconds",
    "metric",
    "unit",
    "internal_source",
    "internal_mode",
    "internal_estimate",
    "internal_ci_low",
    "internal_ci_high",
    "external_source",
    "external_mode",
    "external_estimate",
    "external_ci_low",
    "external_ci_high",
    "external_minus_internal",
    "difference_ci_low",
    "difference_ci_high",
    "internal_users",
    "external_users",
    "internal_sessions",
    "external_sessions",
    "internal_origins",
    "external_origins",
    "bootstrap_replicates",
    "bootstrap_unit",
    "aggregation",
    "post_cqr",
    "causal_source_or_device_effect_claimed",
    "user_level_conformal_guarantee_claimed",
)


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
        writer = csv.DictWriter(handle, fieldnames=list(PUBLIC_RESULT_COLUMNS))
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
    lower, upper = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(lower), float(upper)


def bootstrap_indices(n_users: int, replicates: int, seed: int) -> np.ndarray:
    if n_users < 2:
        raise ValueError("at least two users are required")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    generator = np.random.default_rng(seed)
    return generator.integers(0, n_users, size=(replicates, n_users))


def bootstrap_matrix_mean(
    user_values: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 2_000,
) -> np.ndarray:
    values = np.asarray(user_values, dtype=np.float64)
    indices = np.asarray(indices)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("user_values must be a two-dimensional user-by-metric matrix")
    if indices.ndim != 2 or indices.shape[1] != len(values):
        raise ValueError("bootstrap index shape does not match user_values")
    if not np.isfinite(values).all():
        raise ValueError("natural-composition user metrics must be finite")
    result = np.empty((len(indices), values.shape[1]), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        end = min(len(indices), start + batch_size)
        result[start:end] = values[indices[start:end]].mean(axis=1)
    return result


def bootstrap_family_nanmean(
    user_family_values: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 500,
) -> np.ndarray:
    values = np.asarray(user_family_values, dtype=np.float64)
    indices = np.asarray(indices)
    if values.ndim != 3 or len(values) < 2:
        raise ValueError(
            "user_family_values must be a user-by-family-by-metric cube"
        )
    if indices.ndim != 2 or indices.shape[1] != len(values):
        raise ValueError("bootstrap index shape does not match user_family_values")
    result = np.empty(
        (len(indices), values.shape[1], values.shape[2]), dtype=np.float64
    )
    for start in range(0, len(indices), batch_size):
        end = min(len(indices), start + batch_size)
        sampled = values[indices[start:end]]
        supported = np.isfinite(sampled)
        counts = supported.sum(axis=1)
        if (counts == 0).any():
            raise AssertionError("a bootstrap replicate lost all users for a family")
        result[start:end] = np.nansum(sampled, axis=1) / counts
    require(np.isfinite(result).all(), "non-finite family bootstrap estimate")
    return result


def hierarchical_user_metrics(
    metrics: dict[str, np.ndarray],
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.DataFrame:
    if not metrics:
        raise ValueError("at least one metric is required")
    lengths = {len(np.asarray(value)) for value in metrics.values()}
    lengths.update((len(users), len(sessions)))
    if len(lengths) != 1:
        raise ValueError(f"metric and identifier length mismatch: {lengths}")
    metric_names = list(metrics)
    frame = pd.DataFrame(
        {
            **{name: np.asarray(value) for name, value in metrics.items()},
            "user": np.asarray(users),
            "session": np.asarray(sessions),
        }
    )
    session_level = frame.groupby(["user", "session"], sort=False)[
        metric_names
    ].mean()
    return session_level.groupby(level="user", sort=False).mean()


def hierarchical_user_family_metrics(
    metrics: dict[str, np.ndarray],
    users: np.ndarray,
    sessions: np.ndarray,
    sports: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not metrics:
        raise ValueError("at least one metric is required")
    lengths = {len(np.asarray(value)) for value in metrics.values()}
    lengths.update((len(users), len(sessions), len(sports)))
    if len(lengths) != 1:
        raise ValueError(f"metric and identifier length mismatch: {lengths}")
    metric_names = list(metrics)
    frame = pd.DataFrame(
        {
            **{name: np.asarray(value) for name, value in metrics.items()},
            "user": np.asarray(users),
            "session": np.asarray(sessions),
            "sport": np.asarray(sports),
        }
    )
    session_level = frame.groupby(
        ["user", "sport", "session"], sort=False
    )[metric_names].mean()
    user_family = session_level.groupby(level=["user", "sport"], sort=False)[
        metric_names
    ].mean()
    unique_users = np.unique(np.asarray(users))
    aligned_index = pd.MultiIndex.from_product(
        [unique_users, list(SPORTS)], names=["user", "sport"]
    )
    aligned = user_family.reindex(aligned_index)
    cube = aligned.to_numpy(dtype=np.float64).reshape(
        len(unique_users), len(SPORTS), len(metric_names)
    )
    return unique_users, cube


def weighted_family_metrics(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(SPORTS),):
        raise ValueError("family weights have the wrong shape")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("family weights must be finite and nonnegative")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("family weights must sum to one")
    if values.ndim < 2 or values.shape[-2] != len(SPORTS):
        raise ValueError("family values must use the penultimate axis for family")
    result = np.tensordot(values, weights, axes=([-2], [0]))
    require(np.isfinite(result).all(), "non-finite weighted family estimate")
    return result


def origin_uncertainty_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    thresholds: dict[str, list[float]],
    horizon_position: int,
) -> dict[str, np.ndarray]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.ndim != 2 or prediction.shape[1] != 7:
        raise ValueError("prediction must have shape (origins, 7)")
    if target.shape != (len(prediction),):
        raise ValueError("target shape does not match prediction")
    bounds = calibrated_bounds(prediction, thresholds, horizon_position)
    lower_90, upper_90 = bounds[0.90]
    return {
        "picp_90": ((target >= lower_90) & (target <= upper_90)).astype(
            np.float64
        ),
        "mean_90_interval_width_bpm": (upper_90 - lower_90).astype(
            np.float64
        ),
        "weighted_interval_score": weighted_interval_score(
            prediction[:, 3], target, bounds
        ).astype(np.float64),
    }


def comparison_rows(
    *,
    scope: str,
    family: str,
    horizon: int,
    internal_point: np.ndarray,
    external_point: np.ndarray,
    internal_bootstrap: np.ndarray,
    external_bootstrap: np.ndarray,
    internal_support: dict[str, int],
    external_support: dict[str, int],
    replicates: int,
) -> list[dict[str, object]]:
    metric_names = list(METRIC_UNITS)
    internal_point = np.asarray(internal_point, dtype=np.float64)
    external_point = np.asarray(external_point, dtype=np.float64)
    internal_bootstrap = np.asarray(internal_bootstrap, dtype=np.float64)
    external_bootstrap = np.asarray(external_bootstrap, dtype=np.float64)
    expected_point_shape = (len(metric_names),)
    expected_bootstrap_shape = (replicates, len(metric_names))
    if internal_point.shape != expected_point_shape or external_point.shape != expected_point_shape:
        raise ValueError("point estimate shape mismatch")
    if (
        internal_bootstrap.shape != expected_bootstrap_shape
        or external_bootstrap.shape != expected_bootstrap_shape
    ):
        raise ValueError("bootstrap estimate shape mismatch")

    rows: list[dict[str, object]] = []
    difference_bootstrap = external_bootstrap - internal_bootstrap
    for position, metric in enumerate(metric_names):
        internal_low, internal_high = percentile_interval(
            internal_bootstrap[:, position]
        )
        external_low, external_high = percentile_interval(
            external_bootstrap[:, position]
        )
        difference_low, difference_high = percentile_interval(
            difference_bootstrap[:, position]
        )
        rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "source_model_version": SOURCE_MODEL_VERSION,
                "reference_seed": REFERENCE_SEED,
                "comparison_scope": scope,
                "sport_family": family,
                "horizon_seconds": horizon,
                "metric": metric,
                "unit": METRIC_UNITS[metric],
                "internal_source": "Endomondo",
                "internal_mode": "unseen_user_zero_history",
                "internal_estimate": float(internal_point[position]),
                "internal_ci_low": internal_low,
                "internal_ci_high": internal_high,
                "external_source": "GoldenCheetah",
                "external_mode": "forced_zero_history",
                "external_estimate": float(external_point[position]),
                "external_ci_low": external_low,
                "external_ci_high": external_high,
                "external_minus_internal": float(
                    external_point[position] - internal_point[position]
                ),
                "difference_ci_low": difference_low,
                "difference_ci_high": difference_high,
                "internal_users": internal_support["users"],
                "external_users": external_support["users"],
                "internal_sessions": internal_support["sessions"],
                "external_sessions": external_support["sessions"],
                "internal_origins": internal_support["origins"],
                "external_origins": external_support["origins"],
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "user independently within each source",
                "aggregation": (
                    "origin-within-session, session-within-user, equal-user "
                    "family mean, then stated family weighting"
                ),
                "post_cqr": True,
                "causal_source_or_device_effect_claimed": False,
                "user_level_conformal_guarantee_claimed": False,
            }
        )
    return rows


def load_prediction_subset(
    path: Path,
    expected_rows: np.ndarray,
    selected_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with np.load(path) as source:
        required = {"row_index", "zero_history_quantiles"}
        require(
            required.issubset(source.files),
            f"{path}: prediction archive missing {sorted(required - set(source.files))}",
        )
        row_index = np.asarray(source["row_index"], dtype=np.int64)
        require(np.array_equal(row_index, expected_rows), f"{path}: row mapping mismatch")
        quantiles = np.asarray(source["zero_history_quantiles"])
    require(
        quantiles.shape == (len(row_index), len(HORIZONS), 7),
        f"{path}: prediction shape mismatch {quantiles.shape}",
    )
    nonfinite = int((~np.isfinite(quantiles)).sum())
    crossings = int((np.diff(quantiles, axis=2) < -1e-6).sum())
    require(nonfinite == 0, f"{path}: non-finite predictions")
    require(crossings == 0, f"{path}: quantile crossings")
    require(selected_mask.shape == (len(row_index),), f"{path}: selection shape")
    selected_rows = row_index[selected_mask]
    selected_quantiles = quantiles[selected_mask]
    return selected_rows, selected_quantiles, {
        "archive_rows": int(len(row_index)),
        "selected_rows": int(len(selected_rows)),
        "nonfinite_values": nonfinite,
        "quantile_crossings": crossings,
    }


def support_by_family(
    rows: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, int]], np.ndarray, int]:
    users = np.asarray(arrays["users"][rows])
    sessions = np.asarray(arrays["sessions"][rows])
    sports = np.asarray(arrays["sport"][rows])
    session_sport = pd.DataFrame(
        {"session": sessions, "sport": sports}
    ).drop_duplicates()
    conflicts = int(session_sport["session"].duplicated().sum())
    require(conflicts == 0, "a session was assigned to multiple sport families")
    total_sessions = int(len(np.unique(sessions)))
    support: dict[str, dict[str, int]] = {}
    weights: list[float] = []
    for sport_code, family in SPORTS.items():
        selected = sports == sport_code
        family_support = {
            "users": int(len(np.unique(users[selected]))),
            "sessions": int(len(np.unique(sessions[selected]))),
            "origins": int(selected.sum()),
        }
        require(min(family_support.values()) > 1, f"insufficient support for {family}")
        support[family] = family_support
        weights.append(family_support["sessions"] / total_sessions)
    weights_array = np.asarray(weights, dtype=np.float64)
    require(
        np.isclose(weights_array.sum(), 1.0, rtol=0.0, atol=1e-12),
        "session weights do not sum to one",
    )
    return support, weights_array, conflicts


def analyze(args: argparse.Namespace) -> dict[str, object]:
    if args.seed != REFERENCE_SEED:
        raise ValueError(f"this frozen sensitivity requires seed {REFERENCE_SEED}")
    array_names = {
        "targets": "targets.npy",
        "dataset": "dataset_code.npy",
        "evaluation": "evaluation_origin.npy",
        "unseen": "unseen_user_partition.npy",
        "external": "primary_external_partition.npy",
        "sport": "sport_code.npy",
        "users": "user_index.npy",
        "sessions": "session_index.npy",
    }
    source_paths = (
        args.development_predictions,
        args.external_predictions,
        args.thresholds,
        args.development_audit,
        args.external_audit,
        args.freeze_record,
    )
    for path in source_paths:
        require(path.exists(), f"missing source file {path}")
    arrays = {
        name: np.load(args.array_dir / filename, mmap_mode="r")
        for name, filename in array_names.items()
    }
    row_counts = {name: int(len(values)) for name, values in arrays.items()}
    require(len(set(row_counts.values())) == 1, f"array mismatch: {row_counts}")

    development_audit = json.loads(
        args.development_audit.read_text(encoding="utf-8")
    )
    external_audit = json.loads(args.external_audit.read_text(encoding="utf-8"))
    freeze_record = json.loads(args.freeze_record.read_text(encoding="utf-8"))
    for label, payload in (
        ("development", development_audit),
        ("external", external_audit),
    ):
        require(bool(payload.get("all_assertions_pass")), f"{label} audit failed")
        require(int(payload.get("seed", -1)) == REFERENCE_SEED, f"{label} seed mismatch")
        require(
            str(payload.get("model_version")) == SOURCE_MODEL_VERSION,
            f"{label} model version mismatch",
        )
    require(
        development_audit.get("development_only") is True
        and development_audit.get("external_inference_performed") is False,
        "development audit does not document a development-only run",
    )
    require(
        external_audit.get("external_adaptation_or_recalibration") is False,
        "external audit does not exclude adaptation or recalibration",
    )
    require(
        freeze_record.get("status") == "frozen_before_external_inference",
        "freeze record is not finalized",
    )
    require(
        int(freeze_record.get("seed", -1)) == REFERENCE_SEED,
        "freeze-record seed mismatch",
    )

    external_prediction_hash = sha256_file(args.external_predictions)
    require(
        external_audit.get("outputs", {}).get("predictions_sha256")
        == external_prediction_hash,
        "external prediction hash does not match its audit",
    )
    freeze_hash = sha256_file(args.freeze_record)
    require(
        external_audit.get("freeze_record_sha256") == freeze_hash,
        "freeze-record hash does not match external audit",
    )
    threshold_hash_before = sha256_file(args.thresholds)
    require(
        external_audit.get("thresholds_sha256") == threshold_hash_before,
        "threshold hash does not match external audit",
    )
    threshold_payload, thresholds = load_thresholds(args.thresholds)

    calibration_rows = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["evaluation"] == 1)
        & (arrays["unseen"] == PARTITION_CALIBRATION)
    )
    calibration_users = np.unique(arrays["users"][calibration_rows])
    require(
        int(threshold_payload.get("calibration_rows", -1)) == len(calibration_rows),
        "CQR calibration-row count mismatch",
    )
    require(
        int(threshold_payload.get("calibration_users", -1)) == len(calibration_users),
        "CQR calibration-user count mismatch",
    )
    require(
        int((arrays["dataset"][calibration_rows] != 0).sum()) == 0,
        "non-Endomondo row entered threshold calibration",
    )

    expected_development_rows = np.flatnonzero(
        (arrays["dataset"] == 0) & (arrays["evaluation"] == 1)
    )
    development_test_mask = (
        (arrays["unseen"][expected_development_rows] == PARTITION_TEST)
        & np.isin(arrays["sport"][expected_development_rows], list(SPORTS))
    )
    internal_rows, internal_predictions, internal_alignment = load_prediction_subset(
        args.development_predictions,
        expected_development_rows,
        development_test_mask,
    )
    expected_external_rows = np.flatnonzero(
        (arrays["dataset"] == 1)
        & (arrays["external"] == EXTERNAL_FROZEN)
    )
    external_shared_mask = np.isin(
        arrays["sport"][expected_external_rows], list(SPORTS)
    )
    external_rows, external_predictions, external_alignment = load_prediction_subset(
        args.external_predictions,
        expected_external_rows,
        external_shared_mask,
    )

    source_rows = {"internal": internal_rows, "external": external_rows}
    source_predictions = {
        "internal": internal_predictions,
        "external": external_predictions,
    }
    expected_shared_support = {
        "internal": {"origins": 99_921, "sessions": 14_762, "users": 104},
        "external": {"origins": 531_725, "sessions": 31_851, "users": 144},
    }
    shared_support: dict[str, dict[str, int]] = {}
    family_support: dict[str, dict[str, dict[str, int]]] = {}
    session_weights: dict[str, np.ndarray] = {}
    session_sport_conflicts: dict[str, int] = {}
    for source, rows in source_rows.items():
        support = {
            "origins": int(len(rows)),
            "sessions": int(len(np.unique(arrays["sessions"][rows]))),
            "users": int(len(np.unique(arrays["users"][rows]))),
        }
        require(
            support == expected_shared_support[source],
            f"{source}: shared-family support drift: {support}",
        )
        shared_support[source] = support
        family_support[source], session_weights[source], conflicts = support_by_family(
            rows, arrays
        )
        session_sport_conflicts[source] = conflicts

    metric_names = list(METRIC_UNITS)
    point_cache: dict[str, dict[int, dict[str, np.ndarray]]] = {
        "internal": {},
        "external": {},
    }
    bootstrap_cache: dict[str, dict[int, dict[str, np.ndarray]]] = {
        "internal": {},
        "external": {},
    }
    bootstrap_draws: dict[str, np.ndarray] = {}
    for source, rows in source_rows.items():
        users = np.asarray(arrays["users"][rows])
        unique_users = np.unique(users)
        bootstrap_draws[source] = bootstrap_indices(
            len(unique_users),
            args.bootstrap_replicates,
            stable_seed(args.seed, f"{source}|independent-user-clusters"),
        )

    for source, rows in source_rows.items():
        users = np.asarray(arrays["users"][rows])
        sessions = np.asarray(arrays["sessions"][rows])
        sports = np.asarray(arrays["sport"][rows])
        unique_users = np.unique(users)
        indices = bootstrap_draws[source]
        for horizon_position, horizon in enumerate(HORIZONS):
            target = np.asarray(
                arrays["targets"][rows, horizon_position], dtype=np.float32
            )
            metrics = origin_uncertainty_metrics(
                source_predictions[source][:, horizon_position],
                target,
                thresholds,
                horizon_position,
            )
            require(list(metrics) == metric_names, "metric ordering changed")
            overall_user = hierarchical_user_metrics(
                metrics, users, sessions
            ).reindex(unique_users)
            require(not overall_user.isna().any().any(), f"{source}: missing user metric")
            family_users, family_cube = hierarchical_user_family_metrics(
                metrics, users, sessions, sports
            )
            require(
                np.array_equal(family_users, unique_users),
                f"{source}: family user alignment mismatch",
            )
            family_point = np.nanmean(family_cube, axis=0)
            family_bootstrap = bootstrap_family_nanmean(family_cube, indices)
            natural_point = overall_user.to_numpy(dtype=np.float64).mean(axis=0)
            natural_bootstrap = bootstrap_matrix_mean(
                overall_user.to_numpy(dtype=np.float64), indices
            )
            point_cache[source][horizon] = {
                "natural_shared": natural_point,
                "families": family_point,
                "equal_family": family_point.mean(axis=0),
                "endomondo_session_mix": weighted_family_metrics(
                    family_point, session_weights["internal"]
                ),
                "goldencheetah_session_mix": weighted_family_metrics(
                    family_point, session_weights["external"]
                ),
            }
            bootstrap_cache[source][horizon] = {
                "natural_shared": natural_bootstrap,
                "families": family_bootstrap,
                "equal_family": family_bootstrap.mean(axis=1),
                "endomondo_session_mix": weighted_family_metrics(
                    family_bootstrap, session_weights["internal"]
                ),
                "goldencheetah_session_mix": weighted_family_metrics(
                    family_bootstrap, session_weights["external"]
                ),
            }

    output_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        output_rows.extend(
            comparison_rows(
                scope="shared_family_natural_mix",
                family="all_three_supported_families",
                horizon=horizon,
                internal_point=point_cache["internal"][horizon]["natural_shared"],
                external_point=point_cache["external"][horizon]["natural_shared"],
                internal_bootstrap=bootstrap_cache["internal"][horizon]["natural_shared"],
                external_bootstrap=bootstrap_cache["external"][horizon]["natural_shared"],
                internal_support=shared_support["internal"],
                external_support=shared_support["external"],
                replicates=args.bootstrap_replicates,
            )
        )
        for family_position, family in enumerate(SPORTS.values()):
            output_rows.extend(
                comparison_rows(
                    scope="sport_matched",
                    family=family,
                    horizon=horizon,
                    internal_point=point_cache["internal"][horizon]["families"][family_position],
                    external_point=point_cache["external"][horizon]["families"][family_position],
                    internal_bootstrap=bootstrap_cache["internal"][horizon]["families"][:, family_position],
                    external_bootstrap=bootstrap_cache["external"][horizon]["families"][:, family_position],
                    internal_support=family_support["internal"][family],
                    external_support=family_support["external"][family],
                    replicates=args.bootstrap_replicates,
                )
            )
        for scope, family, cache_key in (
            (
                "equal_family_standardized",
                "macro_average_three_families",
                "equal_family",
            ),
            (
                "endomondo_session_mix_standardized",
                "three_families_weighted_to_endomondo_sessions",
                "endomondo_session_mix",
            ),
            (
                "goldencheetah_session_mix_standardized",
                "three_families_weighted_to_goldencheetah_sessions",
                "goldencheetah_session_mix",
            ),
        ):
            output_rows.extend(
                comparison_rows(
                    scope=scope,
                    family=family,
                    horizon=horizon,
                    internal_point=point_cache["internal"][horizon][cache_key],
                    external_point=point_cache["external"][horizon][cache_key],
                    internal_bootstrap=bootstrap_cache["internal"][horizon][cache_key],
                    external_bootstrap=bootstrap_cache["external"][horizon][cache_key],
                    internal_support=shared_support["internal"],
                    external_support=shared_support["external"],
                    replicates=args.bootstrap_replicates,
                )
            )

    require(len(output_rows) == 63, f"expected 63 rows, got {len(output_rows)}")
    output_frame = pd.DataFrame(output_rows)
    require(
        tuple(output_frame.columns) == PUBLIC_RESULT_COLUMNS,
        "public result schema changed",
    )
    forbidden_identifier_columns = {
        "user",
        "user_id",
        "user_index",
        "session",
        "session_id",
        "session_index",
        "row_index",
    }
    require(
        not forbidden_identifier_columns.intersection(output_frame.columns),
        "participant or session identifier entered public output",
    )
    require(
        not output_frame.duplicated(
            ["comparison_scope", "sport_family", "horizon_seconds", "metric"]
        ).any(),
        "duplicate analysis keys",
    )
    numeric_columns = [
        "internal_estimate",
        "internal_ci_low",
        "internal_ci_high",
        "external_estimate",
        "external_ci_low",
        "external_ci_high",
        "external_minus_internal",
        "difference_ci_low",
        "difference_ci_high",
    ]
    require(
        np.isfinite(output_frame[numeric_columns].to_numpy(dtype=np.float64)).all(),
        "non-finite result value",
    )
    for prefix in ("internal", "external"):
        require(
            (
                output_frame[f"{prefix}_ci_low"]
                <= output_frame[f"{prefix}_estimate"]
            ).all()
            and (
                output_frame[f"{prefix}_estimate"]
                <= output_frame[f"{prefix}_ci_high"]
            ).all(),
            f"{prefix} point estimate outside its interval",
        )
    require(
        (output_frame["difference_ci_low"] <= output_frame["difference_ci_high"]).all(),
        "reversed difference interval",
    )
    picp = output_frame[output_frame["metric"] == "picp_90"]
    require(
        picp["internal_estimate"].between(0.0, 1.0).all()
        and picp["external_estimate"].between(0.0, 1.0).all(),
        "PICP outside [0, 1]",
    )
    positive = output_frame[output_frame["metric"] != "picp_90"]
    require(
        (positive["internal_estimate"] > 0).all()
        and (positive["external_estimate"] > 0).all(),
        "nonpositive width or WIS",
    )

    atomic_csv(args.output_csv, output_rows)
    threshold_hash_after = sha256_file(args.thresholds)
    require(
        threshold_hash_after == threshold_hash_before,
        "frozen Endomondo thresholds changed during analysis",
    )
    source_files = {
        "development_predictions": {
            "path": str(args.development_predictions),
            "sha256": sha256_file(args.development_predictions),
        },
        "external_predictions": {
            "path": str(args.external_predictions),
            "sha256": external_prediction_hash,
        },
        "thresholds": {
            "path": str(args.thresholds),
            "sha256_before": threshold_hash_before,
            "sha256_after": threshold_hash_after,
            "unchanged": threshold_hash_before == threshold_hash_after,
        },
        "development_audit": {
            "path": str(args.development_audit),
            "sha256": sha256_file(args.development_audit),
        },
        "external_audit": {
            "path": str(args.external_audit),
            "sha256": sha256_file(args.external_audit),
        },
        "freeze_record": {
            "path": str(args.freeze_record),
            "sha256": freeze_hash,
        },
        "analysis_script": {
            "path": str(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "source_model_version": SOURCE_MODEL_VERSION,
        "reference_seed": REFERENCE_SEED,
        "intended_use": (
            "Post-hoc uncertainty sensitivity comparing frozen zero-history "
            "Endomondo unseen-user and GoldenCheetah source-transport performance "
            "within shared sports and under fixed sport-family composition"
        ),
        "training_or_model_selection_performed": False,
        "external_adaptation_or_recalibration_performed": False,
        "source_files": source_files,
        "array_row_counts": row_counts,
        "alignment": {
            "development": internal_alignment,
            "external": external_alignment,
            "mapping_exact": True,
            "session_sport_conflicts": session_sport_conflicts,
        },
        "frozen_calibration": {
            "source": "Endomondo",
            "partition": "unseen-user calibration",
            "mode": "zero_history",
            "calibration_origins": int(len(calibration_rows)),
            "calibration_users": int(len(calibration_users)),
            "thresholds_bpm": thresholds,
            "threshold_file_unchanged": True,
            "goldencheetah_recalibration_performed": False,
        },
        "shared_three_family_support": shared_support,
        "family_support": family_support,
        "standardization": {
            "shared_sport_families": list(SPORTS.values()),
            "equal_family_weights": {
                family: 1.0 / len(SPORTS) for family in SPORTS.values()
            },
            "session_mix_weights": {
                source: {
                    family: float(weights[position])
                    for position, family in enumerate(SPORTS.values())
                }
                for source, weights in session_weights.items()
            },
            "estimands": [
                "shared_family_natural_mix",
                "sport_matched",
                "equal_family_standardized",
                "endomondo_session_mix_standardized",
                "goldencheetah_session_mix_standardized",
            ],
        },
        "metrics": {
            "picp_90": "post-CQR central 90% empirical coverage",
            "mean_90_interval_width_bpm": "post-CQR central 90% interval width",
            "weighted_interval_score": (
                "post-CQR central 50%, 80%, and 90% intervals plus median; "
                "calculated by bootstrap_external_sport_uncertainty.py using "
                "the same formula as evaluate_probabilistic_metrics.py"
            ),
            "aggregation": (
                "origin-within-session, session-within-user, equal-user family "
                "mean, followed by the stated family weights"
            ),
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "base_seed": args.seed,
            "unit": "user independently within each source",
            "interval": "two-sided 95% percentile",
            "difference": (
                "independent source-specific bootstrap draws paired only by "
                "replicate index to form external-minus-internal draws"
            ),
            "same_source_draws_reused_across_families_metrics_and_horizons": True,
        },
        "interpretation_guards": {
            "causal_source_or_device_effect_claimed": False,
            "finite_sample_user_level_conformal_guarantee_claimed": False,
            "source_specific_user_clusters_treated_as_independent": True,
        },
        "privacy": {
            "participant_or_session_identifiers_written": False,
            "result_contains_aggregate_counts_only": True,
            "public_result_columns": list(PUBLIC_RESULT_COLUMNS),
        },
        "output": {
            "path": str(args.output_csv),
            "sha256": sha256_file(args.output_csv),
            "rows": len(output_rows),
        },
        "limitations": [
            (
                "Standardization controls only the observed three-family sport "
                "composition; users, devices, session structure, sampling, and "
                "unmeasured exercise characteristics can still differ."
            ),
            (
                "The internal indoor/virtual-cycling family has limited user "
                "support, so family-specific and standardized intervals may be wide."
            ),
            (
                "The analysis treats source-specific user clusters as independent, "
                "although cross-platform participant overlap cannot be verified."
            ),
            (
                "Results are conditional on the single frozen reference-seed model "
                "and do not include model-training variability."
            ),
            (
                "The pooled-origin CQR calibration does not provide a finite-sample "
                "user-level or sport-conditional coverage guarantee."
            ),
            (
                "Differences are descriptive and do not identify a causal platform "
                "or wearable-device effect."
            ),
        ],
        "all_assertions_pass": True,
    }
    atomic_json(args.audit_json, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc sport-composition-standardized post-CQR uncertainty "
            "sensitivity for the frozen 20260722 reference seed."
        )
    )
    parser.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    parser.add_argument(
        "--development-predictions",
        type=Path,
        default=Path(
            "outputs/q1_multiseed_v0_21_0/seed_20260722/unseen_main/"
            "development_predictions.npz"
        ),
    )
    parser.add_argument(
        "--external-predictions",
        type=Path,
        default=Path(
            "outputs/q1_multiseed_v0_21_0/seed_20260722/unseen_main/"
            "external_predictions.npz"
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(
            "outputs/q1_multiseed_v0_21_0/seed_20260722/unseen_main/model/"
            "conformal_thresholds_v0_11_0.json"
        ),
    )
    parser.add_argument(
        "--development-audit",
        type=Path,
        default=Path(
            "outputs/q1_multiseed_v0_21_0/seed_20260722/unseen_main/"
            "development_audit.json"
        ),
    )
    parser.add_argument(
        "--external-audit",
        type=Path,
        default=Path(
            "outputs/q1_multiseed_v0_21_0/seed_20260722/unseen_main/"
            "external_audit.json"
        ),
    )
    parser.add_argument(
        "--freeze-record",
        type=Path,
        default=Path(
            "outputs/q1_multiseed_v0_21_0/seed_20260722/unseen_main/"
            "freeze_record.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/results/external_sport_uncertainty_standardization_v0_24_0.csv"
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/audit/external_sport_uncertainty_standardization_v0_24_0.json"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=REFERENCE_SEED)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("at least 1,000 bootstrap replicates are required")
    audit = analyze(args)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
