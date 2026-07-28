from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.20.1"
SOURCE_MODEL_VERSION = "0.11.0"
HORIZONS = (60, 180, 300)
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1
SPORTS = {
    1: "outdoor_cycling",
    2: "indoor_virtual_cycling",
    3: "running",
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


def hierarchical_user_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "absolute_error": np.abs(prediction - target),
            "user": users,
            "session": sessions,
        }
    )
    session = frame.groupby(["user", "session"], sort=False)[
        "absolute_error"
    ].mean()
    return session.groupby(level="user", sort=False).mean()


def user_family_mae_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
    sports: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "absolute_error": np.abs(prediction - target),
            "user": users,
            "session": sessions,
            "sport": sports,
        }
    )
    session = frame.groupby(["user", "sport", "session"], sort=False)[
        "absolute_error"
    ].mean()
    user_family = session.groupby(level=["user", "sport"], sort=False).mean()
    matrix = user_family.unstack("sport").reindex(columns=list(SPORTS))
    return matrix.index.to_numpy(), matrix.to_numpy(dtype=np.float64)


def bootstrap_indices(n_users: int, replicates: int, seed: int) -> np.ndarray:
    if n_users < 2:
        raise ValueError("at least two users are required")
    generator = np.random.default_rng(seed)
    return generator.integers(0, n_users, size=(replicates, n_users))


def bootstrap_nanmean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    sampled = np.asarray(values, dtype=np.float64)[indices]
    with np.errstate(invalid="ignore"):
        result = np.nanmean(sampled, axis=1)
    if not np.isfinite(result).all():
        raise AssertionError("bootstrap replicate lost all supported users")
    return result


def bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)[indices].mean(axis=1)
    if not np.isfinite(result).all():
        raise AssertionError("non-finite bootstrap mean")
    return result


def weighted_family_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(SPORTS),):
        raise ValueError("family weights have the wrong shape")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("family weights must be finite and nonnegative")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("family weights must sum to one")
    if values.shape[-1] != len(SPORTS):
        raise ValueError("family values have the wrong final dimension")
    result = values @ weights
    if not np.isfinite(result).all():
        raise AssertionError("non-finite weighted family average")
    return result


def comparison_row(
    *,
    scope: str,
    family: str,
    horizon: int,
    internal_point: float,
    external_point: float,
    internal_bootstrap: np.ndarray,
    external_bootstrap: np.ndarray,
    internal_users: int,
    external_users: int,
    internal_sessions: int,
    external_sessions: int,
    internal_origins: int,
    external_origins: int,
    replicates: int,
) -> dict[str, object]:
    internal_low, internal_high = percentile_interval(internal_bootstrap)
    external_low, external_high = percentile_interval(external_bootstrap)
    difference_bootstrap = external_bootstrap - internal_bootstrap
    difference_low, difference_high = percentile_interval(difference_bootstrap)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "source_model_version": SOURCE_MODEL_VERSION,
        "comparison_scope": scope,
        "sport_family": family,
        "horizon_seconds": horizon,
        "internal_mae_bpm": internal_point,
        "internal_mae_ci_low_bpm": internal_low,
        "internal_mae_ci_high_bpm": internal_high,
        "external_mae_bpm": external_point,
        "external_mae_ci_low_bpm": external_low,
        "external_mae_ci_high_bpm": external_high,
        "external_minus_internal_bpm": external_point - internal_point,
        "difference_ci_low_bpm": difference_low,
        "difference_ci_high_bpm": difference_high,
        "internal_users": internal_users,
        "external_users": external_users,
        "internal_sessions": internal_sessions,
        "external_sessions": external_sessions,
        "internal_origins": internal_origins,
        "external_origins": external_origins,
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "user within each data source",
        "aggregation": (
            "origin-within-session, session-within-user, equal-user mean"
        ),
        "causal_source_or_device_effect_claimed": False,
    }


def analyze(args: argparse.Namespace) -> dict[str, object]:
    arrays = {
        "targets": np.load(args.array_dir / "targets.npy", mmap_mode="r"),
        "dataset": np.load(args.array_dir / "dataset_code.npy", mmap_mode="r"),
        "unseen": np.load(
            args.array_dir / "unseen_user_partition.npy", mmap_mode="r"
        ),
        "external": np.load(
            args.array_dir / "primary_external_partition.npy", mmap_mode="r"
        ),
        "sport": np.load(args.array_dir / "sport_code.npy", mmap_mode="r"),
        "users": np.load(args.array_dir / "user_index.npy", mmap_mode="r"),
        "sessions": np.load(args.array_dir / "session_index.npy", mmap_mode="r"),
        "evaluation": np.load(
            args.array_dir / "evaluation_origin.npy", mmap_mode="r"
        ),
    }
    row_counts = {name: int(len(values)) for name, values in arrays.items()}
    require(len(set(row_counts.values())) == 1, f"array mismatch: {row_counts}")

    with np.load(args.predictions) as source:
        require(
            {"row_index", "zero_history_quantiles"}.issubset(source.files),
            "prediction archive missing fields",
        )
        row_index = np.asarray(source["row_index"], dtype=np.int64)
        quantiles = np.asarray(source["zero_history_quantiles"])
    expected_rows = np.flatnonzero(
        ((arrays["dataset"] == 0) & (arrays["evaluation"] == 1))
        | (
            (arrays["dataset"] == 1)
            & (arrays["external"] == EXTERNAL_FROZEN)
        )
    )
    require(np.array_equal(row_index, expected_rows), "prediction row mapping mismatch")
    require(
        quantiles.shape == (len(row_index), 3, 7), "prediction shape mismatch"
    )
    require(np.isfinite(quantiles).all(), "non-finite prediction")
    median = quantiles[:, :, 3]
    local = {
        name: np.asarray(values[row_index])
        for name, values in arrays.items()
        if name != "evaluation"
    }
    reported_masks = {
        "internal": (local["dataset"] == 0)
        & (local["unseen"] == PARTITION_TEST),
        "external": (local["dataset"] == 1)
        & (local["external"] == EXTERNAL_FROZEN),
    }
    reported_support = {
        "internal": (101_184, 15_026, 105),
        "external": (531_725, 31_851, 144),
    }
    for name, (origins, sessions, users) in reported_support.items():
        mask = reported_masks[name]
        require(int(mask.sum()) == origins, f"{name}: origin count")
        require(
            len(np.unique(local["sessions"][mask])) == sessions,
            f"{name}: session count",
        )
        require(
            len(np.unique(local["users"][mask])) == users,
            f"{name}: user count",
        )

    shared_sport = np.isin(local["sport"], np.asarray(list(SPORTS)))
    masks = {
        name: mask & shared_sport for name, mask in reported_masks.items()
    }
    shared_support = {
        "internal": (99_921, 14_762, 104),
        "external": (531_725, 31_851, 144),
    }
    for name, (origins, sessions, users) in shared_support.items():
        mask = masks[name]
        require(int(mask.sum()) == origins, f"{name}: shared origin count")
        require(
            len(np.unique(local["sessions"][mask])) == sessions,
            f"{name}: shared session count",
        )
        require(
            len(np.unique(local["users"][mask])) == users,
            f"{name}: shared user count",
        )

    support_rows: list[dict[str, object]] = []
    for data_source, mask in masks.items():
        total_sessions = len(np.unique(local["sessions"][mask]))
        total_origins = int(mask.sum())
        for sport_code, family in SPORTS.items():
            selected = mask & (local["sport"] == sport_code)
            n_sessions = len(np.unique(local["sessions"][selected]))
            n_origins = int(selected.sum())
            support_rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "data_source": data_source,
                    "sport_code": sport_code,
                    "sport_family": family,
                    "users": int(len(np.unique(local["users"][selected]))),
                    "sessions": int(n_sessions),
                    "session_share": n_sessions / total_sessions,
                    "origins": n_origins,
                    "origin_share": n_origins / total_origins,
                }
            )

    support_frame_for_weights = pd.DataFrame(support_rows)
    fixed_session_weights = {
        data_source: support_frame_for_weights[
            support_frame_for_weights["data_source"] == data_source
        ]
        .set_index("sport_code")
        .reindex(list(SPORTS))["session_share"]
        .to_numpy(dtype=np.float64)
        for data_source in masks
    }
    for data_source, weights in fixed_session_weights.items():
        require(
            np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12),
            f"{data_source}: fixed session weights",
        )

    result_rows: list[dict[str, object]] = []
    point_cache: dict[str, dict[int, dict[str, object]]] = {
        "internal": {},
        "external": {},
    }
    bootstrap_cache: dict[str, dict[int, dict[str, np.ndarray]]] = {
        "internal": {},
        "external": {},
    }
    reported_point_cache: dict[str, dict[int, float]] = {
        "internal": {},
        "external": {},
    }
    reported_bootstrap_cache: dict[str, dict[int, np.ndarray]] = {
        "internal": {},
        "external": {},
    }

    for data_source, mask in masks.items():
        selected_users = local["users"][mask]
        selected_sessions = local["sessions"][mask]
        selected_sports = local["sport"][mask]
        unique_users = np.unique(selected_users)
        indices = bootstrap_indices(
            len(unique_users),
            args.bootstrap_replicates,
            stable_seed(args.seed, f"{data_source}|users"),
        )
        for horizon_position, horizon in enumerate(HORIZONS):
            selected_target = local["targets"][mask, horizon_position]
            selected_prediction = median[mask, horizon_position]
            overall_user = hierarchical_user_mae(
                selected_prediction,
                selected_target,
                selected_users,
                selected_sessions,
            ).reindex(unique_users)
            require(not overall_user.isna().any(), f"{data_source}: missing user MAE")
            family_user_ids, family_matrix = user_family_mae_matrix(
                selected_prediction,
                selected_target,
                selected_users,
                selected_sessions,
                selected_sports,
            )
            require(
                np.array_equal(family_user_ids, unique_users),
                f"{data_source}: family user alignment",
            )
            family_point = np.nanmean(family_matrix, axis=0)
            family_bootstrap = np.column_stack(
                [
                    bootstrap_nanmean(family_matrix[:, position], indices)
                    for position in range(len(SPORTS))
                ]
            )
            point_cache[data_source][horizon] = {
                "overall": float(overall_user.mean()),
                "families": family_point,
                "macro_equal": float(family_point.mean()),
                "fixed_internal_session_mix": float(
                    weighted_family_average(
                        family_point, fixed_session_weights["internal"]
                    )
                ),
                "fixed_external_session_mix": float(
                    weighted_family_average(
                        family_point, fixed_session_weights["external"]
                    )
                ),
            }
            bootstrap_cache[data_source][horizon] = {
                "overall": bootstrap_mean(
                    overall_user.to_numpy(dtype=np.float64), indices
                ),
                "families": family_bootstrap,
                "macro_equal": family_bootstrap.mean(axis=1),
                "fixed_internal_session_mix": weighted_family_average(
                    family_bootstrap, fixed_session_weights["internal"]
                ),
                "fixed_external_session_mix": weighted_family_average(
                    family_bootstrap, fixed_session_weights["external"]
                ),
            }

    for data_source, mask in reported_masks.items():
        selected_users = local["users"][mask]
        selected_sessions = local["sessions"][mask]
        unique_users = np.unique(selected_users)
        indices = bootstrap_indices(
            len(unique_users),
            args.bootstrap_replicates,
            stable_seed(args.seed, f"{data_source}|reported-users"),
        )
        for horizon_position, horizon in enumerate(HORIZONS):
            user_mae = hierarchical_user_mae(
                median[mask, horizon_position],
                local["targets"][mask, horizon_position],
                selected_users,
                selected_sessions,
            ).reindex(unique_users)
            require(
                not user_mae.isna().any(),
                f"{data_source}: missing reported user MAE",
            )
            reported_point_cache[data_source][horizon] = float(user_mae.mean())
            reported_bootstrap_cache[data_source][horizon] = bootstrap_mean(
                user_mae.to_numpy(dtype=np.float64), indices
            )

    reference = pd.read_csv(args.point_reference)
    maximum_reference_delta = 0.0
    for horizon in HORIZONS:
        for data_source, regime in (
            ("internal", "unseen_user_test"),
            ("external", "goldencheetah_frozen_external"),
        ):
            selected = reference[
                (reference["regime"] == regime)
                & (reference["mode"] == "zero_history")
                & (reference["horizon_seconds"] == horizon)
            ]
            require(len(selected) == 1, f"{regime}/{horizon}: reference row")
            maximum_reference_delta = max(
                maximum_reference_delta,
                abs(
                    reported_point_cache[data_source][horizon]
                    - float(selected.iloc[0]["mae_bpm"])
                ),
            )

        internal_mask = masks["internal"]
        external_mask = masks["external"]
        result_rows.append(
            comparison_row(
                scope="reported_natural_mix",
                family="all_internal_test_vs_external_shared_three",
                horizon=horizon,
                internal_point=reported_point_cache["internal"][horizon],
                external_point=reported_point_cache["external"][horizon],
                internal_bootstrap=reported_bootstrap_cache["internal"][horizon],
                external_bootstrap=reported_bootstrap_cache["external"][horizon],
                internal_users=105,
                external_users=144,
                internal_sessions=15_026,
                external_sessions=31_851,
                internal_origins=101_184,
                external_origins=531_725,
                replicates=args.bootstrap_replicates,
            )
        )
        result_rows.append(
            comparison_row(
                scope="shared_family_natural_mix",
                family="all_three_supported_families",
                horizon=horizon,
                internal_point=float(point_cache["internal"][horizon]["overall"]),
                external_point=float(point_cache["external"][horizon]["overall"]),
                internal_bootstrap=bootstrap_cache["internal"][horizon]["overall"],
                external_bootstrap=bootstrap_cache["external"][horizon]["overall"],
                internal_users=104,
                external_users=144,
                internal_sessions=14_762,
                external_sessions=31_851,
                internal_origins=99_921,
                external_origins=531_725,
                replicates=args.bootstrap_replicates,
            )
        )

        for family_position, (sport_code, family) in enumerate(SPORTS.items()):
            internal_selected = internal_mask & (local["sport"] == sport_code)
            external_selected = external_mask & (local["sport"] == sport_code)
            result_rows.append(
                comparison_row(
                    scope="sport_matched",
                    family=family,
                    horizon=horizon,
                    internal_point=float(
                        point_cache["internal"][horizon]["families"][family_position]
                    ),
                    external_point=float(
                        point_cache["external"][horizon]["families"][family_position]
                    ),
                    internal_bootstrap=bootstrap_cache["internal"][horizon][
                        "families"
                    ][:, family_position],
                    external_bootstrap=bootstrap_cache["external"][horizon][
                        "families"
                    ][:, family_position],
                    internal_users=int(
                        len(np.unique(local["users"][internal_selected]))
                    ),
                    external_users=int(
                        len(np.unique(local["users"][external_selected]))
                    ),
                    internal_sessions=int(
                        len(np.unique(local["sessions"][internal_selected]))
                    ),
                    external_sessions=int(
                        len(np.unique(local["sessions"][external_selected]))
                    ),
                    internal_origins=int(internal_selected.sum()),
                    external_origins=int(external_selected.sum()),
                    replicates=args.bootstrap_replicates,
                )
            )

        result_rows.append(
            comparison_row(
                scope="equal_family_standardized",
                family="macro_average_three_families",
                horizon=horizon,
                internal_point=float(
                    point_cache["internal"][horizon]["macro_equal"]
                ),
                external_point=float(
                    point_cache["external"][horizon]["macro_equal"]
                ),
                internal_bootstrap=bootstrap_cache["internal"][horizon][
                    "macro_equal"
                ],
                external_bootstrap=bootstrap_cache["external"][horizon][
                    "macro_equal"
                ],
                internal_users=104,
                external_users=144,
                internal_sessions=14_762,
                external_sessions=31_851,
                internal_origins=99_921,
                external_origins=531_725,
                replicates=args.bootstrap_replicates,
            )
        )

        for reference_source, label in (
            ("internal", "endomondo_session_mix_standardized"),
            ("external", "goldencheetah_session_mix_standardized"),
        ):
            cache_key = f"fixed_{reference_source}_session_mix"
            result_rows.append(
                comparison_row(
                    scope=label,
                    family=f"three_families_weighted_to_{reference_source}_sessions",
                    horizon=horizon,
                    internal_point=float(
                        point_cache["internal"][horizon][cache_key]
                    ),
                    external_point=float(
                        point_cache["external"][horizon][cache_key]
                    ),
                    internal_bootstrap=bootstrap_cache["internal"][horizon][
                        cache_key
                    ],
                    external_bootstrap=bootstrap_cache["external"][horizon][
                        cache_key
                    ],
                    internal_users=104,
                    external_users=144,
                    internal_sessions=14_762,
                    external_sessions=31_851,
                    internal_origins=99_921,
                    external_origins=531_725,
                    replicates=args.bootstrap_replicates,
                )
            )

    require(len(result_rows) == 24, f"expected 24 result rows, got {len(result_rows)}")
    require(len(support_rows) == 6, f"expected 6 support rows, got {len(support_rows)}")
    tolerance = 5e-6
    require(
        maximum_reference_delta <= tolerance,
        f"overall reference mismatch: {maximum_reference_delta}",
    )
    result_frame = pd.DataFrame(result_rows)
    numeric = result_frame[
        [
            "internal_mae_bpm",
            "external_mae_bpm",
            "external_minus_internal_bpm",
            "difference_ci_low_bpm",
            "difference_ci_high_bpm",
        ]
    ].to_numpy()
    require(np.isfinite(numeric).all(), "non-finite result output")
    require(
        not result_frame.duplicated(
            ["comparison_scope", "sport_family", "horizon_seconds"]
        ).any(),
        "duplicate result keys",
    )
    support_frame = pd.DataFrame(support_rows)
    for data_source in masks:
        source_support = support_frame[support_frame["data_source"] == data_source]
        require(
            abs(float(source_support["session_share"].sum()) - 1.0) < 1e-12,
            f"{data_source}: session shares",
        )
        require(
            abs(float(source_support["origin_share"].sum()) - 1.0) < 1e-12,
            f"{data_source}: origin shares",
        )

    atomic_csv(args.output_results, result_rows)
    atomic_csv(args.output_support, support_rows)
    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "source_model_version": SOURCE_MODEL_VERSION,
        "intended_use": (
            "Post hoc sport-matched and fixed-composition-standardized description of "
            "the frozen zero-history Endomondo-to-GoldenCheetah performance gap"
        ),
        "source_files": {
            "predictions": str(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
            "point_reference": str(args.point_reference),
            "point_reference_sha256": sha256_file(args.point_reference),
        },
        "row_counts": row_counts,
        "reported_support": reported_support,
        "shared_three_family_support": shared_support,
        "standardization": {
            "shared_sport_families": list(SPORTS.values()),
            "macro_weights": {family: 1.0 / 3.0 for family in SPORTS.values()},
            "fixed_session_weights": {
                source: {
                    family: float(weights[position])
                    for position, family in enumerate(SPORTS.values())
                }
                for source, weights in fixed_session_weights.items()
            },
            "interpretation": (
                "Both data sources are summarized under equal-family, fixed "
                "Endomondo-session, and fixed GoldenCheetah-session weights. These "
                "estimands remove unequal family-mix weighting but do not identify "
                "a causal platform or device effect."
            ),
        },
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "base_seed": args.seed,
            "unit": "user within source",
            "difference": (
                "external and internal source-specific user bootstraps paired by "
                "replicate index"
            ),
        },
        "maximum_absolute_reference_delta": maximum_reference_delta,
        "reference_tolerance": tolerance,
        "outputs": {
            "results": str(args.output_results),
            "results_sha256": sha256_file(args.output_results),
            "result_rows": len(result_rows),
            "support": str(args.output_support),
            "support_sha256": sha256_file(args.output_support),
            "support_rows": len(support_rows),
        },
        "limitations": [
            (
                "The standardization controls only the observed broad sport-family "
                "mix. Users, devices, session structure, sampling, preprocessing, "
                "and unmeasured exercise characteristics can still differ."
            ),
            (
                "The indoor/virtual cycling internal comparison has only 18 users, "
                "so its family-specific and macro-standardized contribution remains "
                "uncertain despite user bootstrap intervals."
            ),
            (
                "Participant independence across sources cannot be verified. The "
                "bootstrap treats source-specific user clusters as separate and is "
                "descriptive rather than causal."
            ),
            (
                "All comparisons are conditional on the existing single-seed frozen "
                "checkpoint and do not include training variability."
            ),
        ],
        "all_assertions_pass": True,
    }
    atomic_json(args.audit_json, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen zero-history performance within shared sports and after "
            "equal-family and fixed-session-mix standardization."
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
        "--point-reference",
        type=Path,
        default=Path("outputs/results/uncertainty_point_metrics_v0_11_0.csv"),
    )
    parser.add_argument(
        "--output-results",
        type=Path,
        default=Path(
            "outputs/results/external_sport_standardization_v0_20_1.csv"
        ),
    )
    parser.add_argument(
        "--output-support",
        type=Path,
        default=Path("outputs/results/external_sport_composition_v0_20_1.csv"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/audit/external_sport_standardization_v0_20_1.json"),
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
