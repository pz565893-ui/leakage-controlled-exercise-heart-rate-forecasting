from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

from build_deliberately_leaky_temporal_partition_v0_28_0 import (
    ANALYSIS_VERSION,
    index_sha256,
    require,
)
from run_q1_multiseed_queue import atomic_json, sha256_file, utc_now
from train_deliberately_leaky_temporal_negative_control_v0_28_0 import (
    PROTOCOL,
    TRAINING_HISTORY_MODE,
    validate_configuration,
)
from train_uncertainty_model import INTERVALS
from train_xgboost_baseline import HORIZONS


EXPECTED_SEEDS = (20260722, 20260723, 20260724)


class PredictionArchive(NamedTuple):
    row_index: np.ndarray
    quantiles: np.ndarray


def load_prediction_archive(path: Path) -> PredictionArchive:
    require(path.is_file(), f"missing predictions: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(
            set(archive.files) == {"row_index", "zero_history_quantiles"},
            f"prediction schema mismatch: {path}",
        )
        row_index = np.asarray(archive["row_index"], dtype=np.int64)
        quantiles = np.asarray(archive["zero_history_quantiles"], dtype=np.float32)
    require(row_index.ndim == 1 and len(row_index) > 0, "invalid row index")
    require(len(np.unique(row_index)) == len(row_index), "duplicate prediction rows")
    require(quantiles.shape == (len(row_index), 3, 7), "quantile shape mismatch")
    require(np.isfinite(quantiles).all(), "non-finite prediction")
    require(not np.any(np.diff(quantiles, axis=2) < -1e-6), "crossed quantiles")
    return PredictionArchive(row_index, quantiles)


def hierarchical_user_point_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    sessions = np.asarray(sessions, dtype=np.int64)
    require(
        prediction.shape == target.shape == users.shape == sessions.shape,
        "point support shape mismatch",
    )
    require(len(prediction) > 0 and np.isfinite(prediction).all(), "point values")
    error = prediction - target
    origin_metrics = np.column_stack((np.abs(error), error * error, error))
    order = np.lexsort((sessions, users))
    sorted_users = users[order]
    sorted_sessions = sessions[order]
    sorted_values = origin_metrics[order]
    session_start = np.r_[
        0,
        np.flatnonzero(
            (np.diff(sorted_users) != 0) | (np.diff(sorted_sessions) != 0)
        )
        + 1,
    ]
    session_count = np.diff(np.r_[session_start, len(sorted_users)])
    session_values = np.add.reduceat(sorted_values, session_start, axis=0)
    session_values = session_values / session_count[:, None]
    session_values[:, 1] = np.sqrt(session_values[:, 1])
    session_users = sorted_users[session_start]
    user_start = np.r_[0, np.flatnonzero(np.diff(session_users) != 0) + 1]
    user_count = np.diff(np.r_[user_start, len(session_users)])
    user_values = np.add.reduceat(session_values, user_start, axis=0)
    user_values = user_values / user_count[:, None]
    user_ids = session_users[user_start]
    require(len(np.unique(user_ids)) == len(user_ids), "duplicate user aggregates")
    return user_ids, {
        "mae_bpm": user_values[:, 0],
        "rmse_bpm": user_values[:, 1],
        "bias_bpm": user_values[:, 2],
    }


def hierarchical_user_average(
    values: np.ndarray, users: np.ndarray, sessions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    sessions = np.asarray(sessions, dtype=np.int64)
    require(values.shape == users.shape == sessions.shape, "average support shape")
    require(len(values) > 0 and np.isfinite(values).all(), "average values")
    order = np.lexsort((sessions, users))
    sorted_users = users[order]
    sorted_sessions = sessions[order]
    sorted_values = values[order]
    session_start = np.r_[
        0,
        np.flatnonzero(
            (np.diff(sorted_users) != 0) | (np.diff(sorted_sessions) != 0)
        )
        + 1,
    ]
    session_count = np.diff(np.r_[session_start, len(sorted_users)])
    session_means = np.add.reduceat(sorted_values, session_start) / session_count
    session_users = sorted_users[session_start]
    user_start = np.r_[0, np.flatnonzero(np.diff(session_users) != 0) + 1]
    user_count = np.diff(np.r_[user_start, len(session_users)])
    user_means = np.add.reduceat(session_means, user_start) / user_count
    return session_users[user_start], user_means


def percentile_user_bootstrap(
    values: np.ndarray, *, replicates: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) >= 2, "bootstrap values")
    require(np.isfinite(values).all(), "non-finite bootstrap values")
    require(replicates >= 1_000, "too few bootstrap replicates")
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1_000):
        end = min(start + 1_000, replicates)
        selection = generator.integers(
            0, len(values), size=(end - start, len(values))
        )
        estimates[start:end] = values[selection].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def stable_bootstrap_seed(base_seed: int, components: tuple[object, ...]) -> int:
    label = "|".join(str(component) for component in components)
    offset = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")
    return int((base_seed + offset) % (2**32))


def load_thresholds(path: Path, *, leaky: bool) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if leaky:
        require(payload.get("valid_for_generalization") is False, "leaky threshold validity")
        require(payload.get("coverage_guarantee_valid") is False, "leaky coverage flag")
        thresholds = payload.get("thresholds")
    else:
        thresholds = payload.get("zero_history")
    require(isinstance(thresholds, dict), f"threshold schema: {path}")
    require(set(thresholds) == {"0.5", "0.8", "0.9"}, "threshold coverage set")
    for values in thresholds.values():
        require(
            isinstance(values, list)
            and len(values) == 3
            and all(float(value) >= 0 and np.isfinite(float(value)) for value in values),
            "invalid conformal threshold",
        )
    return {key: [float(value) for value in values] for key, values in thresholds.items()}


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"no rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "per_seed": output_dir / "paired_metrics_per_seed_v0_28_0.csv",
        "seed_summary": output_dir / "paired_metrics_seed_summary_v0_28_0.csv",
        "user_seed_mean": output_dir / "paired_user_seed_mean_v0_28_0.csv",
        "user_bootstrap": output_dir / "paired_user_bootstrap_v0_28_0.csv",
        "interval": output_dir / "interval_diagnostics_per_seed_v0_28_0.csv",
        "audit": output_dir / "audit.json",
    }


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    require(
        args.acknowledge_invalid_generalization,
        "aggregation requires --acknowledge-invalid-generalization",
    )
    config = json.loads(args.configuration.read_text(encoding="utf-8"))
    validate_configuration(config)
    require(tuple(config["seeds"]) == EXPECTED_SEEDS, "seed lock")
    partition_audit = json.loads(args.partition_audit.read_text(encoding="utf-8"))
    require(partition_audit.get("all_assertions_pass") is True, "partition audit")
    require(partition_audit.get("valid_for_generalization") is False, "partition validity")

    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    users_all = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions_all = np.load(args.array_dir / "session_index.npy", mmap_mode="r")
    expected_hash = str(
        config["fixed_test"]["row_index_sha256_int64_little_endian"]
    )
    expected_rows = int(config["fixed_test"]["rows"])
    per_seed_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    user_pairs: dict[
        tuple[str, int, str, str, str],
        list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ] = defaultdict(list)
    input_hashes: dict[str, dict[str, str]] = {}

    clean_pattern = str(config["clean_control"]["prediction_pattern"])
    for seed in EXPECTED_SEEDS:
        clean_predictions_path = args.clean_root / clean_pattern.format(seed=seed)
        leaky_seed_root = args.leaky_root / f"seed_{seed}"
        leaky_predictions_path = leaky_seed_root / "predictions.npz"
        clean = load_prediction_archive(clean_predictions_path)
        leaky = load_prediction_archive(leaky_predictions_path)
        require(
            np.array_equal(clean.row_index, leaky.row_index),
            f"seed {seed}: clean/leaky row order mismatch",
        )
        require(len(clean.row_index) == expected_rows, "fixed-test row count")
        require(index_sha256(clean.row_index) == expected_hash, "fixed-test row hash")
        seed_audit_path = leaky_seed_root / "audit.json"
        seed_audit = json.loads(seed_audit_path.read_text(encoding="utf-8"))
        require(seed_audit.get("all_assertions_pass") is True, "leaky seed audit")
        require(seed_audit.get("valid_for_generalization") is False, "seed validity")
        require(seed_audit.get("protocol") == PROTOCOL, "seed protocol")
        require(seed_audit.get("seed") == seed, "seed audit identity")
        require(
            seed_audit.get("training_history_mode") == TRAINING_HISTORY_MODE,
            "seed history mode",
        )
        require(
            seed_audit.get("clean_checkpoint_reused_or_warm_started") is False,
            "clean checkpoint reuse",
        )
        row_index = clean.row_index
        target = np.asarray(targets[row_index], dtype=np.float64)
        users = np.asarray(users_all[row_index], dtype=np.int64)
        sessions = np.asarray(sessions_all[row_index], dtype=np.int64)
        require(np.isfinite(target).all(), "non-finite targets")
        require(len(np.unique(users)) == int(config["fixed_test"]["users"]), "user support")
        require(
            len(np.unique(sessions)) == int(config["fixed_test"]["sessions"]),
            "session support",
        )

        for horizon_position, horizon in enumerate(HORIZONS):
            clean_users, clean_metrics = hierarchical_user_point_metrics(
                clean.quantiles[:, horizon_position, 3],
                target[:, horizon_position],
                users,
                sessions,
            )
            leaky_users, leaky_metrics = hierarchical_user_point_metrics(
                leaky.quantiles[:, horizon_position, 3],
                target[:, horizon_position],
                users,
                sessions,
            )
            require(np.array_equal(clean_users, leaky_users), "point user alignment")
            aggregates: dict[str, float] = {}
            for metric in ("mae_bpm", "rmse_bpm", "bias_bpm"):
                clean_value = clean_metrics[metric]
                leaky_value = leaky_metrics[metric]
                difference = leaky_value - clean_value
                aggregates[f"clean_{metric}"] = float(clean_value.mean())
                aggregates[f"leaky_{metric}"] = float(leaky_value.mean())
                aggregates[f"leaky_minus_clean_{metric}"] = float(difference.mean())
                user_pairs[("point", int(horizon), "", "", metric)].append(
                    (clean_users, clean_value, leaky_value)
                )
            clean_mae = aggregates["clean_mae_bpm"]
            aggregates["relative_mae_optimism_percent"] = (
                100.0 * (clean_mae - aggregates["leaky_mae_bpm"]) / clean_mae
            )
            per_seed_rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "valid_for_generalization": False,
                    "leaderboard_eligible": False,
                    "seed": seed,
                    "horizon_seconds": horizon,
                    **aggregates,
                    "users": int(len(clean_users)),
                    "sessions": int(len(np.unique(sessions))),
                    "origins": int(len(row_index)),
                    "difference_definition": "leaky minus clean; negative MAE denotes optimism",
                }
            )

        clean_thresholds = load_thresholds(
            args.clean_root
            / f"seed_{seed}"
            / "strict_temporal"
            / "m"
            / "conformal_thresholds.json",
            leaky=False,
        )
        leaky_thresholds = load_thresholds(
            leaky_seed_root / "m" / "conformal_thresholds.json", leaky=True
        )
        for coverage, (lower_position, upper_position) in INTERVALS.items():
            coverage_key = str(float(coverage))
            for calibrated in (False, True):
                for horizon_position, horizon in enumerate(HORIZONS):
                    clean_adjustment = (
                        clean_thresholds[coverage_key][horizon_position]
                        if calibrated
                        else 0.0
                    )
                    leaky_adjustment = (
                        leaky_thresholds[coverage_key][horizon_position]
                        if calibrated
                        else 0.0
                    )
                    clean_lower = np.clip(
                        clean.quantiles[:, horizon_position, lower_position]
                        - clean_adjustment,
                        30.0,
                        240.0,
                    )
                    clean_upper = np.clip(
                        clean.quantiles[:, horizon_position, upper_position]
                        + clean_adjustment,
                        30.0,
                        240.0,
                    )
                    leaky_lower = np.clip(
                        leaky.quantiles[:, horizon_position, lower_position]
                        - leaky_adjustment,
                        30.0,
                        240.0,
                    )
                    leaky_upper = np.clip(
                        leaky.quantiles[:, horizon_position, upper_position]
                        + leaky_adjustment,
                        30.0,
                        240.0,
                    )
                    horizon_target = target[:, horizon_position]
                    clean_covered = (
                        (horizon_target >= clean_lower)
                        & (horizon_target <= clean_upper)
                    ).astype(np.float64)
                    leaky_covered = (
                        (horizon_target >= leaky_lower)
                        & (horizon_target <= leaky_upper)
                    ).astype(np.float64)
                    clean_width = clean_upper - clean_lower
                    leaky_width = leaky_upper - leaky_lower
                    coverage_users, clean_user_coverage = hierarchical_user_average(
                        clean_covered, users, sessions
                    )
                    leaky_coverage_users, leaky_user_coverage = hierarchical_user_average(
                        leaky_covered, users, sessions
                    )
                    width_users, clean_user_width = hierarchical_user_average(
                        clean_width, users, sessions
                    )
                    leaky_width_users, leaky_user_width = hierarchical_user_average(
                        leaky_width, users, sessions
                    )
                    require(
                        np.array_equal(coverage_users, leaky_coverage_users)
                        and np.array_equal(coverage_users, width_users)
                        and np.array_equal(coverage_users, leaky_width_users),
                        "interval user alignment",
                    )
                    calibrated_label = str(bool(calibrated)).lower()
                    user_pairs[
                        (
                            "interval",
                            int(horizon),
                            coverage_key,
                            calibrated_label,
                            "picp",
                        )
                    ].append((coverage_users, clean_user_coverage, leaky_user_coverage))
                    user_pairs[
                        (
                            "interval",
                            int(horizon),
                            coverage_key,
                            calibrated_label,
                            "mean_interval_width_bpm",
                        )
                    ].append((coverage_users, clean_user_width, leaky_user_width))
                    clean_picp = float(clean_user_coverage.mean())
                    leaky_picp = float(leaky_user_coverage.mean())
                    clean_mean_width = float(clean_user_width.mean())
                    leaky_mean_width = float(leaky_user_width.mean())
                    interval_rows.append(
                        {
                            "analysis_version": ANALYSIS_VERSION,
                            "valid_for_generalization": False,
                            "coverage_guarantee_valid": False,
                            "seed": seed,
                            "horizon_seconds": horizon,
                            "nominal_coverage": coverage,
                            "calibrated": calibrated,
                            "clean_picp": clean_picp,
                            "leaky_picp": leaky_picp,
                            "leaky_minus_clean_picp": leaky_picp - clean_picp,
                            "clean_absolute_coverage_error": abs(clean_picp - coverage),
                            "leaky_absolute_coverage_error": abs(leaky_picp - coverage),
                            "clean_mean_interval_width_bpm": clean_mean_width,
                            "leaky_mean_interval_width_bpm": leaky_mean_width,
                            "leaky_minus_clean_width_bpm": leaky_mean_width
                            - clean_mean_width,
                            "clean_conformal_adjustment_bpm": clean_adjustment,
                            "leaky_conformal_adjustment_bpm": leaky_adjustment,
                            "users": int(len(coverage_users)),
                            "sessions": int(len(np.unique(sessions))),
                            "origins": int(len(row_index)),
                        }
                    )

        input_hashes[str(seed)] = {
            "clean_predictions": sha256_file(clean_predictions_path),
            "leaky_predictions": sha256_file(leaky_predictions_path),
            "leaky_audit": sha256_file(seed_audit_path),
        }

    user_seed_mean_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    reporting = config["reporting"]
    replicates = int(reporting["user_bootstrap_replicates"])
    base_seed = int(reporting["user_bootstrap_seed"])
    for key, seed_pairs in sorted(user_pairs.items()):
        family, horizon, nominal, calibrated, metric = key
        require(len(seed_pairs) == len(EXPECTED_SEEDS), f"incomplete matched seeds: {key}")
        reference_users = seed_pairs[0][0]
        require(
            all(np.array_equal(reference_users, item[0]) for item in seed_pairs),
            f"user support differs across seeds: {key}",
        )
        clean_seed_mean = np.mean(np.stack([item[1] for item in seed_pairs]), axis=0)
        leaky_seed_mean = np.mean(np.stack([item[2] for item in seed_pairs]), axis=0)
        difference = leaky_seed_mean - clean_seed_mean
        for position, user in enumerate(reference_users):
            user_seed_mean_rows.append(
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "valid_for_generalization": False,
                    "metric_family": family,
                    "horizon_seconds": horizon,
                    "nominal_coverage": nominal,
                    "calibrated": calibrated,
                    "metric": metric,
                    "user_index": int(user),
                    "clean_seed_mean": float(clean_seed_mean[position]),
                    "leaky_seed_mean": float(leaky_seed_mean[position]),
                    "leaky_minus_clean": float(difference[position]),
                    "matched_seeds": ";".join(str(seed) for seed in EXPECTED_SEEDS),
                }
            )
        estimate, low, high = percentile_user_bootstrap(
            difference,
            replicates=replicates,
            seed=stable_bootstrap_seed(base_seed, key),
        )
        bootstrap_rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "valid_for_generalization": False,
                "metric_family": family,
                "horizon_seconds": horizon,
                "nominal_coverage": nominal,
                "calibrated": calibrated,
                "metric": metric,
                "leaky_minus_clean_estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "users": int(len(reference_users)),
                "matched_seed_count": len(EXPECTED_SEEDS),
                "matched_seeds": ";".join(str(seed) for seed in EXPECTED_SEEDS),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "user",
                "bootstrap_order": reporting["bootstrap_order"],
                "direction": (
                    "negative denotes optimistic error"
                    if family == "point" and metric in {"mae_bpm", "rmse_bpm"}
                    else "leaky minus clean"
                ),
            }
        )

    seed_summary_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        selected = [row for row in per_seed_rows if row["horizon_seconds"] == horizon]
        require(len(selected) == len(EXPECTED_SEEDS), "point seed summary support")
        summary: dict[str, object] = {
            "analysis_version": ANALYSIS_VERSION,
            "valid_for_generalization": False,
            "horizon_seconds": horizon,
            "n_seeds": len(EXPECTED_SEEDS),
            "seeds": ";".join(str(seed) for seed in EXPECTED_SEEDS),
            "seeds_are_independent_participants": False,
        }
        for column in (
            "clean_mae_bpm",
            "leaky_mae_bpm",
            "leaky_minus_clean_mae_bpm",
            "relative_mae_optimism_percent",
            "clean_rmse_bpm",
            "leaky_rmse_bpm",
            "leaky_minus_clean_rmse_bpm",
            "clean_bias_bpm",
            "leaky_bias_bpm",
            "leaky_minus_clean_bias_bpm",
        ):
            values = np.asarray([float(row[column]) for row in selected])
            summary[f"{column}_median"] = float(np.median(values))
            summary[f"{column}_minimum"] = float(values.min())
            summary[f"{column}_maximum"] = float(values.max())
        seed_summary_rows.append(summary)

    paths = output_paths(args.output_dir)
    atomic_csv(paths["per_seed"], per_seed_rows)
    atomic_csv(paths["seed_summary"], seed_summary_rows)
    atomic_csv(paths["user_seed_mean"], user_seed_mean_rows)
    atomic_csv(paths["user_bootstrap"], bootstrap_rows)
    atomic_csv(paths["interval"], interval_rows)
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "protocol": PROTOCOL,
        "valid_for_generalization": False,
        "leaderboard_eligible": False,
        "invalid_reason": "same-session overlapping windows deliberately contaminate fitting and calibration",
        "acknowledge_invalid_generalization": True,
        "matched_seed_count": len(EXPECTED_SEEDS),
        "matched_seeds": list(EXPECTED_SEEDS),
        "seeds_are_independent_participants": False,
        "fixed_test_rows": expected_rows,
        "fixed_test_row_index_sha256_int64_little_endian": expected_hash,
        "clean_leaky_exact_row_order_match_all_seeds": True,
        "training_history_mode": TRAINING_HISTORY_MODE,
        "paired_hierarchy": "origin within session, session within user, paired user difference",
        "user_effect_seed_aggregation": "mean across three matched seeds before bootstrap",
        "bootstrap_replicates": replicates,
        "bootstrap_unit": "user",
        "coverage_guarantee_valid": False,
        "output_rows": {
            "per_seed": len(per_seed_rows),
            "seed_summary": len(seed_summary_rows),
            "user_seed_mean": len(user_seed_mean_rows),
            "user_bootstrap": len(bootstrap_rows),
            "interval": len(interval_rows),
        },
        "configuration": str(args.configuration.resolve()),
        "configuration_sha256": sha256_file(args.configuration),
        "partition_audit": str(args.partition_audit.resolve()),
        "partition_audit_sha256": sha256_file(args.partition_audit),
        "input_hashes": input_hashes,
        "outputs": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in paths.items()
            if key != "audit"
        },
    }
    payload["all_assertions_pass"] = (
        len(input_hashes) == len(EXPECTED_SEEDS)
        and len(per_seed_rows) == len(EXPECTED_SEEDS) * len(HORIZONS)
        and bool(user_seed_mean_rows)
        and bool(bootstrap_rows)
        and bool(interval_rows)
    )
    atomic_json(paths["audit"], payload)
    require(bool(payload["all_assertions_pass"]), "aggregation audit failed")
    return payload


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(
        description="Aggregate paired clean-versus-leaky v0.28 negative-control results."
    )
    result.add_argument(
        "--acknowledge-invalid-generalization", action="store_true", required=True
    )
    result.add_argument(
        "--configuration",
        type=Path,
        default=root / "configs" / "leaky_negative_control_v0_28_0.json",
    )
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--leaky-root", type=Path, required=True)
    result.add_argument("--clean-root", type=Path, required=True)
    result.add_argument("--partition-audit", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(aggregate(parser().parse_args()), ensure_ascii=False))
