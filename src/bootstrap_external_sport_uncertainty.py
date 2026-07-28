from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.21.0"
SOURCE_MODEL_VERSION = "0.11.0"
HORIZONS = (60, 180, 300)
QUANTILES = np.asarray(
    [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float32
)
INTERVALS = {
    0.50: (2, 4),
    0.80: (1, 5),
    0.90: (0, 6),
}
SPORTS = {
    1: "outdoor_cycling",
    2: "indoor_virtual_cycling",
    3: "running",
}
PARTITION_CALIBRATION = 3
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
    values = np.asarray(user_values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("expected at least two users and a two-dimensional matrix")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    generator = np.random.default_rng(seed)
    result = np.empty((replicates, values.shape[1]), dtype=np.float64)
    for start in range(0, replicates, batch_size):
        end = min(replicates, start + batch_size)
        indices = generator.integers(
            0, len(values), size=(end - start, len(values))
        )
        result[start:end] = values[indices].mean(axis=1)
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
    frame = pd.DataFrame(
        {
            **{name: np.asarray(value) for name, value in metrics.items()},
            "user": users,
            "session": sessions,
        }
    )
    metric_names = list(metrics)
    session_level = frame.groupby(["user", "session"], sort=False)[
        metric_names
    ].mean()
    return session_level.groupby(level="user", sort=False).mean()


def calibrated_bounds(
    prediction: np.ndarray,
    thresholds: dict[str, list[float]],
    horizon_position: int,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    if prediction.ndim != 2 or prediction.shape[1] != len(QUANTILES):
        raise ValueError("prediction must have shape (origins, 7)")
    result: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for coverage, (lower_position, upper_position) in INTERVALS.items():
        adjustment = float(thresholds[str(coverage)][horizon_position])
        result[coverage] = (
            np.clip(prediction[:, lower_position] - adjustment, 30.0, 240.0),
            np.clip(prediction[:, upper_position] + adjustment, 30.0, 240.0),
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


def reference_interval_row(
    frame: pd.DataFrame,
    family: str,
    horizon: int,
    coverage: float,
) -> pd.Series:
    selected = frame[
        (frame["regime"] == f"goldencheetah_external__{family}")
        & (frame["mode"] == "zero_history")
        & (frame["horizon_seconds"] == horizon)
        & np.isclose(frame["nominal_coverage"], coverage)
        & frame["calibrated"]
    ]
    require(
        len(selected) == 1,
        f"interval reference row count for {family}/{horizon}/{coverage}: "
        f"{len(selected)}",
    )
    return selected.iloc[0]


def reference_wis_row(
    frame: pd.DataFrame,
    family: str,
    horizon: int,
) -> pd.Series:
    selected = frame[
        (frame["regime"] == f"goldencheetah_external__{family}")
        & (frame["mode"] == "zero_history")
        & (frame["horizon_seconds"] == horizon)
        & frame["calibrated"]
    ]
    require(
        len(selected) == 1,
        f"WIS reference row count for {family}/{horizon}: {len(selected)}",
    )
    return selected.iloc[0]


def load_thresholds(path: Path) -> tuple[dict[str, object], dict[str, list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = payload.get("thresholds", payload)
    require("zero_history" in root, f"{path}: missing zero_history thresholds")
    thresholds = root["zero_history"]
    for coverage in ("0.5", "0.8", "0.9"):
        require(
            coverage in thresholds and len(thresholds[coverage]) == 3,
            f"{path}: invalid {coverage} thresholds",
        )
        require(
            np.isfinite(np.asarray(thresholds[coverage], dtype=np.float64)).all(),
            f"{path}: non-finite {coverage} thresholds",
        )
    return payload, thresholds


def analyze(args: argparse.Namespace) -> dict[str, object]:
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
    arrays = {
        name: np.load(args.array_dir / filename, mmap_mode="r")
        for name, filename in array_names.items()
    }
    row_counts = {name: int(len(value)) for name, value in arrays.items()}
    require(len(set(row_counts.values())) == 1, f"array mismatch: {row_counts}")
    total_rows = next(iter(row_counts.values()))

    for path in (
        args.predictions,
        args.thresholds,
        args.model_audit,
        args.interval_reference,
        args.probabilistic_reference,
    ):
        require(path.exists(), f"missing source file {path}")

    threshold_hash_before = sha256_file(args.thresholds)
    threshold_payload, thresholds = load_thresholds(args.thresholds)
    model_audit = json.loads(args.model_audit.read_text(encoding="utf-8"))
    require(
        str(model_audit.get("model_version")) == SOURCE_MODEL_VERSION,
        "source model version mismatch",
    )
    require(
        bool(model_audit.get("all_assertions_pass")),
        "source model audit did not pass",
    )

    with np.load(args.predictions) as source:
        required_fields = {
            "row_index",
            "history_quantiles",
            "zero_history_quantiles",
        }
        require(
            required_fields.issubset(source.files),
            f"prediction NPZ missing {sorted(required_fields - set(source.files))}",
        )
        prediction_rows = np.asarray(source["row_index"], dtype=np.int64)
        external_mask = (
            (arrays["dataset"][prediction_rows] == 1)
            & (arrays["external"][prediction_rows] == EXTERNAL_FROZEN)
        )
        external_predictions = np.asarray(source["zero_history_quantiles"])[
            external_mask
        ]

    expected_prediction_rows = np.flatnonzero(
        ((arrays["dataset"] == 0) & (arrays["evaluation"] == 1))
        | (
            (arrays["dataset"] == 1)
            & (arrays["external"] == EXTERNAL_FROZEN)
        )
    )
    require(
        np.array_equal(prediction_rows, expected_prediction_rows),
        "prediction NPZ row mapping mismatch",
    )
    external_rows = prediction_rows[external_mask]
    expected_external_rows = np.flatnonzero(
        (arrays["dataset"] == 1)
        & (arrays["external"] == EXTERNAL_FROZEN)
    )
    require(
        np.array_equal(external_rows, expected_external_rows),
        "frozen external row mapping mismatch",
    )
    require(
        external_predictions.shape == (len(external_rows), 3, 7),
        f"external prediction shape mismatch: {external_predictions.shape}",
    )
    require(
        np.isfinite(external_predictions).all(),
        "non-finite external predictions",
    )
    quantile_crossings = int(
        (np.diff(external_predictions, axis=2) < -1e-6).sum()
    )
    require(quantile_crossings == 0, "external prediction quantile crossings")

    external_sports = np.asarray(arrays["sport"][external_rows])
    require(
        set(np.unique(external_sports).tolist()) == set(SPORTS),
        f"unexpected frozen external sport codes: {np.unique(external_sports)}",
    )
    external_users = np.asarray(arrays["users"][external_rows])
    external_sessions = np.asarray(arrays["sessions"][external_rows])
    session_sport = pd.DataFrame(
        {"session": external_sessions, "sport": external_sports}
    ).drop_duplicates()
    session_sport_conflicts = int(session_sport["session"].duplicated().sum())
    require(
        session_sport_conflicts == 0,
        f"external sessions assigned to multiple sports: {session_sport_conflicts}",
    )

    calibration_rows = np.flatnonzero(
        (arrays["dataset"] == 0)
        & (arrays["unseen"] == PARTITION_CALIBRATION)
        & (arrays["evaluation"] == 1)
    )
    calibration_users = np.unique(arrays["users"][calibration_rows])
    require(
        int(threshold_payload.get("calibration_rows", -1))
        == len(calibration_rows),
        "threshold calibration-row count mismatch",
    )
    require(
        int(threshold_payload.get("calibration_users", -1))
        == len(calibration_users),
        "threshold calibration-user count mismatch",
    )
    require(
        int((arrays["dataset"][calibration_rows] != 0).sum()) == 0,
        "non-Endomondo row entered threshold calibration",
    )

    interval_reference = pd.read_csv(args.interval_reference)
    probabilistic_reference = pd.read_csv(args.probabilistic_reference)
    for frame in (interval_reference, probabilistic_reference):
        frame["calibrated"] = (
            frame["calibrated"].astype(str).str.lower() == "true"
        )

    output_rows: list[dict[str, object]] = []
    maximum_reference_delta = {
        "picp": 0.0,
        "mean_interval_width_bpm": 0.0,
        "weighted_interval_score": 0.0,
    }
    family_support: dict[str, dict[str, int]] = {}

    for sport_code, family in SPORTS.items():
        family_mask = external_sports == sport_code
        rows = external_rows[family_mask]
        predictions = external_predictions[family_mask]
        users = np.asarray(arrays["users"][rows])
        sessions = np.asarray(arrays["sessions"][rows])
        support = {
            "users": int(len(np.unique(users))),
            "sessions": int(len(np.unique(sessions))),
            "origins": int(len(rows)),
        }
        family_support[family] = support
        require(min(support.values()) > 1, f"insufficient support for {family}")

        for horizon_position, horizon in enumerate(HORIZONS):
            target = np.asarray(
                arrays["targets"][rows, horizon_position], dtype=np.float32
            )
            prediction = predictions[:, horizon_position]
            median = prediction[:, 3]
            bounds = calibrated_bounds(
                prediction, thresholds, horizon_position
            )
            metrics: dict[str, np.ndarray] = {}
            for coverage, (lower, upper) in bounds.items():
                label = int(round(100 * coverage))
                metrics[f"picp_{label}"] = (
                    (target >= lower) & (target <= upper)
                ).astype(np.float32)
                metrics[f"width_{label}"] = upper - lower
            metrics["wis"] = weighted_interval_score(
                median, target, bounds
            )
            user_metrics = hierarchical_user_metrics(
                metrics, users, sessions
            )
            bootstrap = bootstrap_user_means(
                user_metrics[list(metrics)].to_numpy(dtype=np.float64),
                args.bootstrap_replicates,
                stable_seed(args.seed, f"{family}|{horizon}|all_metrics"),
            )

            row: dict[str, object] = {
                "analysis_version": ANALYSIS_VERSION,
                "source_model_version": SOURCE_MODEL_VERSION,
                "data_source": "GoldenCheetah",
                "sport_family": family,
                "horizon_seconds": horizon,
                "mode": "zero_history",
                "calibrated": True,
            }
            for metric_position, metric_name in enumerate(metrics):
                point = float(user_metrics[metric_name].mean())
                low, high = percentile_interval(bootstrap[:, metric_position])
                if metric_name.startswith("width_"):
                    label = metric_name.removeprefix("width_")
                    output_name = f"mean_{label}_interval_width_bpm"
                elif metric_name == "wis":
                    output_name = "weighted_interval_score"
                else:
                    output_name = metric_name
                row[output_name] = point
                row[f"{output_name}_ci_low"] = low
                row[f"{output_name}_ci_high"] = high

            for coverage in INTERVALS:
                label = int(round(100 * coverage))
                reference = reference_interval_row(
                    interval_reference, family, horizon, coverage
                )
                recomputed = {
                    "picp": (
                        float(row[f"picp_{label}"]),
                        float(reference["picp"]),
                    ),
                    "mean_interval_width_bpm": (
                        float(row[f"mean_{label}_interval_width_bpm"]),
                        float(reference["mean_interval_width_bpm"]),
                    ),
                }
                for metric, (value, expected) in recomputed.items():
                    maximum_reference_delta[metric] = max(
                        maximum_reference_delta[metric], abs(value - expected)
                    )
                require(
                    int(reference["users"]) == support["users"]
                    and int(reference["sessions"]) == support["sessions"]
                    and int(reference["origins"]) == support["origins"],
                    f"interval support mismatch for {family}/{horizon}/{coverage}",
                )

            wis_reference = reference_wis_row(
                probabilistic_reference, family, horizon
            )
            maximum_reference_delta["weighted_interval_score"] = max(
                maximum_reference_delta["weighted_interval_score"],
                abs(
                    float(row["weighted_interval_score"])
                    - float(wis_reference["weighted_interval_score"])
                ),
            )
            require(
                int(wis_reference["users"]) == support["users"]
                and int(wis_reference["sessions"]) == support["sessions"]
                and int(wis_reference["origins"]) == support["origins"],
                f"WIS support mismatch for {family}/{horizon}",
            )

            row.update(
                {
                    **support,
                    "bootstrap_replicates": args.bootstrap_replicates,
                    "bootstrap_unit": "user",
                    "aggregation": (
                        "origin-within-session, session-within-user, "
                        "equal-user mean"
                    ),
                }
            )
            output_rows.append(row)

    require(len(output_rows) == 9, f"expected 9 rows, got {len(output_rows)}")
    require(
        sum(value["origins"] for value in family_support.values())
        == len(external_rows),
        "sport-family origin supports do not partition the external set",
    )
    require(
        sum(value["sessions"] for value in family_support.values())
        == len(np.unique(external_sessions)),
        "sport-family session supports do not partition the external set",
    )
    tolerance = 5e-6
    require(
        max(maximum_reference_delta.values()) <= tolerance,
        f"reference mismatch: {maximum_reference_delta}",
    )

    output_frame = pd.DataFrame(output_rows)
    metric_names = [
        *(f"picp_{int(coverage * 100)}" for coverage in INTERVALS),
        *(
            f"mean_{int(coverage * 100)}_interval_width_bpm"
            for coverage in INTERVALS
        ),
        "weighted_interval_score",
    ]
    numeric_columns = [
        column
        for metric in metric_names
        for column in (metric, f"{metric}_ci_low", f"{metric}_ci_high")
    ]
    output_checks = {
        "duplicate_analysis_keys": int(
            output_frame.duplicated(["sport_family", "horizon_seconds"]).sum()
        ),
        "nonfinite_metric_values": int(
            (~np.isfinite(output_frame[numeric_columns].to_numpy())).sum()
        ),
        "picp_range_failures": int(
            sum(
                (~output_frame[f"picp_{int(coverage * 100)}"].between(0.0, 1.0)).sum()
                for coverage in INTERVALS
            )
        ),
        "nonpositive_width_or_wis_failures": int(
            sum(
                (output_frame[f"mean_{int(coverage * 100)}_interval_width_bpm"] <= 0).sum()
                for coverage in INTERVALS
            )
            + (output_frame["weighted_interval_score"] <= 0).sum()
        ),
        "point_outside_ci": int(
            sum(
                (
                    (output_frame[metric] < output_frame[f"{metric}_ci_low"])
                    | (output_frame[metric] > output_frame[f"{metric}_ci_high"])
                ).sum()
                for metric in metric_names
            )
        ),
        "reversed_ci": int(
            sum(
                (
                    output_frame[f"{metric}_ci_low"]
                    > output_frame[f"{metric}_ci_high"]
                ).sum()
                for metric in metric_names
            )
        ),
    }
    require(
        not any(output_checks.values()),
        f"output validation failed: {output_checks}",
    )

    atomic_csv(args.output_csv, output_rows)
    threshold_hash_after = sha256_file(args.thresholds)
    require(
        threshold_hash_after == threshold_hash_before,
        "frozen threshold file changed during analysis",
    )
    source_files: dict[str, object] = {
        "predictions": {
            "path": str(args.predictions),
            "sha256": sha256_file(args.predictions),
        },
        "thresholds": {
            "path": str(args.thresholds),
            "sha256_before": threshold_hash_before,
            "sha256_after": threshold_hash_after,
            "unchanged": threshold_hash_before == threshold_hash_after,
        },
        "source_model_audit": {
            "path": str(args.model_audit),
            "sha256": sha256_file(args.model_audit),
        },
        "interval_reference": {
            "path": str(args.interval_reference),
            "sha256": sha256_file(args.interval_reference),
        },
        "probabilistic_reference": {
            "path": str(args.probabilistic_reference),
            "sha256": sha256_file(args.probabilistic_reference),
        },
        "arrays": {
            name: {
                "path": str(args.array_dir / filename),
                "sha256": sha256_file(args.array_dir / filename),
            }
            for name, filename in array_names.items()
        },
    }
    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "intended_use": (
            "Frozen GoldenCheetah sport-stratified post-CQR uncertainty "
            "metrics with user-bootstrap confidence intervals; no training "
            "or recalibration"
        ),
        "source_files": source_files,
        "array_rows": total_rows,
        "array_row_counts": row_counts,
        "alignment": {
            "prediction_npz_rows": int(len(prediction_rows)),
            "reconstructed_prediction_rows": int(len(expected_prediction_rows)),
            "prediction_mapping_exact": True,
            "external_npz_rows": int(len(external_rows)),
            "reconstructed_external_rows": int(len(expected_external_rows)),
            "external_mapping_exact": True,
            "quantile_crossing_failures": quantile_crossings,
            "nonfinite_prediction_values": 0,
            "external_session_sport_conflicts": session_sport_conflicts,
        },
        "frozen_calibration": {
            "calibration_source": "Endomondo",
            "calibration_partition": "unseen-user calibration",
            "reconstructed_origins": int(len(calibration_rows)),
            "reconstructed_users": int(len(calibration_users)),
            "threshold_payload_origins": int(threshold_payload["calibration_rows"]),
            "threshold_payload_users": int(threshold_payload["calibration_users"]),
            "non_endomondo_calibration_origins": 0,
            "goldencheetah_recalibration_performed": False,
            "threshold_mode": "zero_history",
            "thresholds_bpm": thresholds,
            "threshold_file_unchanged": True,
        },
        "external_support": {
            "users": int(len(np.unique(external_users))),
            "sessions": int(len(np.unique(external_sessions))),
            "origins": int(len(external_rows)),
            "by_sport_family": family_support,
        },
        "metrics": {
            "picp_and_width": (
                "post-CQR central 50%, 80%, and 90% intervals"
            ),
            "weighted_interval_score": (
                "post-CQR central 50%, 80%, and 90% intervals plus median; "
                "same formula as evaluate_probabilistic_metrics.py"
            ),
            "aggregation": (
                "origin-within-session, session-within-user, equal-user mean"
            ),
        },
        "bootstrap": {
            "unit": "user",
            "replicates": args.bootstrap_replicates,
            "base_seed": args.seed,
            "interval": "two-sided 95% percentile",
            "all_metrics_resampled_jointly_within_sport_horizon": True,
        },
        "maximum_absolute_reference_recomputation_delta": maximum_reference_delta,
        "reference_tolerance": tolerance,
        "output_csv": str(args.output_csv),
        "output_csv_sha256": sha256_file(args.output_csv),
        "output_rows": len(output_rows),
        "output_checks": output_checks,
        "limitations": [
            (
                "Bootstrap intervals quantify between-user sampling variation "
                "conditional on the single frozen v0.11.0 checkpoint; they do "
                "not include model-training or seed variability."
            ),
            (
                "The bootstrap resamples users only. It treats each user's "
                "observed session set as fixed and therefore does not separately "
                "quantify within-user session sampling variation."
            ),
            (
                "The original CQR thresholds were estimated from correlated "
                "Endomondo forecast origins. These empirical intervals do not "
                "establish finite-sample user-level or sport-conditional coverage."
            ),
            (
                "Sport families have unequal support and users may contribute to "
                "more than one family; intervals are descriptive within each "
                "family and are not independent across families."
            ),
            (
                "The three sport families partition the frozen primary external "
                "origin set, but their composition differs from Endomondo; this "
                "analysis does not standardize sport mix."
            ),
        ],
        "all_assertions_pass": True,
    }
    atomic_json(args.audit_json, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap GoldenCheetah sport-stratified post-CQR uncertainty "
            "metrics without retraining or recalibration."
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
        "--model-audit",
        type=Path,
        default=Path(
            "outputs/audit/uncertainty_user_generalization_v0_11_0.json"
        ),
    )
    parser.add_argument(
        "--interval-reference",
        type=Path,
        default=Path(
            "outputs/results/uncertainty_interval_metrics_v0_11_0.csv"
        ),
    )
    parser.add_argument(
        "--probabilistic-reference",
        type=Path,
        default=Path("outputs/results/probabilistic_metrics_v0_11_0.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/results/external_sport_uncertainty_bootstrap_v0_21_0.csv"
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/audit/external_sport_uncertainty_bootstrap_v0_21_0.json"
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
