"""Audit target-availability selection with horizon-specific evaluation cohorts.

The fitted models and the session sets are deliberately left unchanged.  Within
each original evaluation session, this script rebuilds 5-min-spaced forecast
origins from the raw heart-rate stream and evaluates a horizon whenever that
horizon's target is available.  The original complete-three-target cohort is
rebuilt in parallel as a hard reproducibility control.

This is a parameter-free persistence sensitivity, not a retraining analysis.
It isolates endpoint selection from changes in users, sessions, models, or
calibration data.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from build_forecast_origins import (
    CONTEXT_SECONDS,
    MAX_CONTEXT_GAP_SECONDS,
    MIN_CONTEXT_BIN_COVERAGE,
    context_bin_index,
    interpolate_target,
    parse_golden_session,
    valid_hr,
)
from data_audit import extract_array_bytes, parse_numeric_array
from run_naive_baselines import extract_hr_context, load_series, persistence_prediction, valid_observations


VERSION = "0.29.0"
HORIZONS = (60, 180, 300)
EVALUATION_STRIDE_SECONDS = 300
TOTAL_CONTEXT_BINS = 30
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260728
SHARED_EXTERNAL_SPORT_CODES = {1, 2, 3}

REGIME_LABELS = {
    "strict_temporal": "within_user_temporal_test",
    "unseen_user": "unseen_user_test",
    "external": "goldencheetah_frozen_external",
}

EXPECTED_COMMON = {
    "within_user_temporal_test": {
        "users": 948,
        "sessions": 16_012,
        "origins": 104_144,
        "mae": {60: 6.7259420633675, 180: 8.75667761981665, 300: 9.587486016407192},
    },
    "unseen_user_test": {
        "users": 105,
        "sessions": 15_026,
        "origins": 101_184,
        "mae": {60: 6.559022520910067, 180: 8.626678339409843, 300: 9.599249692307811},
    },
    "goldencheetah_frozen_external": {
        "users": 144,
        "sessions": 31_851,
        "origins": 531_725,
        "mae": {60: 7.929412012105658, 180: 11.282177513374664, 300: 12.569113924961153},
    },
}


@dataclass(frozen=True)
class SessionSpec:
    dataset: str
    session_key: str
    session_index: int
    user_index: int
    sport_code: int
    regimes: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def first_evaluation_time(lower_bound: float) -> int:
    return int(math.ceil(lower_bound / EVALUATION_STRIDE_SECONDS) * EVALUATION_STRIDE_SECONDS)


def context_is_eligible(valid_times: Sequence[float], origin_time: float) -> bool:
    context_start = origin_time - CONTEXT_SECONDS
    left = bisect.bisect_left(valid_times, context_start)
    right = bisect.bisect_right(valid_times, origin_time)
    context_times = valid_times[left:right]
    if not context_times:
        return False

    occupied = {
        index
        for timestamp in context_times
        if (index := context_bin_index(timestamp, context_start)) is not None
    }
    if len(occupied) / TOTAL_CONTEXT_BINS < MIN_CONTEXT_BIN_COVERAGE:
        return False

    gaps = [context_times[0] - context_start, origin_time - context_times[-1]]
    gaps.extend(current - previous for previous, current in zip(context_times, context_times[1:]))
    return max(gaps) <= MAX_CONTEXT_GAP_SECONDS


def candidate_origins(timestamps: Sequence[float]) -> range:
    if len(timestamps) < 2:
        return range(0)
    first = first_evaluation_time(float(timestamps[0]) + CONTEXT_SECONDS)
    last = int(math.floor(float(timestamps[-1]) - min(HORIZONS)))
    if first > last:
        return range(0)
    return range(first, last + 1, EVALUATION_STRIDE_SECONDS)


def build_session_errors(
    timestamps: Sequence[float],
    heart_rate: Sequence[float],
    persistence_for_origin,
) -> dict[tuple[int, str], list[float]]:
    """Return absolute errors by (horizon, cohort) for one session."""
    if len(timestamps) != len(heart_rate) or len(timestamps) < 2:
        return {}
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        return {}
    valid_pairs = [
        (float(timestamp), float(hr))
        for timestamp, hr in zip(timestamps, heart_rate)
        if valid_hr(float(hr))
    ]
    if len(valid_pairs) < 2:
        return {}
    valid_times = [item[0] for item in valid_pairs]
    valid_values = [item[1] for item in valid_pairs]

    output: defaultdict[tuple[int, str], list[float]] = defaultdict(list)
    for origin in candidate_origins(timestamps):
        origin_time = float(origin)
        if not context_is_eligible(valid_times, origin_time):
            continue
        targets = {
            horizon: interpolate_target(valid_times, valid_values, origin_time + horizon)
            for horizon in HORIZONS
        }
        if not any(target is not None for target in targets.values()):
            continue
        prediction = float(persistence_for_origin(origin_time))
        common = all(target is not None for target in targets.values())
        for horizon, target in targets.items():
            if target is None:
                continue
            error = abs(prediction - float(target[0]))
            output[(horizon, "horizon_specific")].append(error)
            if common:
                output[(horizon, "common_three_target")].append(error)
    return dict(output)


def load_session_specs(array_dir: Path) -> dict[tuple[str, str], SessionSpec]:
    sessions = pd.read_csv(array_dir / "sessions.csv", dtype={"session_key": str}, low_memory=False)
    dataset = np.load(array_dir / "dataset_code.npy", mmap_mode="r")
    evaluation = np.load(array_dir / "evaluation_origin.npy", mmap_mode="r")
    row_sessions = np.load(array_dir / "session_index.npy", mmap_mode="r")
    strict = np.load(array_dir / "temporal_partition_strict.npy", mmap_mode="r")
    unseen = np.load(array_dir / "unseen_user_partition.npy", mmap_mode="r")
    external = np.load(array_dir / "primary_external_partition.npy", mmap_mode="r")
    sport = np.load(array_dir / "sport_code.npy", mmap_mode="r")

    masks = {
        "strict_temporal": (dataset == 0) & (evaluation == 1) & (strict == 4),
        "unseen_user": (dataset == 0) & (evaluation == 1) & (unseen == 4),
        "external": (
            (dataset == 1)
            & (evaluation == 1)
            & (external == 1)
            & np.isin(sport, list(SHARED_EXTERNAL_SPORT_CODES))
        ),
    }
    session_regimes: defaultdict[int, set[str]] = defaultdict(set)
    for regime, mask in masks.items():
        for index in np.unique(row_sessions[mask]):
            session_regimes[int(index)].add(regime)

    indexed = sessions.set_index("session_index", drop=False)
    specs: dict[tuple[str, str], SessionSpec] = {}
    for session_index, regimes in session_regimes.items():
        row = indexed.loc[session_index]
        spec = SessionSpec(
            dataset=str(row.dataset),
            session_key=str(row.session_key),
            session_index=int(row.session_index),
            user_index=int(row.user_index),
            sport_code=int(row.sport_code),
            regimes=tuple(sorted(regimes)),
        )
        specs[(spec.dataset, spec.session_key)] = spec
    return specs


def session_persistence_function(
    series_connection: sqlite3.Connection, dataset: str, session_key: str
):
    grid_start, n_bins, hr_values, hr_mask = load_series(series_connection, dataset, session_key)

    def predict(origin_time: float) -> float:
        values, mask = extract_hr_context(origin_time, grid_start, n_bins, hr_values, hr_mask)
        return persistence_prediction(valid_observations(values, mask))

    return predict


def iter_endomondo(
    source: Path, specs: dict[tuple[str, str], SessionSpec]
) -> Iterable[tuple[SessionSpec, list[float], list[float]]]:
    wanted = {int(key) for dataset, key in specs if dataset == "Endomondo"}
    if not wanted:
        return
    remaining = set(wanted)
    with source.open("rb") as handle:
        for record_index, line in enumerate(handle, start=1):
            if record_index not in remaining:
                continue
            key = str(record_index)
            timestamps = parse_numeric_array(extract_array_bytes(line, "timestamp"), cast=float)
            heart_rate = parse_numeric_array(extract_array_bytes(line, "heart_rate"), cast=float)
            yield specs[("Endomondo", key)], timestamps, heart_rate
            remaining.remove(record_index)
            if not remaining:
                break
    if remaining:
        raise AssertionError(f"missing {len(remaining)} requested Endomondo sessions")


def iter_golden(
    root: Path, specs: dict[tuple[str, str], SessionSpec]
) -> Iterable[tuple[SessionSpec, list[float], list[float]]]:
    golden = sorted(
        (spec for spec in specs.values() if spec.dataset == "GoldenCheetah"),
        key=lambda item: item.session_key,
    )
    for spec in golden:
        timestamps, heart_rate = parse_golden_session(root / Path(spec.session_key))
        yield spec, timestamps, heart_rate


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        stop = min(start + 250, BOOTSTRAP_REPLICATES)
        draw = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = values[draw].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def aggregate(session_rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(session_rows)
    user = (
        frame.groupby(["regime", "horizon_seconds", "cohort", "user_index"], sort=True)
        .agg(user_mae_bpm=("session_mae_bpm", "mean"), sessions=("session_key", "size"), origins=("origins", "sum"))
        .reset_index()
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for (regime, horizon, cohort), group in user.groupby(
        ["regime", "horizon_seconds", "cohort"], sort=True
    ):
        values = group.user_mae_bpm.to_numpy(dtype=np.float64)
        lower, upper = bootstrap_interval(values, rng)
        rows.append(
            {
                "analysis_version": VERSION,
                "regime": regime,
                "horizon_seconds": int(horizon),
                "cohort": cohort,
                "users": int(len(group)),
                "sessions": int(group.sessions.sum()),
                "origins": int(group.origins.sum()),
                "hierarchical_mae_bpm": float(values.mean()),
                "mae_ci_lower_bpm": lower,
                "mae_ci_upper_bpm": upper,
            }
        )
    result = pd.DataFrame(rows)

    result["common_origin_retention"] = np.nan
    result["added_origins_vs_common"] = 0
    result["mae_delta_vs_common_bpm"] = 0.0
    result["delta_ci_lower_bpm"] = 0.0
    result["delta_ci_upper_bpm"] = 0.0
    for regime in result.regime.unique():
        for horizon in HORIZONS:
            select = (result.regime == regime) & (result.horizon_seconds == horizon)
            common_row = result[select & (result.cohort == "common_three_target")].iloc[0]
            specific_index = result.index[select & (result.cohort == "horizon_specific")][0]
            specific_row = result.loc[specific_index]
            result.loc[select, "common_origin_retention"] = (
                int(common_row.origins) / int(specific_row.origins)
            )
            result.loc[specific_index, "added_origins_vs_common"] = int(specific_row.origins) - int(common_row.origins)
            result.loc[specific_index, "mae_delta_vs_common_bpm"] = (
                float(specific_row.hierarchical_mae_bpm) - float(common_row.hierarchical_mae_bpm)
            )

            common_users = user[
                (user.regime == regime)
                & (user.horizon_seconds == horizon)
                & (user.cohort == "common_three_target")
            ][["user_index", "user_mae_bpm"]].rename(columns={"user_mae_bpm": "common"})
            specific_users = user[
                (user.regime == regime)
                & (user.horizon_seconds == horizon)
                & (user.cohort == "horizon_specific")
            ][["user_index", "user_mae_bpm"]].rename(columns={"user_mae_bpm": "specific"})
            paired = common_users.merge(specific_users, on="user_index", validate="one_to_one")
            if len(paired) != int(common_row.users):
                raise AssertionError("paired-user support changed")
            delta = paired.specific.to_numpy(dtype=np.float64) - paired.common.to_numpy(dtype=np.float64)
            lower, upper = bootstrap_interval(delta, rng)
            result.loc[specific_index, "delta_ci_lower_bpm"] = lower
            result.loc[specific_index, "delta_ci_upper_bpm"] = upper
    return result.sort_values(["regime", "horizon_seconds", "cohort"]).reset_index(drop=True)


def validate_common(result: pd.DataFrame) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for regime, expected in EXPECTED_COMMON.items():
        for horizon in HORIZONS:
            row = result[
                (result.regime == regime)
                & (result.horizon_seconds == horizon)
                & (result.cohort == "common_three_target")
            ].iloc[0]
            for field in ("users", "sessions", "origins"):
                observed = int(row[field])
                target = int(expected[field])
                checks.append(
                    {
                        "check": f"{regime}_{horizon}_{field}",
                        "observed": observed,
                        "expected": target,
                        "pass": observed == target,
                    }
                )
            observed_mae = float(row.hierarchical_mae_bpm)
            target_mae = float(expected["mae"][horizon])
            checks.append(
                {
                    "check": f"{regime}_{horizon}_persistence_mae",
                    "observed": observed_mae,
                    "expected": target_mae,
                    "absolute_difference": abs(observed_mae - target_mae),
                    "pass": math.isclose(observed_mae, target_mae, rel_tol=0.0, abs_tol=1e-6),
                }
            )
    return checks


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    specs = load_session_specs(args.array_dir)
    expected_spec_counts = {dataset: sum(spec.dataset == dataset for spec in specs.values()) for dataset in ("Endomondo", "GoldenCheetah")}
    series = sqlite3.connect(args.session_series)
    session_rows: list[dict[str, object]] = []
    errors: list[str] = []
    processed = defaultdict(int)

    streams = (
        iter_endomondo(args.endomondo_hr, specs),
        iter_golden(args.golden_root, specs),
    )
    for stream in streams:
        for spec, timestamps, heart_rate in stream:
            try:
                prediction = session_persistence_function(series, spec.dataset, spec.session_key)
                grouped_errors = build_session_errors(timestamps, heart_rate, prediction)
                for regime_key in spec.regimes:
                    regime = REGIME_LABELS[regime_key]
                    for (horizon, cohort), values in grouped_errors.items():
                        if not values:
                            continue
                        session_rows.append(
                            {
                                "regime": regime,
                                "horizon_seconds": horizon,
                                "cohort": cohort,
                                "user_index": spec.user_index,
                                "session_key": f"{spec.dataset}:{spec.session_key}",
                                "session_mae_bpm": float(np.mean(values)),
                                "origins": len(values),
                            }
                        )
                processed[spec.dataset] += 1
                total = sum(processed.values())
                if total % 1000 == 0:
                    print(f"sessions processed: {total:,}", flush=True)
            except Exception as exc:
                errors.append(f"{spec.dataset}/{spec.session_key}: {type(exc).__name__}: {exc}")
    series.close()
    if errors:
        raise AssertionError(f"session processing failures: {len(errors)}; first={errors[0]}")
    if dict(processed) != expected_spec_counts:
        raise AssertionError(f"session count mismatch: {dict(processed)} != {expected_spec_counts}")

    result = aggregate(session_rows)
    checks = validate_common(result)
    all_pass = all(bool(item["pass"]) for item in checks)
    if not all_pass:
        failed = [item for item in checks if not item["pass"]]
        raise AssertionError(f"common-cohort reproduction failed: {failed[:3]}")

    write_csv_atomic(result, args.output)
    audit: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "analysis_version": VERSION,
        "design": "fixed original evaluation-session sets; horizon-specific target eligibility within session",
        "scope": "parameter-free persistence sensitivity; no model fitting, adaptation, or calibration",
        "context_rule": {
            "past_seconds": CONTEXT_SECONDS,
            "grid_seconds": 10,
            "minimum_observed_bin_fraction": MIN_CONTEXT_BIN_COVERAGE,
            "maximum_context_gap_seconds": MAX_CONTEXT_GAP_SECONDS,
        },
        "target_rule": {
            "horizons_seconds": list(HORIZONS),
            "maximum_interpolation_span_seconds": 30,
            "common_three_target": "all three targets available",
            "horizon_specific": "only the evaluated horizon target required",
        },
        "aggregation": "mean origin absolute error within session, mean session MAE within user, equal-user mean",
        "uncertainty": {
            "method": "nonparametric user bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "paired_delta": True,
        },
        "privacy": "no raw user identifiers or origin-level records are written",
        "fixed_session_counts": expected_spec_counts,
        "reproduction_checks": checks,
        "all_assertions_pass": all_pass,
        "inputs": {
            "array_dir": str(args.array_dir.resolve()),
            "session_series": str(args.session_series.resolve()),
            "endomondo_hr": str(args.endomondo_hr.resolve()),
            "golden_root": str(args.golden_root.resolve()),
        },
        "outputs": {"results": str(args.output.resolve())},
    }
    write_json_atomic(audit, args.audit)
    audit["output_sha256"] = sha256_file(args.output)
    write_json_atomic(audit, args.audit)
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--array-dir", type=Path, default=Path("outputs/features/model_arrays_v0_6_0"))
    result.add_argument("--session-series", type=Path, default=Path("outputs/features/session_series_v0_4_0.sqlite"))
    result.add_argument("--endomondo-hr", type=Path, default=Path("../endomondoHR.json/endomondoHR.json"))
    result.add_argument("--golden-root", type=Path, default=Path("../GoldenCheetah_extracted"))
    result.add_argument("--output", type=Path, default=Path("outputs/results/horizon_specific_eligibility_v0_29_0.csv"))
    result.add_argument("--audit", type=Path, default=Path("outputs/audit/horizon_specific_eligibility_v0_29_0.json"))
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
