from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.21.0"
ARRAY_VERSION = "0.6.0"
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_REPLICATES = 2_000
HORIZONS = (60, 180, 300)
SPORTS = {
    0: "other_unknown",
    1: "outdoor_cycling",
    2: "indoor_virtual_cycling",
    3: "running",
    4: "walking_hiking",
    5: "swimming",
    6: "skiing",
    7: "strength_cross_training",
}
EXPECTED_SUPPORT = {
    "Endomondo_unseen_user_zero_history": {
        "origins": 101_184,
        "sessions": 15_026,
        "users": 105,
    },
    "GoldenCheetah_frozen_external_zero_history": {
        "origins": 531_725,
        "sessions": 31_851,
        "users": 144,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def grid_bin_count(
    session_start_time: np.ndarray | float,
    session_end_time: np.ndarray | float,
    grid_seconds: int = 10,
) -> np.ndarray:
    """Match the right-closed grid construction in build_session_series.py."""
    start = np.asarray(session_start_time, dtype=np.float64)
    end = np.asarray(session_end_time, dtype=np.float64)
    result = np.ceil(end / grid_seconds) - np.ceil(start / grid_seconds) + 1
    if np.any(~np.isfinite(result)) or np.any(result < 1):
        raise ValueError("invalid session grid bounds")
    return result.astype(np.int64)


def origin_to_user_values(
    values: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.Series:
    """Average origins in sessions, then sessions in users."""
    values = np.asarray(values, dtype=np.float64)
    users = np.asarray(users)
    sessions = np.asarray(sessions)
    if not (len(values) == len(users) == len(sessions)):
        raise ValueError("origin arrays differ in length")
    frame = pd.DataFrame(
        {"value": values, "user": users, "session": sessions}
    )
    session_values = frame.groupby(["user", "session"], sort=False)["value"].mean()
    result = session_values.groupby(level="user", sort=False).mean()
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise AssertionError("non-finite hierarchical user values")
    return result


def session_to_user_values(
    values: np.ndarray,
    users: np.ndarray,
) -> pd.Series:
    """Average selected sessions within each user."""
    values = np.asarray(values, dtype=np.float64)
    users = np.asarray(users)
    if len(values) != len(users):
        raise ValueError("session arrays differ in length")
    frame = pd.DataFrame({"value": values, "user": users})
    result = frame.groupby("user", sort=False)["value"].mean()
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise AssertionError("non-finite session-to-user values")
    return result


def bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = values[indices].mean(axis=1)
    if not np.isfinite(result).all():
        raise AssertionError("non-finite bootstrap mean")
    return result


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def compare_user_values(
    *,
    metric: str,
    category: str,
    unit: str,
    internal: pd.Series,
    external: pd.Series,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    aggregation: str = (
        "origin-within-session, session-within-user, equal-user mean"
    ),
) -> dict[str, object]:
    internal_values = internal.to_numpy(dtype=np.float64)
    external_values = external.to_numpy(dtype=np.float64)
    require(len(internal_values) >= 2, f"{metric}: too few internal users")
    require(len(external_values) >= 2, f"{metric}: too few external users")
    generator = np.random.default_rng(stable_seed(bootstrap_seed, metric))
    internal_indices = generator.integers(
        0, len(internal_values), size=(bootstrap_replicates, len(internal_values))
    )
    external_indices = generator.integers(
        0, len(external_values), size=(bootstrap_replicates, len(external_values))
    )
    internal_bootstrap = bootstrap_mean(internal_values, internal_indices)
    external_bootstrap = bootstrap_mean(external_values, external_indices)
    difference_bootstrap = external_bootstrap - internal_bootstrap
    internal_point = float(internal_values.mean())
    external_point = float(external_values.mean())
    internal_low, internal_high = percentile_interval(internal_bootstrap)
    external_low, external_high = percentile_interval(external_bootstrap)
    difference_low, difference_high = percentile_interval(difference_bootstrap)
    if internal_point > 0 and np.all(internal_bootstrap > 0):
        ratio_point = external_point / internal_point
        ratio_low, ratio_high = percentile_interval(
            external_bootstrap / internal_bootstrap
        )
    else:
        ratio_point = ratio_low = ratio_high = math.nan
    return {
        "analysis_version": ANALYSIS_VERSION,
        "metric": metric,
        "category": category,
        "unit": unit,
        "internal_point": internal_point,
        "internal_ci_low": internal_low,
        "internal_ci_high": internal_high,
        "external_point": external_point,
        "external_ci_low": external_low,
        "external_ci_high": external_high,
        "external_minus_internal": external_point - internal_point,
        "difference_ci_low": difference_low,
        "difference_ci_high": difference_high,
        "external_to_internal_ratio": ratio_point,
        "ratio_ci_low": ratio_low,
        "ratio_ci_high": ratio_high,
        "internal_users": int(len(internal_values)),
        "external_users": int(len(external_values)),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_unit": "user, independently within each data source",
        "aggregation": aggregation,
    }


def fixed_history_row() -> dict[str, object]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "metric": "model_user_history_available",
        "category": "deployment",
        "unit": "percent",
        "internal_point": 0.0,
        "internal_ci_low": math.nan,
        "internal_ci_high": math.nan,
        "external_point": 0.0,
        "external_ci_low": math.nan,
        "external_ci_high": math.nan,
        "external_minus_internal": 0.0,
        "difference_ci_low": math.nan,
        "difference_ci_high": math.nan,
        "external_to_internal_ratio": math.nan,
        "ratio_ci_low": math.nan,
        "ratio_ci_high": math.nan,
        "internal_users": EXPECTED_SUPPORT[
            "Endomondo_unseen_user_zero_history"
        ]["users"],
        "external_users": EXPECTED_SUPPORT[
            "GoldenCheetah_frozen_external_zero_history"
        ]["users"],
        "bootstrap_replicates": 0,
        "bootstrap_unit": "not applicable; fixed by deployment protocol",
        "aggregation": (
            "history mask forced to zero for both evaluations; this is model "
            "information availability, not absence of earlier raw workouts"
        ),
    }


def sport_user_session_share(
    session_users: np.ndarray,
    session_sports: np.ndarray,
    sport_code: int,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "user": np.asarray(session_users),
            "is_family": np.asarray(session_sports) == sport_code,
        }
    )
    result = frame.groupby("user", sort=False)["is_family"].mean()
    require(np.all((result >= 0) & (result <= 1)), "invalid sport share")
    return result.astype(np.float64)


def sport_comparison_row(
    *,
    sport_code: int,
    internal_session_users: np.ndarray,
    internal_session_sports: np.ndarray,
    external_session_users: np.ndarray,
    external_session_sports: np.ndarray,
    internal_origin_sports: np.ndarray,
    external_origin_sports: np.ndarray,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    family = SPORTS[sport_code]
    internal_user_share = sport_user_session_share(
        internal_session_users, internal_session_sports, sport_code
    )
    external_user_share = sport_user_session_share(
        external_session_users, external_session_sports, sport_code
    )
    internal_values = internal_user_share.to_numpy()
    external_values = external_user_share.to_numpy()
    generator = np.random.default_rng(stable_seed(bootstrap_seed, f"sport|{family}"))
    internal_indices = generator.integers(
        0, len(internal_values), size=(bootstrap_replicates, len(internal_values))
    )
    external_indices = generator.integers(
        0, len(external_values), size=(bootstrap_replicates, len(external_values))
    )
    internal_bootstrap = 100.0 * bootstrap_mean(internal_values, internal_indices)
    external_bootstrap = 100.0 * bootstrap_mean(external_values, external_indices)
    difference = external_bootstrap - internal_bootstrap
    internal_low, internal_high = percentile_interval(internal_bootstrap)
    external_low, external_high = percentile_interval(external_bootstrap)
    difference_low, difference_high = percentile_interval(difference)
    internal_session_mask = internal_session_sports == sport_code
    external_session_mask = external_session_sports == sport_code
    internal_origin_mask = internal_origin_sports == sport_code
    external_origin_mask = external_origin_sports == sport_code
    return {
        "analysis_version": ANALYSIS_VERSION,
        "sport_code": sport_code,
        "sport_family": family,
        "internal_sessions": int(internal_session_mask.sum()),
        "internal_session_share_percent": 100.0
        * float(internal_session_mask.mean()),
        "internal_origins": int(internal_origin_mask.sum()),
        "internal_origin_share_percent": 100.0 * float(internal_origin_mask.mean()),
        "internal_users_with_family": int(
            np.unique(internal_session_users[internal_session_mask]).size
        ),
        "internal_equal_user_session_share_percent": 100.0
        * float(internal_values.mean()),
        "internal_equal_user_ci_low": internal_low,
        "internal_equal_user_ci_high": internal_high,
        "external_sessions": int(external_session_mask.sum()),
        "external_session_share_percent": 100.0
        * float(external_session_mask.mean()),
        "external_origins": int(external_origin_mask.sum()),
        "external_origin_share_percent": 100.0 * float(external_origin_mask.mean()),
        "external_users_with_family": int(
            np.unique(external_session_users[external_session_mask]).size
        ),
        "external_equal_user_session_share_percent": 100.0
        * float(external_values.mean()),
        "external_equal_user_ci_low": external_low,
        "external_equal_user_ci_high": external_high,
        "external_minus_internal_equal_user_percentage_points": 100.0
        * float(external_values.mean() - internal_values.mean()),
        "difference_ci_low": difference_low,
        "difference_ci_high": difference_high,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_unit": "user, independently within each data source",
        "session_share_estimand": (
            "family fraction within each user's selected sessions, then equal-user mean"
        ),
        "origin_share_inferential": False,
    }


def session_distribution_rows(
    source: str,
    session_frame: pd.DataFrame,
    metrics: list[tuple[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, unit in metrics:
        values = session_frame[metric].to_numpy(dtype=np.float64)
        require(np.isfinite(values).all(), f"{source}/{metric}: non-finite")
        p05, p25, median, p75, p95 = np.quantile(
            values, [0.05, 0.25, 0.50, 0.75, 0.95]
        )
        rows.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "source": source,
                "metric": metric,
                "unit": unit,
                "sessions": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "p05": float(p05),
                "p25": float(p25),
                "median": float(median),
                "p75": float(p75),
                "p95": float(p95),
                "inferential": False,
                "note": (
                    "unweighted selected-session distribution; sessions are not "
                    "treated as independent inferential units"
                ),
            }
        )
    return rows


def load_selected_session_frame(
    *,
    sessions_csv: Path,
    endomondo_quality: Path,
    golden_quality: Path,
    selected_session_ids: np.ndarray,
    selected_origin_session_ids: np.ndarray,
) -> pd.DataFrame:
    sessions = pd.read_csv(
        sessions_csv,
        dtype={"session_key": "string", "dataset": "string"},
    )
    sessions = sessions.loc[sessions["session_index"].isin(selected_session_ids)].copy()
    require(
        len(sessions) == len(selected_session_ids),
        "selected session metadata is incomplete",
    )
    require(sessions["session_index"].is_unique, "duplicate session index")
    endomondo = pd.read_csv(
        endomondo_quality,
        usecols=[
            "record_index",
            "valid_hr_coverage",
            "median_positive_gap_seconds",
            "max_positive_gap_seconds",
        ],
        dtype={"record_index": "string"},
    ).rename(columns={"record_index": "session_key"})
    endomondo["dataset"] = "Endomondo"
    golden = pd.read_csv(
        golden_quality,
        usecols=[
            "csv_relative_path",
            "valid_hr_coverage",
            "median_positive_gap_seconds",
            "max_positive_gap_seconds",
        ],
        dtype={"csv_relative_path": "string"},
    ).rename(columns={"csv_relative_path": "session_key"})
    golden["dataset"] = "GoldenCheetah"
    quality = pd.concat([endomondo, golden], ignore_index=True)
    require(
        not quality.duplicated(["dataset", "session_key"]).any(),
        "duplicate quality key",
    )
    sessions = sessions.merge(
        quality,
        on=["dataset", "session_key"],
        how="left",
        validate="one_to_one",
    )
    required_quality = [
        "valid_hr_coverage",
        "median_positive_gap_seconds",
        "max_positive_gap_seconds",
    ]
    require(
        not sessions[required_quality].isna().any().any(),
        "quality join has missing values",
    )
    origin_counts = pd.Series(selected_origin_session_ids).value_counts(sort=False)
    sessions["evaluation_origins_per_session"] = sessions["session_index"].map(
        origin_counts
    )
    require(
        not sessions["evaluation_origins_per_session"].isna().any(),
        "origin counts missing for selected sessions",
    )
    sessions["grid_bins"] = grid_bin_count(
        sessions["session_start_time"].to_numpy(),
        sessions["session_end_time"].to_numpy(),
    )
    sessions["duration_minutes"] = sessions["duration_seconds"] / 60.0
    sessions["raw_valid_hr_coverage_percent"] = (
        100.0 * sessions["valid_hr_coverage"]
    )
    sessions["causal_grid_hr_observed_percent"] = (
        100.0 * sessions["hr_observed_bins"] / sessions["grid_bins"]
    )
    sessions["causal_grid_speed_observed_percent"] = (
        100.0 * sessions["speed_observed_bins"] / sessions["grid_bins"]
    )
    sessions["causal_grid_altitude_observed_percent"] = (
        100.0 * sessions["altitude_observed_bins"] / sessions["grid_bins"]
    )
    coverage_columns = [
        "raw_valid_hr_coverage_percent",
        "causal_grid_hr_observed_percent",
        "causal_grid_speed_observed_percent",
        "causal_grid_altitude_observed_percent",
    ]
    for column in coverage_columns:
        require(
            bool(((sessions[column] >= 0) & (sessions[column] <= 100)).all()),
            f"{column}: invalid coverage",
        )
    return sessions.sort_values("session_index").reset_index(drop=True)


def build_origin_user_metrics(
    *,
    row_indices: np.ndarray,
    values: np.memmap,
    masks: np.memmap,
    targets: np.memmap,
    users: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, pd.Series]:
    local_masks = np.asarray(masks[row_indices], dtype=np.float64)
    require(local_masks.shape[1:] == (30, 3), "unexpected context-mask shape")
    require(
        bool(np.logical_or(local_masks == 0, local_masks == 1).all()),
        "context masks are not binary",
    )
    hr_values = np.asarray(values[row_indices, :, 0], dtype=np.float64)
    hr_masks = local_masks[:, :, 0]
    observed_hr_bins = hr_masks.sum(axis=1)
    require(bool((observed_hr_bins > 0).all()), "origin with no observed context HR")
    context_hr = (hr_values * hr_masks).sum(axis=1) / observed_hr_bins
    local_targets = np.asarray(targets[row_indices], dtype=np.float64)
    require(local_targets.shape[1] == 3, "unexpected target shape")
    require(np.isfinite(local_targets).all(), "non-finite target")
    result = {
        "context_hr_bpm": origin_to_user_values(
            context_hr, users[row_indices], sessions[row_indices]
        ),
        "context_hr_missing_percent": origin_to_user_values(
            100.0 * (1.0 - local_masks[:, :, 0].mean(axis=1)),
            users[row_indices],
            sessions[row_indices],
        ),
        "context_speed_missing_percent": origin_to_user_values(
            100.0 * (1.0 - local_masks[:, :, 1].mean(axis=1)),
            users[row_indices],
            sessions[row_indices],
        ),
        "context_altitude_missing_percent": origin_to_user_values(
            100.0 * (1.0 - local_masks[:, :, 2].mean(axis=1)),
            users[row_indices],
            sessions[row_indices],
        ),
    }
    for horizon_index, horizon in enumerate(HORIZONS):
        result[f"target_hr_{horizon}_bpm"] = origin_to_user_values(
            local_targets[:, horizon_index],
            users[row_indices],
            sessions[row_indices],
        )
    return result


def markdown_report(
    characteristic_rows: list[dict[str, object]],
    sport_rows: list[dict[str, object]],
    support: dict[str, dict[str, int]],
    bootstrap_replicates: int,
) -> str:
    characteristic = {str(row["metric"]): row for row in characteristic_rows}

    def fmt(metric: str, column: str, digits: int = 1) -> str:
        return f"{float(characteristic[metric][column]):.{digits}f}"

    sport_sorted = sorted(
        sport_rows,
        key=lambda row: float(row["external_session_share_percent"]),
        reverse=True,
    )
    sport_lines = []
    for row in sport_sorted:
        if int(row["internal_sessions"]) == 0 and int(row["external_sessions"]) == 0:
            continue
        sport_lines.append(
            "| {family} | {internal:.1f} | {external:.1f} | {internal_origin:.1f} | {external_origin:.1f} |".format(
                family=row["sport_family"],
                internal=float(row["internal_session_share_percent"]),
                external=float(row["external_session_share_percent"]),
                internal_origin=float(row["internal_origin_share_percent"]),
                external_origin=float(row["external_origin_share_percent"]),
            )
        )
    return f"""# Frozen internal--external source-shift characterization (v{ANALYSIS_VERSION})

## Estimands and support

This audit compares the Endomondo unseen-user **forced-zero-history** test set with the frozen GoldenCheetah external **zero-history** evaluation. It characterizes the evaluated data distributions; it does not attribute differences causally to platform, device, sport, or population.

| Evaluation | Users | Sessions | 300-s origins |
|---|---:|---:|---:|
| Endomondo unseen-user zero-history | {support['Endomondo_unseen_user_zero_history']['users']:,} | {support['Endomondo_unseen_user_zero_history']['sessions']:,} | {support['Endomondo_unseen_user_zero_history']['origins']:,} |
| GoldenCheetah frozen external zero-history | {support['GoldenCheetah_frozen_external_zero_history']['users']:,} | {support['GoldenCheetah_frozen_external_zero_history']['sessions']:,} | {support['GoldenCheetah_frozen_external_zero_history']['origins']:,} |

Origin-based numeric comparisons average origins within sessions, sessions within users, and then weight users equally. Session-based metrics average selected sessions within users and then weight users equally. Confidence intervals are percentile intervals from {bootstrap_replicates:,} independent user-cluster bootstrap replicates within each source. The separate session-distribution file is descriptive and deliberately makes no session-level independence claim.

## Main shifts

- Mean selected-session duration was {fmt('session_duration_minutes', 'internal_point')} min internally and {fmt('session_duration_minutes', 'external_point')} min externally (external minus internal {fmt('session_duration_minutes', 'external_minus_internal')} min; 95% CI {fmt('session_duration_minutes', 'difference_ci_low')} to {fmt('session_duration_minutes', 'difference_ci_high')}).
- The raw median positive sampling gap averaged {fmt('raw_median_sampling_gap_seconds', 'internal_point')} s internally and {fmt('raw_median_sampling_gap_seconds', 'external_point')} s externally. These are source-format support descriptors, not model inputs.
- Mean context HR was {fmt('context_hr_bpm', 'internal_point')} versus {fmt('context_hr_bpm', 'external_point')} bpm. Target HR differences (external minus internal) were {fmt('target_hr_60_bpm', 'external_minus_internal')} bpm at +1 min, {fmt('target_hr_180_bpm', 'external_minus_internal')} bpm at +3 min, and {fmt('target_hr_300_bpm', 'external_minus_internal')} bpm at +5 min.
- Context missingness differed most for speed ({fmt('context_speed_missing_percent', 'internal_point')}% internal; {fmt('context_speed_missing_percent', 'external_point')}% external) and altitude ({fmt('context_altitude_missing_percent', 'internal_point')}% internal; {fmt('context_altitude_missing_percent', 'external_point')}% external). Missingness is retained as an observed mask and was not repaired using future samples.
- The model-level user-history input was forced to zero in both comparisons. This protocol setting does not assert that earlier raw workouts were absent for every user.

## Sport composition

Natural source composition is reported at both the selected-session and overlapping-origin levels. Origin shares are descriptive only.

| Sport family | Internal sessions (%) | External sessions (%) | Internal origins (%) | External origins (%) |
|---|---:|---:|---:|---:|
{chr(10).join(sport_lines)}

The external cohort was restricted prospectively to the three sport families supported by the frozen primary external model scope; the internal unseen-user test also contains small walking/hiking, skiing, and strength/cross-training components. Consequently, natural-mix source differences combine source, sensor, user, session, and sport-composition shifts. The separate sport-standardization analysis should be used when asking how much broad three-family composition explains performance differences.

## Interpretation limits

- Users are the inferential resampling unit; repeated forecast origins are not treated as independent.
- Confidence intervals quantify finite-sample user heterogeneity within these two selected cohorts, not population-representative sampling uncertainty.
- Raw gap and coverage fields come from frozen quality manifests; 10-s grid support and context missingness come from frozen model arrays.
- This characterization is descriptive. It cannot isolate device effects, platform effects, physiology, training status, or demographic selection.
"""


def analyze(args: argparse.Namespace) -> dict[str, object]:
    array_paths = {
        "dataset": args.array_dir / "dataset_code.npy",
        "evaluation": args.array_dir / "evaluation_origin.npy",
        "unseen": args.array_dir / "unseen_user_partition.npy",
        "external": args.array_dir / "primary_external_partition.npy",
        "users": args.array_dir / "user_index.npy",
        "sessions": args.array_dir / "session_index.npy",
        "sport": args.array_dir / "sport_code.npy",
        "values": args.array_dir / "sequence_values.npy",
        "masks": args.array_dir / "sequence_masks.npy",
        "targets": args.array_dir / "targets.npy",
    }
    for path in [
        *array_paths.values(),
        args.array_dir / "sessions.csv",
        args.array_dir / "metadata.json",
        args.endomondo_quality,
        args.golden_quality,
    ]:
        require(path.exists(), f"missing input: {path}")
    metadata = json.loads(
        (args.array_dir / "metadata.json").read_text(encoding="utf-8")
    )
    require(metadata.get("array_version") == ARRAY_VERSION, "array version mismatch")
    require(metadata.get("all_assertions_pass") is True, "array audit did not pass")
    arrays = {
        name: np.load(path, mmap_mode="r") for name, path in array_paths.items()
    }
    lengths = {name: int(len(value)) for name, value in arrays.items()}
    require(len(set(lengths.values())) == 1, f"array length mismatch: {lengths}")
    internal_mask = (
        (arrays["dataset"] == 0)
        & (arrays["evaluation"] == 1)
        & (arrays["unseen"] == PARTITION_TEST)
    )
    external_mask = (
        (arrays["dataset"] == 1) & (arrays["external"] == EXTERNAL_FROZEN)
    )
    require(not np.any(internal_mask & external_mask), "source selections overlap")
    row_indices = {
        "internal": np.flatnonzero(internal_mask),
        "external": np.flatnonzero(external_mask),
    }
    support: dict[str, dict[str, int]] = {}
    for source, label in [
        ("internal", "Endomondo_unseen_user_zero_history"),
        ("external", "GoldenCheetah_frozen_external_zero_history"),
    ]:
        rows = row_indices[source]
        observed = {
            "origins": int(len(rows)),
            "sessions": int(np.unique(arrays["sessions"][rows]).size),
            "users": int(np.unique(arrays["users"][rows]).size),
        }
        require(observed == EXPECTED_SUPPORT[label], f"{label}: {observed}")
        support[label] = observed
    all_rows = np.concatenate([row_indices["internal"], row_indices["external"]])
    selected_sessions = np.unique(arrays["sessions"][all_rows])
    session_frame = load_selected_session_frame(
        sessions_csv=args.array_dir / "sessions.csv",
        endomondo_quality=args.endomondo_quality,
        golden_quality=args.golden_quality,
        selected_session_ids=selected_sessions,
        selected_origin_session_ids=np.asarray(arrays["sessions"][all_rows]),
    )
    internal_sessions = session_frame.loc[session_frame["dataset"] == "Endomondo"].copy()
    external_sessions = session_frame.loc[
        session_frame["dataset"] == "GoldenCheetah"
    ].copy()
    require(len(internal_sessions) == 15_026, "internal session frame support")
    require(len(external_sessions) == 31_851, "external session frame support")
    require(internal_sessions["user_index"].nunique() == 105, "internal users")
    require(external_sessions["user_index"].nunique() == 144, "external users")
    session_user_lookup = np.full(
        int(session_frame["session_index"].max()) + 1, -1, dtype=np.int32
    )
    session_sport_lookup = np.full_like(session_user_lookup, -1)
    lookup_indices = session_frame["session_index"].to_numpy(dtype=np.int64)
    session_user_lookup[lookup_indices] = session_frame["user_index"].to_numpy(
        dtype=np.int32
    )
    session_sport_lookup[lookup_indices] = session_frame["sport_code"].to_numpy(
        dtype=np.int32
    )
    selected_origin_sessions = np.asarray(arrays["sessions"][all_rows])
    require(
        np.array_equal(
            session_user_lookup[selected_origin_sessions],
            np.asarray(arrays["users"][all_rows]),
        ),
        "origin/session user mapping mismatch",
    )
    require(
        np.array_equal(
            session_sport_lookup[selected_origin_sessions],
            np.asarray(arrays["sport"][all_rows]),
        ),
        "origin/session sport mapping mismatch",
    )

    origin_metrics = {
        source: build_origin_user_metrics(
            row_indices=rows,
            values=arrays["values"],
            masks=arrays["masks"],
            targets=arrays["targets"],
            users=arrays["users"],
            sessions=arrays["sessions"],
        )
        for source, rows in row_indices.items()
    }
    metric_specifications = [
        ("context_hr_bpm", "heart_rate", "bpm"),
        ("target_hr_60_bpm", "heart_rate", "bpm"),
        ("target_hr_180_bpm", "heart_rate", "bpm"),
        ("target_hr_300_bpm", "heart_rate", "bpm"),
        ("context_hr_missing_percent", "context_missingness", "percent"),
        ("context_speed_missing_percent", "context_missingness", "percent"),
        ("context_altitude_missing_percent", "context_missingness", "percent"),
    ]
    characteristic_rows: list[dict[str, object]] = []
    for metric, category, unit in metric_specifications:
        characteristic_rows.append(
            compare_user_values(
                metric=metric,
                category=category,
                unit=unit,
                internal=origin_metrics["internal"][metric],
                external=origin_metrics["external"][metric],
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
    session_metric_specifications = [
        (
            "duration_minutes",
            "session_duration_minutes",
            "session_support",
            "minutes",
        ),
        (
            "raw_valid_hr_coverage_percent",
            "raw_valid_hr_coverage_percent",
            "raw_support",
            "percent",
        ),
        (
            "median_positive_gap_seconds",
            "raw_median_sampling_gap_seconds",
            "raw_support",
            "seconds",
        ),
        (
            "max_positive_gap_seconds",
            "raw_max_sampling_gap_seconds",
            "raw_support",
            "seconds",
        ),
        (
            "causal_grid_hr_observed_percent",
            "causal_grid_hr_observed_percent",
            "causal_grid_support",
            "percent",
        ),
        (
            "causal_grid_speed_observed_percent",
            "causal_grid_speed_observed_percent",
            "causal_grid_support",
            "percent",
        ),
        (
            "causal_grid_altitude_observed_percent",
            "causal_grid_altitude_observed_percent",
            "causal_grid_support",
            "percent",
        ),
        (
            "evaluation_origins_per_session",
            "evaluation_origins_per_session",
            "evaluation_support",
            "origins/session",
        ),
    ]
    for session_column, metric, category, unit in session_metric_specifications:
        characteristic_rows.append(
            compare_user_values(
                metric=metric,
                category=category,
                unit=unit,
                internal=session_to_user_values(
                    internal_sessions[session_column].to_numpy(),
                    internal_sessions["user_index"].to_numpy(),
                ),
                external=session_to_user_values(
                    external_sessions[session_column].to_numpy(),
                    external_sessions["user_index"].to_numpy(),
                ),
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                aggregation="session-within-user, equal-user mean",
            )
        )
    characteristic_rows.append(fixed_history_row())

    internal_rows = row_indices["internal"]
    external_rows = row_indices["external"]
    present_sports = sorted(
        set(np.unique(arrays["sport"][internal_rows]).tolist())
        | set(np.unique(arrays["sport"][external_rows]).tolist())
    )
    require(all(code in SPORTS for code in present_sports), "unknown sport code")
    sport_rows = [
        sport_comparison_row(
            sport_code=int(code),
            internal_session_users=internal_sessions["user_index"].to_numpy(),
            internal_session_sports=internal_sessions["sport_code"].to_numpy(),
            external_session_users=external_sessions["user_index"].to_numpy(),
            external_session_sports=external_sessions["sport_code"].to_numpy(),
            internal_origin_sports=np.asarray(arrays["sport"][internal_rows]),
            external_origin_sports=np.asarray(arrays["sport"][external_rows]),
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        for code in present_sports
    ]
    require(
        np.isclose(
            sum(float(row["internal_session_share_percent"]) for row in sport_rows),
            100.0,
        ),
        "internal session sport shares do not sum to 100",
    )
    require(
        np.isclose(
            sum(float(row["external_session_share_percent"]) for row in sport_rows),
            100.0,
        ),
        "external session sport shares do not sum to 100",
    )
    distribution_metrics = [
        ("duration_minutes", "minutes"),
        ("valid_hr_coverage", "proportion"),
        ("median_positive_gap_seconds", "seconds"),
        ("max_positive_gap_seconds", "seconds"),
        ("causal_grid_hr_observed_percent", "percent"),
        ("causal_grid_speed_observed_percent", "percent"),
        ("causal_grid_altitude_observed_percent", "percent"),
        ("evaluation_origins_per_session", "origins/session"),
    ]
    distribution_rows = session_distribution_rows(
        "Endomondo_unseen_user_test", internal_sessions, distribution_metrics
    ) + session_distribution_rows(
        "GoldenCheetah_frozen_external", external_sessions, distribution_metrics
    )

    atomic_csv(args.characteristics_output, characteristic_rows)
    atomic_csv(args.sport_output, sport_rows)
    atomic_csv(args.session_distribution_output, distribution_rows)
    report = markdown_report(
        characteristic_rows,
        sport_rows,
        support,
        args.bootstrap_replicates,
    )
    atomic_text(args.markdown_output, report)
    inputs = [
        *array_paths.values(),
        args.array_dir / "sessions.csv",
        args.array_dir / "metadata.json",
        args.endomondo_quality,
        args.golden_quality,
    ]
    outputs = [
        args.characteristics_output,
        args.sport_output,
        args.session_distribution_output,
        args.markdown_output,
    ]
    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "array_version": ARRAY_VERSION,
        "analysis_purpose": (
            "descriptive characterization of the frozen internal unseen-user "
            "zero-history and GoldenCheetah external zero-history evaluation distributions"
        ),
        "selection": {
            "internal": (
                "dataset_code=Endomondo, evaluation_origin=1, "
                "unseen_user_partition=test"
            ),
            "external": (
                "dataset_code=GoldenCheetah, primary_external_partition="
                "frozen_external_test"
            ),
            "history_input": (
                "forced zero in both deployment evaluations; does not imply raw "
                "historical workouts are absent"
            ),
        },
        "support": support,
        "bootstrap": {
            "seed": args.bootstrap_seed,
            "replicates": args.bootstrap_replicates,
            "interval": "equal-tail percentile 2.5%--97.5%",
            "unit": "user sampled independently with replacement within each source",
            "aggregation": (
                "origin mean within session, session mean within user, equal-user mean"
            ),
        },
        "descriptive_distributions": {
            "unit": "selected session",
            "inferential": False,
            "reason": "sessions from the same user are not independent",
        },
        "sport_composition": {
            "session_share": "natural selected-session composition",
            "origin_share": (
                "natural 300-s-origin composition; descriptive because origins overlap"
            ),
            "equal_user_session_share": (
                "within-user session fraction, then equal-user mean with user bootstrap"
            ),
        },
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "output_sha256": {str(path): sha256_file(path) for path in outputs},
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "assertions": {
            "frozen_support_exact": True,
            "source_selections_disjoint": True,
            "array_lengths_identical": True,
            "array_build_assertions_passed": True,
            "session_quality_join_complete": True,
            "origin_session_user_mapping_exact": True,
            "origin_session_sport_mapping_exact": True,
            "context_masks_binary": True,
            "targets_finite": True,
            "sport_session_shares_sum_to_100_percent": True,
            "all_reported_numeric_metrics_finite_except_protocol_undefined_ratios": True,
        },
        "limitations": [
            "This is descriptive and does not identify a causal platform or device effect.",
            "Source shifts combine user, sport, device, session, and data-format differences.",
            "The external set is prospectively restricted to three supported sport families.",
            "Bootstrap intervals describe user heterogeneity in selected cohorts, not population representativeness.",
            "Raw sampling-gap fields and 10-s causal-grid support measure different stages of data processing.",
        ],
        "all_assertions_pass": True,
    }
    atomic_json(args.audit_output, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize frozen internal--external data-source shift."
    )
    parser.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    parser.add_argument(
        "--endomondo-quality",
        type=Path,
        default=Path("outputs/manifests/endomondo_session_quality_v0_2_0.csv"),
    )
    parser.add_argument(
        "--golden-quality",
        type=Path,
        default=Path("outputs/manifests/goldencheetah_session_quality_v0_2_0.csv"),
    )
    parser.add_argument(
        "--characteristics-output",
        type=Path,
        default=Path(
            "outputs/results/source_shift_characterization_v0_21_0.csv"
        ),
    )
    parser.add_argument(
        "--sport-output",
        type=Path,
        default=Path(
            "outputs/results/source_shift_sport_composition_v0_21_0.csv"
        ),
    )
    parser.add_argument(
        "--session-distribution-output",
        type=Path,
        default=Path(
            "outputs/results/source_shift_session_distributions_v0_21_0.csv"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("outputs/audit/source_shift_characterization_v0_21_0.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("protocol/SOURCE_SHIFT_CHARACTERIZATION_V0_21_0.md"),
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1_000:
        parser.error("bootstrap-replicates must be at least 1,000")
    return args


def main() -> None:
    audit = analyze(parse_args())
    print(json.dumps(audit["support"], ensure_ascii=False, indent=2))
    print("all assertions passed")


if __name__ == "__main__":
    main()
