from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from build_session_series import GRID_SECONDS, decompress_float32, decompress_uint8


BASELINE_VERSION = "0.5.0"
CONTEXT_BINS = 30
HORIZONS = (60, 180, 300)
EWMA_CANDIDATES = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
HR_MIN = 30.0
HR_MAX = 240.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def extract_hr_context(
    origin_time: float,
    grid_start_bin: int,
    n_bins: int,
    hr_values: Sequence[float],
    hr_mask: Sequence[int],
) -> tuple[list[float], list[int]]:
    origin_bin_float = origin_time / GRID_SECONDS
    origin_bin = int(round(origin_bin_float))
    if not math.isclose(origin_bin_float, origin_bin, abs_tol=1e-7):
        raise ValueError("forecast origin is not aligned to the 10-second grid")
    end_position = origin_bin - grid_start_bin
    start_position = end_position - CONTEXT_BINS + 1
    if start_position < 0 or end_position >= n_bins:
        raise IndexError("forecast context lies outside the cached session series")
    values = list(hr_values[start_position : end_position + 1])
    mask = list(hr_mask[start_position : end_position + 1])
    if len(values) != CONTEXT_BINS or len(mask) != CONTEXT_BINS:
        raise AssertionError("forecast context has the wrong number of bins")
    return values, mask


def valid_observations(
    values: Sequence[float], mask: Sequence[int]
) -> list[tuple[float, float]]:
    observations = []
    for index, (value, observed) in enumerate(zip(values, mask)):
        if observed:
            time_seconds = (index - (CONTEXT_BINS - 1)) * GRID_SECONDS
            observations.append((float(time_seconds), float(value)))
    if not observations:
        raise ValueError("context contains no observed heart-rate values")
    return observations


def persistence_prediction(observations: Sequence[tuple[float, float]]) -> float:
    return float(observations[-1][1])


def ewma_prediction(
    observations: Sequence[tuple[float, float]], alpha: float
) -> float:
    if not 0 < alpha <= 1:
        raise ValueError("EWMA alpha must be in (0, 1]")
    estimate = float(observations[0][1])
    for _, value in observations[1:]:
        estimate = alpha * float(value) + (1.0 - alpha) * estimate
    return min(HR_MAX, max(HR_MIN, estimate))


def linear_trend_predictions(
    observations: Sequence[tuple[float, float]],
    horizons: Sequence[int] = HORIZONS,
) -> tuple[float, ...]:
    n = len(observations)
    if n < 2:
        value = persistence_prediction(observations)
        return tuple(value for _ in horizons)
    sum_x = sum(item[0] for item in observations)
    sum_y = sum(item[1] for item in observations)
    sum_xx = sum(item[0] * item[0] for item in observations)
    sum_xy = sum(item[0] * item[1] for item in observations)
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        value = persistence_prediction(observations)
        return tuple(value for _ in horizons)
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    mean_x = sum_x / n
    mean_y = sum_y / n
    return tuple(
        min(HR_MAX, max(HR_MIN, mean_y + slope * (horizon - mean_x)))
        for horizon in horizons
    )


def load_series(
    connection: sqlite3.Connection, dataset: str, session_key: str
) -> tuple[int, int, Sequence[float], Sequence[int]]:
    row = connection.execute(
        """
        SELECT grid_start_bin, n_bins, hr_values_zlib, hr_mask_zlib
        FROM session_series WHERE dataset = ? AND session_key = ?
        """,
        (dataset, session_key),
    ).fetchone()
    if row is None:
        raise KeyError(f"missing feature series: {dataset}/{session_key}")
    grid_start_bin, n_bins, values_blob, mask_blob = row
    return (
        int(grid_start_bin),
        int(n_bins),
        decompress_float32(values_blob, int(n_bins)),
        decompress_uint8(mask_blob, int(n_bins)),
    )


def grouped_origin_rows(
    cursor: Iterable[sqlite3.Row],
) -> Iterator[tuple[tuple[str, str], Iterator[sqlite3.Row]]]:
    return itertools.groupby(cursor, key=lambda row: (row["dataset"], row["session_key"]))


def tune_ewma(
    origins_path: Path, features_path: Path, output_path: Path
) -> dict[str, object]:
    origins = sqlite3.connect(origins_path)
    origins.row_factory = sqlite3.Row
    features = sqlite3.connect(features_path)
    query = origins.execute(
        """
        SELECT dataset, session_key, user_id, origin_time, context_valid_bins,
               target_hr_60, target_hr_180, target_hr_300
        FROM origins
        WHERE dataset = 'Endomondo'
          AND evaluation_origin = 1
          AND unseen_user_partition = 'validation'
        ORDER BY dataset, session_key, origin_time
        """
    )
    user_session_metric: defaultdict[tuple[float, int, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    mask_mismatches = 0
    origins_used = 0
    sessions_used = 0
    for (dataset, session_key), rows_iterator in grouped_origin_rows(query):
        rows = list(rows_iterator)
        grid_start, n_bins, hr_values, hr_mask = load_series(
            features, dataset, session_key
        )
        session_errors: defaultdict[tuple[float, int], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        user_id = str(rows[0]["user_id"])
        for row in rows:
            values, mask = extract_hr_context(
                float(row["origin_time"]), grid_start, n_bins, hr_values, hr_mask
            )
            if sum(mask) != int(row["context_valid_bins"]):
                mask_mismatches += 1
            observations = valid_observations(values, mask)
            targets = {
                60: float(row["target_hr_60"]),
                180: float(row["target_hr_180"]),
                300: float(row["target_hr_300"]),
            }
            for alpha in EWMA_CANDIDATES:
                prediction = ewma_prediction(observations, alpha)
                for horizon in HORIZONS:
                    accumulator = session_errors[(alpha, horizon)]
                    accumulator[0] += abs(prediction - targets[horizon])
                    accumulator[1] += 1
            origins_used += 1
        for (alpha, horizon), (absolute_error, count) in session_errors.items():
            accumulator = user_session_metric[(alpha, horizon, user_id)]
            accumulator[0] += absolute_error / count
            accumulator[1] += 1
        sessions_used += 1
    horizon_scores: dict[float, dict[int, float]] = {}
    aggregate_scores: dict[float, float] = {}
    for alpha in EWMA_CANDIDATES:
        horizon_scores[alpha] = {}
        for horizon in HORIZONS:
            user_values = [
                total / sessions
                for (candidate, candidate_horizon, _), (total, sessions) in user_session_metric.items()
                if candidate == alpha and candidate_horizon == horizon
            ]
            horizon_scores[alpha][horizon] = sum(user_values) / len(user_values)
        aggregate_scores[alpha] = sum(horizon_scores[alpha].values()) / len(HORIZONS)
    selected_alpha = min(EWMA_CANDIDATES, key=lambda value: (aggregate_scores[value], value))
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "baseline_version": BASELINE_VERSION,
        "selection_dataset": "Endomondo",
        "selection_partition": "unseen_user_validation",
        "evaluation_origin_stride_seconds": 300,
        "selection_metric": "mean of 1/3/5-min user-session-hierarchical MAE",
        "candidate_alphas": list(EWMA_CANDIDATES),
        "horizon_mae_by_alpha": {
            str(alpha): {str(horizon): score for horizon, score in scores.items()}
            for alpha, scores in horizon_scores.items()
        },
        "aggregate_mae_by_alpha": {
            str(alpha): score for alpha, score in aggregate_scores.items()
        },
        "selected_alpha": selected_alpha,
        "origins_used": origins_used,
        "sessions_used": sessions_used,
        "users_used": len({key[2] for key in user_session_metric}),
        "context_mask_mismatches": mask_mismatches,
        "all_assertions_pass": mask_mismatches == 0,
    }
    origins.close()
    features.close()
    if not payload["all_assertions_pass"]:
        raise AssertionError("feature masks disagree with the locked forecast-origin index")
    atomic_json(output_path, payload)
    return payload


def connect_prediction_database(path: Path, ewma_alpha: float) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-200000")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            dataset TEXT NOT NULL,
            session_key TEXT NOT NULL,
            origin_time REAL NOT NULL,
            persistence REAL NOT NULL,
            ewma REAL NOT NULL,
            linear_trend_60 REAL NOT NULL,
            linear_trend_180 REAL NOT NULL,
            linear_trend_300 REAL NOT NULL,
            PRIMARY KEY (dataset, session_key, origin_time)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_sessions (
            dataset TEXT NOT NULL,
            session_key TEXT NOT NULL,
            status TEXT NOT NULL,
            origins_added INTEGER NOT NULL,
            processed_at_utc TEXT NOT NULL,
            PRIMARY KEY (dataset, session_key)
        ) WITHOUT ROWID
        """
    )
    metadata = {
        "baseline_version": BASELINE_VERSION,
        "ewma_alpha": str(ewma_alpha),
        "evaluation_origin_stride_seconds": "300",
    }
    for key, value in metadata.items():
        existing = connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if existing is not None and existing[0] != value:
            raise ValueError(f"prediction database metadata mismatch for {key}")
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, value)
        )
    connection.commit()
    return connection


def build_predictions(
    origins_path: Path,
    features_path: Path,
    prediction_path: Path,
    ewma_alpha: float,
) -> dict[str, object]:
    origins = sqlite3.connect(origins_path)
    origins.row_factory = sqlite3.Row
    features = sqlite3.connect(features_path)
    output = connect_prediction_database(prediction_path, ewma_alpha)
    completed = {
        (row[0], row[1])
        for row in output.execute(
            "SELECT dataset, session_key FROM processed_sessions WHERE status = 'processed'"
        )
    }
    query = origins.execute(
        """
        SELECT dataset, session_key, origin_time, context_valid_bins
        FROM origins WHERE evaluation_origin = 1
        ORDER BY dataset, session_key, origin_time
        """
    )
    origins_added = 0
    sessions_added = 0
    mask_mismatches = 0
    missing_series = 0
    output.execute("BEGIN")
    for (dataset, session_key), rows_iterator in grouped_origin_rows(query):
        rows = list(rows_iterator)
        if (dataset, session_key) in completed:
            continue
        try:
            grid_start, n_bins, hr_values, hr_mask = load_series(
                features, dataset, session_key
            )
        except KeyError:
            missing_series += 1
            continue
        predictions = []
        for row in rows:
            values, mask = extract_hr_context(
                float(row["origin_time"]), grid_start, n_bins, hr_values, hr_mask
            )
            if sum(mask) != int(row["context_valid_bins"]):
                mask_mismatches += 1
            observations = valid_observations(values, mask)
            persistence = persistence_prediction(observations)
            ewma = ewma_prediction(observations, ewma_alpha)
            trend = linear_trend_predictions(observations)
            predictions.append(
                (
                    dataset,
                    session_key,
                    float(row["origin_time"]),
                    persistence,
                    ewma,
                    trend[0],
                    trend[1],
                    trend[2],
                )
            )
        output.executemany(
            """
            INSERT OR REPLACE INTO predictions(
                dataset, session_key, origin_time, persistence, ewma,
                linear_trend_60, linear_trend_180, linear_trend_300
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            predictions,
        )
        output.execute(
            """
            INSERT OR REPLACE INTO processed_sessions(
                dataset, session_key, status, origins_added, processed_at_utc
            ) VALUES (?, ?, 'processed', ?, ?)
            """,
            (dataset, session_key, len(predictions), utc_now()),
        )
        origins_added += len(predictions)
        sessions_added += 1
        if sessions_added % 500 == 0:
            output.commit()
            output.execute("BEGIN")
            print(
                f"Baseline sessions processed: {sessions_added:,}; predictions added: {origins_added:,}",
                flush=True,
            )
    output.commit()
    expected = origins.execute(
        "SELECT COUNT(*) FROM origins WHERE evaluation_origin = 1"
    ).fetchone()[0]
    actual = output.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    duplicates = output.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT dataset, session_key, origin_time, COUNT(*) AS n
            FROM predictions GROUP BY dataset, session_key, origin_time HAVING n > 1
        )
        """
    ).fetchone()[0]
    out_of_range = output.execute(
        """
        SELECT COUNT(*) FROM predictions
        WHERE persistence NOT BETWEEN ? AND ? OR ewma NOT BETWEEN ? AND ?
           OR linear_trend_60 NOT BETWEEN ? AND ?
           OR linear_trend_180 NOT BETWEEN ? AND ?
           OR linear_trend_300 NOT BETWEEN ? AND ?
        """,
        (HR_MIN, HR_MAX, HR_MIN, HR_MAX, HR_MIN, HR_MAX, HR_MIN, HR_MAX, HR_MIN, HR_MAX),
    ).fetchone()[0]
    integrity = output.execute("PRAGMA integrity_check").fetchone()[0]
    output.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "baseline_version": BASELINE_VERSION,
        "ewma_alpha": ewma_alpha,
        "expected_evaluation_origins": expected,
        "prediction_rows": actual,
        "new_sessions_processed": sessions_added,
        "new_prediction_rows": origins_added,
        "context_mask_mismatches": mask_mismatches,
        "missing_feature_series": missing_series,
        "duplicate_prediction_keys": duplicates,
        "out_of_range_predictions": out_of_range,
        "sqlite_integrity_check": integrity,
    }
    payload["all_assertions_pass"] = (
        expected == actual
        and mask_mismatches == 0
        and missing_series == 0
        and duplicates == 0
        and out_of_range == 0
        and integrity == "ok"
    )
    origins.close()
    features.close()
    output.close()
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def regimes_for_row(row: sqlite3.Row) -> list[str]:
    regimes: list[str] = []
    dataset = row["dataset"]
    if dataset == "Endomondo":
        if row["within_user_temporal_partition"] == "test":
            regimes.append("internal_temporal_test")
        if row["unseen_user_partition"] == "test":
            regimes.append("unseen_user_test")
        if int(row["sport_shift_candidate"]) == 1:
            family = row["sport_family"]
            regimes.append(f"unseen_sport__{family}")
            if row["joint_shift_user_partition"] == "test":
                regimes.append(f"joint_user_sport__{family}")
    elif (
        dataset == "GoldenCheetah"
        and row["primary_external_partition"] == "frozen_external_test"
    ):
        regimes.append("goldencheetah_frozen_external")
        regimes.append(f"goldencheetah_external__{row['sport_family']}")
    return regimes


def evaluate_predictions(
    origins_path: Path, prediction_path: Path, metrics_path: Path
) -> dict[str, object]:
    connection = sqlite3.connect(prediction_path)
    connection.row_factory = sqlite3.Row
    connection.execute("ATTACH DATABASE ? AS origin_db", (str(origins_path.resolve()),))
    query = connection.execute(
        """
        SELECT o.dataset, o.session_key, o.user_id, o.sport_family,
               o.within_user_temporal_partition, o.unseen_user_partition,
               o.sport_shift_candidate, o.joint_shift_user_partition,
               o.primary_external_partition,
               o.target_hr_60, o.target_hr_180, o.target_hr_300,
               p.persistence, p.ewma,
               p.linear_trend_60, p.linear_trend_180, p.linear_trend_300
        FROM predictions AS p
        JOIN origin_db.origins AS o
          ON o.dataset = p.dataset
         AND o.session_key = p.session_key
         AND o.origin_time = p.origin_time
        WHERE o.evaluation_origin = 1
        ORDER BY o.dataset, o.session_key, o.origin_time
        """
    )
    user_stats: defaultdict[tuple[str, str, int, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    evaluated_rows = 0
    for (_, _), rows_iterator in grouped_origin_rows(query):
        session_stats: defaultdict[tuple[str, str, int], list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0, 0.0]
        )
        user_id = ""
        for row in rows_iterator:
            user_id = str(row["user_id"])
            targets = {
                60: float(row["target_hr_60"]),
                180: float(row["target_hr_180"]),
                300: float(row["target_hr_300"]),
            }
            models = {
                "persistence": {
                    horizon: float(row["persistence"]) for horizon in HORIZONS
                },
                "ewma": {horizon: float(row["ewma"]) for horizon in HORIZONS},
                "linear_trend": {
                    60: float(row["linear_trend_60"]),
                    180: float(row["linear_trend_180"]),
                    300: float(row["linear_trend_300"]),
                },
            }
            for regime in regimes_for_row(row):
                for model, horizon_predictions in models.items():
                    for horizon in HORIZONS:
                        error = horizon_predictions[horizon] - targets[horizon]
                        accumulator = session_stats[(regime, model, horizon)]
                        accumulator[0] += abs(error)
                        accumulator[1] += error * error
                        accumulator[2] += error
                        accumulator[3] += 1
            evaluated_rows += 1
        for (regime, model, horizon), (sum_abs, sum_sq, sum_error, count) in session_stats.items():
            accumulator = user_stats[(regime, model, horizon, user_id)]
            accumulator[0] += sum_abs / count
            accumulator[1] += math.sqrt(sum_sq / count)
            accumulator[2] += sum_error / count
            accumulator[3] += 1
            accumulator[4] += count
    grouped: defaultdict[tuple[str, str, int], list[tuple[float, float, float, int, int]]] = defaultdict(list)
    for (regime, model, horizon, _), values in user_stats.items():
        sum_session_mae, sum_session_rmse, sum_session_bias, sessions, origins = values
        grouped[(regime, model, horizon)].append(
            (
                sum_session_mae / sessions,
                sum_session_rmse / sessions,
                sum_session_bias / sessions,
                int(sessions),
                int(origins),
            )
        )
    rows: list[dict[str, object]] = []
    for (regime, model, horizon), users in sorted(grouped.items()):
        rows.append(
            {
                "baseline_version": BASELINE_VERSION,
                "regime": regime,
                "model": model,
                "horizon_seconds": horizon,
                "mae_bpm": sum(value[0] for value in users) / len(users),
                "rmse_bpm": sum(value[1] for value in users) / len(users),
                "bias_bpm": sum(value[2] for value in users) / len(users),
                "users": len(users),
                "sessions": sum(value[3] for value in users),
                "origins": sum(value[4] for value in users),
                "aggregation": "origin-within-session, session-within-user, equal-user mean",
            }
        )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    joined = connection.execute(
        """
        SELECT COUNT(*) FROM predictions AS p JOIN origin_db.origins AS o
          ON o.dataset=p.dataset AND o.session_key=p.session_key AND o.origin_time=p.origin_time
        WHERE o.evaluation_origin=1
        """
    ).fetchone()[0]
    prediction_count = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    connection.close()
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "baseline_version": BASELINE_VERSION,
        "prediction_rows": prediction_count,
        "joined_evaluation_rows": joined,
        "streamed_evaluation_rows": evaluated_rows,
        "metric_rows": len(rows),
        "regimes": sorted({str(row["regime"]) for row in rows}),
        "models": sorted({str(row["model"]) for row in rows}),
        "horizons_seconds": list(HORIZONS),
        "metrics_output": str(metrics_path),
        "all_assertions_pass": joined == prediction_count == evaluated_rows and bool(rows),
    }
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune, build, and evaluate causal no-training heart-rate baselines."
    )
    parser.add_argument(
        "command", choices=("tune", "predict", "evaluate", "all"), nargs="?", default="all"
    )
    parser.add_argument("--origins", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--tuning-output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    existing_audit: dict[str, object] = {}
    if args.audit_output.exists():
        existing_audit = json.loads(args.audit_output.read_text(encoding="utf-8"))
    tuning: dict[str, object] | None = None
    prediction_audit: dict[str, object] | None = existing_audit.get("prediction_build")
    evaluation_audit: dict[str, object] | None = existing_audit.get("evaluation")
    if args.command in ("tune", "all"):
        tuning = tune_ewma(args.origins, args.features, args.tuning_output)
        print(json.dumps(tuning, ensure_ascii=False), flush=True)
    if tuning is None:
        tuning = json.loads(args.tuning_output.read_text(encoding="utf-8"))
    alpha = float(tuning["selected_alpha"])
    if args.command in ("predict", "all"):
        prediction_audit = build_predictions(
            args.origins, args.features, args.predictions, alpha
        )
        print(json.dumps(prediction_audit, ensure_ascii=False), flush=True)
    if args.command in ("evaluate", "all"):
        evaluation_audit = evaluate_predictions(
            args.origins, args.predictions, args.metrics_output
        )
        print(json.dumps(evaluation_audit, ensure_ascii=False), flush=True)
    audit = {
        "generated_at_utc": utc_now(),
        "baseline_version": BASELINE_VERSION,
        "tuning": tuning,
        "prediction_build": prediction_audit,
        "evaluation": evaluation_audit,
    }
    atomic_json(args.audit_output, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
