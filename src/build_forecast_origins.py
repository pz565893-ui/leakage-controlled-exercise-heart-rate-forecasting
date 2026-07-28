from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from data_audit import extract_array_bytes, parse_numeric_array


WINDOW_VERSION = "0.3.1"
CONTEXT_SECONDS = 300
GRID_SECONDS = 10
ORIGIN_STRIDE_SECONDS = 60
EVALUATION_STRIDE_SECONDS = 300
HORIZONS = (60, 180, 300)
MIN_CONTEXT_BIN_COVERAGE = 0.80
MAX_CONTEXT_GAP_SECONDS = 60
MAX_TARGET_INTERPOLATION_SPAN_SECONDS = 30
VALID_HR_MIN = 30.0
VALID_HR_MAX = 240.0


ORIGIN_COLUMNS = (
    "dataset",
    "session_key",
    "user_id",
    "sport_family",
    "origin_time",
    "origin_offset_seconds",
    "context_start_time",
    "context_raw_points",
    "context_valid_bins",
    "context_total_bins",
    "context_hr_coverage",
    "context_max_gap_seconds",
    "input_last_time",
    "target_hr_60",
    "target_hr_180",
    "target_hr_300",
    "target_span_60",
    "target_span_180",
    "target_span_300",
    "target_min_source_time",
    "target_max_source_time",
    "evaluation_origin",
    "unseen_user_partition",
    "within_user_temporal_partition",
    "sport_shift_candidate",
    "joint_shift_user_partition",
    "primary_external_partition",
    "secondary_adaptation_partition",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_eligible_splits(path: Path, key_field: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["analysis_eligible"] == "True":
                key = row[key_field]
                if key in rows:
                    raise ValueError(f"duplicate split key: {key}")
                rows[key] = row
    return rows


def evenly_select_mapping(
    rows: dict[str, dict[str, str]], requested: int | None
) -> dict[str, dict[str, str]]:
    if requested is None or requested >= len(rows):
        return rows
    if requested <= 0:
        return {}
    items = list(rows.items())
    selected: dict[str, dict[str, str]] = {}
    for index in range(requested):
        position = min(
            len(items) - 1,
            int((index + 0.5) * len(items) / requested),
        )
        key, row = items[position]
        selected[key] = row
    return selected


def connect_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-200000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_sessions (
            dataset TEXT NOT NULL,
            session_key TEXT NOT NULL,
            status TEXT NOT NULL,
            candidate_origins INTEGER NOT NULL,
            accepted_origins INTEGER NOT NULL,
            rejection_counts_json TEXT NOT NULL,
            processed_at_utc TEXT NOT NULL,
            PRIMARY KEY (dataset, session_key)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS origins (
            dataset TEXT NOT NULL,
            session_key TEXT NOT NULL,
            user_id TEXT NOT NULL,
            sport_family TEXT NOT NULL,
            origin_time REAL NOT NULL,
            origin_offset_seconds REAL NOT NULL,
            context_start_time REAL NOT NULL,
            context_raw_points INTEGER NOT NULL,
            context_valid_bins INTEGER NOT NULL,
            context_total_bins INTEGER NOT NULL,
            context_hr_coverage REAL NOT NULL,
            context_max_gap_seconds REAL NOT NULL,
            input_last_time REAL NOT NULL,
            target_hr_60 REAL NOT NULL,
            target_hr_180 REAL NOT NULL,
            target_hr_300 REAL NOT NULL,
            target_span_60 REAL NOT NULL,
            target_span_180 REAL NOT NULL,
            target_span_300 REAL NOT NULL,
            target_min_source_time REAL NOT NULL,
            target_max_source_time REAL NOT NULL,
            evaluation_origin INTEGER NOT NULL,
            unseen_user_partition TEXT NOT NULL,
            within_user_temporal_partition TEXT NOT NULL,
            sport_shift_candidate INTEGER NOT NULL,
            joint_shift_user_partition TEXT NOT NULL,
            primary_external_partition TEXT NOT NULL,
            secondary_adaptation_partition TEXT NOT NULL,
            UNIQUE (dataset, session_key, origin_time)
        )
        """
    )
    metadata = {
        "window_version": WINDOW_VERSION,
        "context_seconds": str(CONTEXT_SECONDS),
        "grid_seconds": str(GRID_SECONDS),
        "origin_stride_seconds": str(ORIGIN_STRIDE_SECONDS),
        "evaluation_stride_seconds": str(EVALUATION_STRIDE_SECONDS),
        "horizons_seconds": json.dumps(HORIZONS),
        "min_context_bin_coverage": str(MIN_CONTEXT_BIN_COVERAGE),
        "max_context_gap_seconds": str(MAX_CONTEXT_GAP_SECONDS),
        "max_target_interpolation_span_seconds": str(
            MAX_TARGET_INTERPOLATION_SPAN_SECONDS
        ),
    }
    for key, value in metadata.items():
        existing = connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if existing is not None and existing[0] != value:
            raise ValueError(
                f"database metadata mismatch for {key}: {existing[0]!r} != {value!r}"
            )
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, value)
        )
    connection.commit()
    return connection


def valid_hr(value: float) -> bool:
    return math.isfinite(value) and VALID_HR_MIN <= value <= VALID_HR_MAX


def interpolate_target(
    times: list[float],
    values: list[float],
    target_time: float,
) -> tuple[float, float, float, float] | None:
    position = bisect.bisect_left(times, target_time)
    if position < len(times) and times[position] == target_time:
        return values[position], 0.0, times[position], times[position]
    if position == 0 or position >= len(times):
        return None
    left_time = times[position - 1]
    right_time = times[position]
    span = right_time - left_time
    if span <= 0 or span > MAX_TARGET_INTERPOLATION_SPAN_SECONDS:
        return None
    weight = (target_time - left_time) / span
    value = values[position - 1] + weight * (values[position] - values[position - 1])
    if not valid_hr(value):
        return None
    return value, span, left_time, right_time


def first_stride_aligned_time(lower_bound: float) -> int:
    return int(math.ceil(lower_bound / ORIGIN_STRIDE_SECONDS) * ORIGIN_STRIDE_SECONDS)


def context_bin_index(timestamp: float, context_start: float) -> int | None:
    relative = timestamp - context_start
    if relative <= 0 or relative > CONTEXT_SECONDS:
        return None
    return min(
        CONTEXT_SECONDS // GRID_SECONDS - 1,
        int(math.ceil(relative / GRID_SECONDS) - 1),
    )


def build_session_origins(
    dataset: str,
    session_key: str,
    split: dict[str, str],
    timestamps: list[float],
    heart_rate: list[float],
) -> tuple[list[tuple[object, ...]], int, Counter[str]]:
    rejection_counts: Counter[str] = Counter()
    if len(timestamps) != len(heart_rate) or len(timestamps) < 2:
        rejection_counts["invalid_or_unaligned_arrays"] += 1
        return [], 0, rejection_counts
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        rejection_counts["non_monotonic_timestamps"] += 1
        return [], 0, rejection_counts

    valid_pairs = [
        (float(timestamp), float(hr))
        for timestamp, hr in zip(timestamps, heart_rate)
        if valid_hr(float(hr))
    ]
    if len(valid_pairs) < 2:
        rejection_counts["insufficient_valid_hr"] += 1
        return [], 0, rejection_counts
    valid_times = [pair[0] for pair in valid_pairs]
    valid_values = [pair[1] for pair in valid_pairs]
    session_start = float(timestamps[0])
    session_end = float(timestamps[-1])
    first_origin = first_stride_aligned_time(session_start + CONTEXT_SECONDS)
    last_origin = session_end - max(HORIZONS)
    if first_origin > last_origin:
        rejection_counts["session_too_short_after_alignment"] += 1
        return [], 0, rejection_counts

    candidate_origins = list(
        range(first_origin, int(math.floor(last_origin)) + 1, ORIGIN_STRIDE_SECONDS)
    )
    accepted: list[tuple[object, ...]] = []
    total_bins = CONTEXT_SECONDS // GRID_SECONDS

    for origin in candidate_origins:
        origin_time = float(origin)
        context_start = origin_time - CONTEXT_SECONDS
        left = bisect.bisect_left(valid_times, context_start)
        right = bisect.bisect_right(valid_times, origin_time)
        context_times = valid_times[left:right]
        if not context_times:
            rejection_counts["empty_context"] += 1
            continue

        occupied_bins: set[int] = set()
        for timestamp in context_times:
            bin_index = context_bin_index(timestamp, context_start)
            if bin_index is not None:
                occupied_bins.add(bin_index)
        valid_bins = len(occupied_bins)
        coverage = valid_bins / total_bins
        if coverage < MIN_CONTEXT_BIN_COVERAGE:
            rejection_counts["context_bin_coverage"] += 1
            continue

        gap_candidates = [context_times[0] - context_start, origin_time - context_times[-1]]
        gap_candidates.extend(
            current - previous
            for previous, current in zip(context_times, context_times[1:])
        )
        max_context_gap = max(gap_candidates)
        if max_context_gap > MAX_CONTEXT_GAP_SECONDS:
            rejection_counts["context_gap"] += 1
            continue

        target_results: dict[int, tuple[float, float, float, float]] = {}
        target_failed = False
        for horizon in HORIZONS:
            result = interpolate_target(valid_times, valid_values, origin_time + horizon)
            if result is None:
                rejection_counts[f"target_{horizon}_unavailable"] += 1
                target_failed = True
                break
            target_results[horizon] = result
        if target_failed:
            continue

        target_min_source = min(result[2] for result in target_results.values())
        target_max_source = max(result[3] for result in target_results.values())
        if context_times[-1] > origin_time:
            raise AssertionError("input timestamp exceeds forecast origin")
        if target_min_source <= origin_time:
            raise AssertionError("target source timestamp is not strictly after origin")

        row = {
            "dataset": dataset,
            "session_key": session_key,
            "user_id": split["user_id"],
            "sport_family": split["sport_family"],
            "origin_time": origin_time,
            "origin_offset_seconds": origin_time - session_start,
            "context_start_time": context_start,
            "context_raw_points": len(context_times),
            "context_valid_bins": valid_bins,
            "context_total_bins": total_bins,
            "context_hr_coverage": coverage,
            "context_max_gap_seconds": max_context_gap,
            "input_last_time": context_times[-1],
            "target_hr_60": target_results[60][0],
            "target_hr_180": target_results[180][0],
            "target_hr_300": target_results[300][0],
            "target_span_60": target_results[60][1],
            "target_span_180": target_results[180][1],
            "target_span_300": target_results[300][1],
            "target_min_source_time": target_min_source,
            "target_max_source_time": target_max_source,
            "evaluation_origin": int(origin % EVALUATION_STRIDE_SECONDS == 0),
            "unseen_user_partition": split.get("unseen_user_partition", ""),
            "within_user_temporal_partition": split.get(
                "within_user_temporal_partition", ""
            ),
            "sport_shift_candidate": int(split.get("sport_shift_candidate") == "True"),
            "joint_shift_user_partition": split.get("joint_shift_user_partition", ""),
            "primary_external_partition": split.get("primary_external_partition", ""),
            "secondary_adaptation_partition": split.get(
                "secondary_adaptation_partition", ""
            ),
        }
        accepted.append(tuple(row[column] for column in ORIGIN_COLUMNS))

    return accepted, len(candidate_origins), rejection_counts


def insert_session_result(
    connection: sqlite3.Connection,
    dataset: str,
    session_key: str,
    rows: list[tuple[object, ...]],
    candidate_origins: int,
    rejection_counts: Counter[str],
    status: str = "processed",
) -> None:
    placeholders = ",".join("?" for _ in ORIGIN_COLUMNS)
    connection.executemany(
        f"INSERT INTO origins ({','.join(ORIGIN_COLUMNS)}) VALUES ({placeholders})",
        rows,
    )
    connection.execute(
        """
        INSERT INTO processed_sessions(
            dataset, session_key, status, candidate_origins, accepted_origins,
            rejection_counts_json, processed_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset,
            session_key,
            status,
            candidate_origins,
            len(rows),
            json.dumps(rejection_counts, ensure_ascii=False, sort_keys=True),
            utc_now(),
        ),
    )


def processed_keys(connection: sqlite3.Connection, dataset: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT session_key FROM processed_sessions WHERE dataset = ?", (dataset,)
        )
    }


def parse_golden_session(path: Path) -> tuple[list[float], list[float]]:
    timestamps: list[float] = []
    heart_rate: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        index = {name.strip(): position for position, name in enumerate(header)}
        if "secs" not in index or "hr" not in index:
            return timestamps, heart_rate
        secs_index = index["secs"]
        hr_index = index["hr"]
        for row in reader:
            if secs_index >= len(row) or hr_index >= len(row):
                continue
            try:
                timestamp = float(row[secs_index])
                hr = float(row[hr_index])
            except ValueError:
                continue
            if not math.isfinite(timestamp) or not math.isfinite(hr):
                continue
            timestamps.append(timestamp)
            heart_rate.append(hr)
    return timestamps, heart_rate


def process_endomondo(
    connection: sqlite3.Connection,
    source: Path,
    split_path: Path,
    limit_sessions: int | None,
) -> dict[str, object]:
    dataset = "Endomondo"
    all_eligible = load_eligible_splits(split_path, "record_index")
    eligible = evenly_select_mapping(all_eligible, limit_sessions)
    completed = processed_keys(connection, dataset)
    newly_processed = 0
    origins_added = 0
    errors = 0
    connection.execute("BEGIN")
    with source.open("rb") as handle:
        for record_index, line in enumerate(handle, start=1):
            session_key = str(record_index)
            if session_key not in eligible or session_key in completed:
                continue
            split = eligible[session_key]
            try:
                timestamps = parse_numeric_array(
                    extract_array_bytes(line, "timestamp"), cast=float
                )
                heart_rate = parse_numeric_array(
                    extract_array_bytes(line, "heart_rate"), cast=float
                )
                rows, candidates, rejections = build_session_origins(
                    dataset, session_key, split, timestamps, heart_rate
                )
                insert_session_result(
                    connection, dataset, session_key, rows, candidates, rejections
                )
                origins_added += len(rows)
            except Exception as exc:
                errors += 1
                insert_session_result(
                    connection,
                    dataset,
                    session_key,
                    [],
                    0,
                    Counter({f"{type(exc).__name__}: {exc}": 1}),
                    status="error",
                )
            newly_processed += 1
            if newly_processed % 250 == 0:
                connection.commit()
                connection.execute("BEGIN")
                print(
                    f"Endomondo sessions processed: {newly_processed:,}; origins added: {origins_added:,}",
                    flush=True,
                )
    connection.commit()
    return {
        "dataset": dataset,
        "eligible_sessions_in_split": len(all_eligible),
        "sessions_selected_for_this_build_scope": len(eligible),
        "previously_processed_sessions": len(completed),
        "newly_processed_sessions": newly_processed,
        "origins_added": origins_added,
        "session_errors": errors,
    }


def process_golden(
    connection: sqlite3.Connection,
    root: Path,
    split_path: Path,
    limit_sessions: int | None,
) -> dict[str, object]:
    dataset = "GoldenCheetah"
    all_eligible = load_eligible_splits(split_path, "csv_relative_path")
    eligible = evenly_select_mapping(all_eligible, limit_sessions)
    completed = processed_keys(connection, dataset)
    newly_processed = 0
    origins_added = 0
    errors = 0
    connection.execute("BEGIN")
    for session_key, split in eligible.items():
        if session_key in completed:
            continue
        try:
            timestamps, heart_rate = parse_golden_session(root / Path(session_key))
            rows, candidates, rejections = build_session_origins(
                dataset, session_key, split, timestamps, heart_rate
            )
            insert_session_result(
                connection, dataset, session_key, rows, candidates, rejections
            )
            origins_added += len(rows)
        except Exception as exc:
            errors += 1
            insert_session_result(
                connection,
                dataset,
                session_key,
                [],
                0,
                Counter({f"{type(exc).__name__}: {exc}": 1}),
                status="error",
            )
        newly_processed += 1
        if newly_processed % 250 == 0:
            connection.commit()
            connection.execute("BEGIN")
            print(
                f"GoldenCheetah sessions processed: {newly_processed:,}; origins added: {origins_added:,}",
                flush=True,
            )
    connection.commit()
    return {
        "dataset": dataset,
        "eligible_sessions_in_split": len(all_eligible),
        "sessions_selected_for_this_build_scope": len(eligible),
        "previously_processed_sessions": len(completed),
        "newly_processed_sessions": newly_processed,
        "origins_added": origins_added,
        "session_errors": errors,
    }


def create_final_indexes(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_origins_dataset_user ON origins(dataset, user_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_origins_dataset_sport ON origins(dataset, sport_family)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_origins_unseen_user ON origins(dataset, unseen_user_partition)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_origins_temporal ON origins(dataset, within_user_temporal_partition)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_origins_external ON origins(dataset, primary_external_partition)"
    )
    connection.commit()


def database_audit(connection: sqlite3.Connection) -> dict[str, object]:
    total_origins = connection.execute("SELECT COUNT(*) FROM origins").fetchone()[0]
    processed = [
        {
            "dataset": row[0],
            "status": row[1],
            "sessions": row[2],
            "candidate_origins": row[3],
            "accepted_origins": row[4],
        }
        for row in connection.execute(
            """
            SELECT dataset, status, COUNT(*), SUM(candidate_origins), SUM(accepted_origins)
            FROM processed_sessions GROUP BY dataset, status ORDER BY dataset, status
            """
        )
    ]
    origins_by_dataset = [
        {"dataset": row[0], "origins": row[1], "evaluation_origins": row[2]}
        for row in connection.execute(
            """
            SELECT dataset, COUNT(*), SUM(evaluation_origin)
            FROM origins GROUP BY dataset ORDER BY dataset
            """
        )
    ]
    origins_by_family = [
        {"dataset": row[0], "sport_family": row[1], "origins": row[2]}
        for row in connection.execute(
            """
            SELECT dataset, sport_family, COUNT(*)
            FROM origins GROUP BY dataset, sport_family
            ORDER BY dataset, COUNT(*) DESC
            """
        )
    ]
    assertions = {
        "session_processing_errors": connection.execute(
            "SELECT COUNT(*) FROM processed_sessions WHERE status = 'error'"
        ).fetchone()[0],
        "input_after_origin": connection.execute(
            "SELECT COUNT(*) FROM origins WHERE input_last_time > origin_time"
        ).fetchone()[0],
        "target_not_strictly_future": connection.execute(
            "SELECT COUNT(*) FROM origins WHERE target_min_source_time <= origin_time"
        ).fetchone()[0],
        "context_coverage_below_threshold": connection.execute(
            "SELECT COUNT(*) FROM origins WHERE context_hr_coverage < ?",
            (MIN_CONTEXT_BIN_COVERAGE,),
        ).fetchone()[0],
        "context_gap_above_threshold": connection.execute(
            "SELECT COUNT(*) FROM origins WHERE context_max_gap_seconds > ?",
            (MAX_CONTEXT_GAP_SECONDS,),
        ).fetchone()[0],
        "target_hr_out_of_range": connection.execute(
            """
            SELECT COUNT(*) FROM origins
            WHERE target_hr_60 NOT BETWEEN ? AND ?
               OR target_hr_180 NOT BETWEEN ? AND ?
               OR target_hr_300 NOT BETWEEN ? AND ?
            """,
            (
                VALID_HR_MIN,
                VALID_HR_MAX,
                VALID_HR_MIN,
                VALID_HR_MAX,
                VALID_HR_MIN,
                VALID_HR_MAX,
            ),
        ).fetchone()[0],
        "target_interpolation_span_above_threshold": connection.execute(
            """
            SELECT COUNT(*) FROM origins
            WHERE target_span_60 > ? OR target_span_180 > ? OR target_span_300 > ?
            """,
            (
                MAX_TARGET_INTERPOLATION_SPAN_SECONDS,
                MAX_TARGET_INTERPOLATION_SPAN_SECONDS,
                MAX_TARGET_INTERPOLATION_SPAN_SECONDS,
            ),
        ).fetchone()[0],
        "evaluation_stride_violation": connection.execute(
            "SELECT COUNT(*) FROM origins WHERE evaluation_origin = 1 AND CAST(origin_time AS INTEGER) % ? != 0",
            (EVALUATION_STRIDE_SECONDS,),
        ).fetchone()[0],
        "duplicate_origin_keys": connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT dataset, session_key, origin_time, COUNT(*) AS n
                FROM origins GROUP BY dataset, session_key, origin_time HAVING n > 1
            )
            """
        ).fetchone()[0],
    }
    return {
        "generated_at_utc": utc_now(),
        "window_version": WINDOW_VERSION,
        "total_origins": total_origins,
        "processed_sessions": processed,
        "origins_by_dataset": origins_by_dataset,
        "origins_by_family": origins_by_family,
        "assertion_failure_counts": assertions,
        "all_assertions_pass": all(value == 0 for value in assertions.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a resumable, leakage-controlled multi-horizon forecast-origin index."
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    endomondo = subparsers.add_parser("endomondo")
    endomondo.add_argument("--source", type=Path, required=True)
    endomondo.add_argument("--split", type=Path, required=True)
    endomondo.add_argument("--database", type=Path, required=True)
    endomondo.add_argument("--summary-output", type=Path, required=True)
    endomondo.add_argument("--limit-sessions", type=int)
    endomondo.add_argument("--finalize-indexes", action="store_true")

    golden = subparsers.add_parser("goldencheetah")
    golden.add_argument("--root", type=Path, required=True)
    golden.add_argument("--split", type=Path, required=True)
    golden.add_argument("--database", type=Path, required=True)
    golden.add_argument("--summary-output", type=Path, required=True)
    golden.add_argument("--limit-sessions", type=int)
    golden.add_argument("--finalize-indexes", action="store_true")
    args = parser.parse_args()

    connection = connect_database(args.database)
    try:
        if args.dataset == "endomondo":
            run_summary = process_endomondo(
                connection, args.source, args.split, args.limit_sessions
            )
        else:
            run_summary = process_golden(
                connection, args.root, args.split, args.limit_sessions
            )
        if args.finalize_indexes:
            create_final_indexes(connection)
        audit = database_audit(connection)
        audit["run"] = run_summary
        audit["database"] = str(args.database)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        audit["database_bytes"] = args.database.stat().st_size
        atomic_json(args.summary_output, audit)
        print(json.dumps(audit, ensure_ascii=False)[:12000], flush=True)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
