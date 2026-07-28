from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import struct
import sys
import zlib
from array import array
from datetime import datetime, timezone
from pathlib import Path

from build_forecast_origins import evenly_select_mapping
from data_audit import extract_array_bytes, parse_numeric_array


FEATURE_VERSION = "0.4.0"
GRID_SECONDS = 10
VALID_HR_MIN = 30.0
VALID_HR_MAX = 240.0
VALID_ALTITUDE_MIN = -500.0
VALID_ALTITUDE_MAX = 9000.0
VALID_SPEED_MIN = 0.0
VALID_SPEED_MAX = 200.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def connect_database(path: Path) -> sqlite3.Connection:
    if sys.byteorder != "little":
        raise RuntimeError("session-series blobs require a little-endian runtime")
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
            error TEXT NOT NULL,
            processed_at_utc TEXT NOT NULL,
            PRIMARY KEY (dataset, session_key)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS session_series (
            dataset TEXT NOT NULL,
            session_key TEXT NOT NULL,
            grid_start_bin INTEGER NOT NULL,
            grid_seconds INTEGER NOT NULL,
            n_bins INTEGER NOT NULL,
            session_start_time REAL NOT NULL,
            session_end_time REAL NOT NULL,
            speed_source TEXT NOT NULL,
            hr_coverage REAL NOT NULL,
            altitude_coverage REAL NOT NULL,
            speed_coverage REAL NOT NULL,
            hr_values_zlib BLOB NOT NULL,
            hr_mask_zlib BLOB NOT NULL,
            altitude_values_zlib BLOB NOT NULL,
            altitude_mask_zlib BLOB NOT NULL,
            speed_values_zlib BLOB NOT NULL,
            speed_mask_zlib BLOB NOT NULL,
            PRIMARY KEY (dataset, session_key)
        )
        """
    )
    metadata = {
        "feature_version": FEATURE_VERSION,
        "grid_seconds": str(GRID_SECONDS),
        "value_encoding": "zlib-compressed little-endian float32",
        "mask_encoding": "zlib-compressed uint8",
        "valid_hr_bpm": json.dumps([VALID_HR_MIN, VALID_HR_MAX]),
        "valid_altitude_m": json.dumps([VALID_ALTITUDE_MIN, VALID_ALTITUDE_MAX]),
        "valid_speed_kmh": json.dumps([VALID_SPEED_MIN, VALID_SPEED_MAX]),
    }
    for key, value in metadata.items():
        existing = connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if existing is not None and existing[0] != value:
            raise ValueError(f"feature database metadata mismatch for {key}")
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, value)
        )
    connection.commit()
    return connection


def compress_float32(values: array) -> bytes:
    if values.typecode != "f":
        raise TypeError("float array must use typecode 'f'")
    return zlib.compress(values.tobytes(), level=6)


def compress_uint8(values: array) -> bytes:
    if values.typecode != "B":
        raise TypeError("mask array must use typecode 'B'")
    return zlib.compress(values.tobytes(), level=6)


def decompress_float32(blob: bytes, expected_length: int) -> array:
    values = array("f")
    values.frombytes(zlib.decompress(blob))
    if len(values) != expected_length:
        raise ValueError("decoded float sequence has the wrong length")
    return values


def decompress_uint8(blob: bytes, expected_length: int) -> array:
    values = array("B")
    values.frombytes(zlib.decompress(blob))
    if len(values) != expected_length:
        raise ValueError("decoded mask sequence has the wrong length")
    return values


def valid_finite(value: float, lower: float, upper: float) -> bool:
    return math.isfinite(value) and lower <= value <= upper


def bin_end_index(timestamp: float) -> int:
    return int(math.ceil(timestamp / GRID_SECONDS))


def make_empty_series(n_bins: int) -> tuple[array, array]:
    return array("f", [0.0]) * n_bins, array("B", [0]) * n_bins


def assign_last_in_bin(
    timestamps: list[float],
    values: list[float],
    grid_start_bin: int,
    n_bins: int,
    lower: float,
    upper: float,
) -> tuple[array, array]:
    output, mask = make_empty_series(n_bins)
    if len(timestamps) != len(values):
        return output, mask
    for timestamp, value in zip(timestamps, values):
        if not valid_finite(float(value), lower, upper):
            continue
        position = bin_end_index(float(timestamp)) - grid_start_bin
        if 0 <= position < n_bins:
            output[position] = float(value)
            mask[position] = 1
    return output, mask


def haversine_km(
    latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float
) -> float:
    radius_km = 6371.0088
    lat_1 = math.radians(latitude_1)
    lat_2 = math.radians(latitude_2)
    delta_lat = lat_2 - lat_1
    delta_lon = math.radians(longitude_2 - longitude_1)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * radius_km * math.asin(min(1.0, math.sqrt(value)))


def derive_gps_speed(
    timestamps: list[float], latitude: list[float], longitude: list[float]
) -> list[float]:
    result = [math.nan] * len(timestamps)
    if not (len(timestamps) == len(latitude) == len(longitude)):
        return result
    for index in range(1, len(timestamps)):
        delta_seconds = timestamps[index] - timestamps[index - 1]
        if not 0 < delta_seconds <= 60:
            continue
        coordinates = (
            latitude[index - 1],
            longitude[index - 1],
            latitude[index],
            longitude[index],
        )
        if not all(math.isfinite(float(value)) for value in coordinates):
            continue
        if not (-90 <= latitude[index - 1] <= 90 and -90 <= latitude[index] <= 90):
            continue
        if not (
            -180 <= longitude[index - 1] <= 180 and -180 <= longitude[index] <= 180
        ):
            continue
        speed = (
            haversine_km(
                latitude[index - 1],
                longitude[index - 1],
                latitude[index],
                longitude[index],
            )
            * 3600
            / delta_seconds
        )
        if VALID_SPEED_MIN <= speed <= VALID_SPEED_MAX:
            result[index] = speed
    return result


def derive_distance_speed(timestamps: list[float], distance_km: list[float]) -> list[float]:
    result = [math.nan] * len(timestamps)
    if len(timestamps) != len(distance_km):
        return result
    for index in range(1, len(timestamps)):
        delta_seconds = timestamps[index] - timestamps[index - 1]
        if not 0 < delta_seconds <= 60:
            continue
        previous = distance_km[index - 1]
        current = distance_km[index]
        if not (math.isfinite(previous) and math.isfinite(current)):
            continue
        delta_distance = current - previous
        if delta_distance < 0:
            continue
        speed = delta_distance * 3600 / delta_seconds
        if VALID_SPEED_MIN <= speed <= VALID_SPEED_MAX:
            result[index] = speed
    return result


def build_series_row(
    dataset: str,
    session_key: str,
    timestamps: list[float],
    heart_rate: list[float],
    altitude: list[float],
    speed: list[float],
    speed_source: str,
) -> tuple[object, ...]:
    if len(timestamps) < 2:
        raise ValueError("fewer than two timestamps")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("timestamps are not strictly increasing")
    grid_start_bin = bin_end_index(timestamps[0])
    grid_end_bin = bin_end_index(timestamps[-1])
    n_bins = grid_end_bin - grid_start_bin + 1
    if not 1 <= n_bins <= 8641:
        raise ValueError(f"unexpected grid length: {n_bins}")

    hr_values, hr_mask = assign_last_in_bin(
        timestamps,
        heart_rate,
        grid_start_bin,
        n_bins,
        VALID_HR_MIN,
        VALID_HR_MAX,
    )
    altitude_values, altitude_mask = assign_last_in_bin(
        timestamps,
        altitude,
        grid_start_bin,
        n_bins,
        VALID_ALTITUDE_MIN,
        VALID_ALTITUDE_MAX,
    )
    speed_values, speed_mask = assign_last_in_bin(
        timestamps,
        speed,
        grid_start_bin,
        n_bins,
        VALID_SPEED_MIN,
        VALID_SPEED_MAX,
    )
    hr_count = sum(hr_mask)
    altitude_count = sum(altitude_mask)
    speed_count = sum(speed_mask)
    return (
        dataset,
        session_key,
        grid_start_bin,
        GRID_SECONDS,
        n_bins,
        float(timestamps[0]),
        float(timestamps[-1]),
        speed_source,
        hr_count / n_bins,
        altitude_count / n_bins,
        speed_count / n_bins,
        compress_float32(hr_values),
        compress_uint8(hr_mask),
        compress_float32(altitude_values),
        compress_uint8(altitude_mask),
        compress_float32(speed_values),
        compress_uint8(speed_mask),
    )


def eligible_origin_sessions(
    origin_database: Path, dataset: str
) -> dict[str, dict[str, str]]:
    connection = sqlite3.connect(origin_database)
    try:
        return {
            row[0]: {"session_key": row[0]}
            for row in connection.execute(
                """
                SELECT session_key FROM processed_sessions
                WHERE dataset = ? AND status = 'processed' AND accepted_origins > 0
                ORDER BY session_key
                """,
                (dataset,),
            )
        }
    finally:
        connection.close()


def processed_keys(connection: sqlite3.Connection, dataset: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT session_key FROM processed_sessions WHERE dataset = ?", (dataset,)
        )
    }


def insert_series(
    connection: sqlite3.Connection,
    row: tuple[object, ...],
) -> None:
    placeholders = ",".join("?" for _ in row)
    connection.execute(
        f"""
        INSERT INTO session_series(
            dataset, session_key, grid_start_bin, grid_seconds, n_bins,
            session_start_time, session_end_time, speed_source,
            hr_coverage, altitude_coverage, speed_coverage,
            hr_values_zlib, hr_mask_zlib,
            altitude_values_zlib, altitude_mask_zlib,
            speed_values_zlib, speed_mask_zlib
        ) VALUES ({placeholders})
        """,
        row,
    )
    connection.execute(
        """
        INSERT INTO processed_sessions(dataset, session_key, status, error, processed_at_utc)
        VALUES (?, ?, 'processed', '', ?)
        """,
        (row[0], row[1], utc_now()),
    )


def insert_error(
    connection: sqlite3.Connection, dataset: str, session_key: str, exc: Exception
) -> None:
    connection.execute(
        """
        INSERT INTO processed_sessions(dataset, session_key, status, error, processed_at_utc)
        VALUES (?, ?, 'error', ?, ?)
        """,
        (dataset, session_key, f"{type(exc).__name__}: {exc}", utc_now()),
    )


def process_endomondo(
    connection: sqlite3.Connection,
    source: Path,
    origin_database: Path,
    limit_sessions: int | None,
) -> dict[str, object]:
    dataset = "Endomondo"
    all_sessions = eligible_origin_sessions(origin_database, dataset)
    selected = evenly_select_mapping(all_sessions, limit_sessions)
    completed = processed_keys(connection, dataset)
    new_sessions = 0
    errors = 0
    connection.execute("BEGIN")
    with source.open("rb") as handle:
        for record_index, line in enumerate(handle, start=1):
            session_key = str(record_index)
            if session_key not in selected or session_key in completed:
                continue
            try:
                timestamps = parse_numeric_array(
                    extract_array_bytes(line, "timestamp"), cast=float
                )
                heart_rate = parse_numeric_array(
                    extract_array_bytes(line, "heart_rate"), cast=float
                )
                altitude = parse_numeric_array(
                    extract_array_bytes(line, "altitude"), cast=float
                )
                direct_speed = parse_numeric_array(
                    extract_array_bytes(line, "speed"), cast=float
                )
                if len(direct_speed) == len(timestamps) and any(
                    valid_finite(value, VALID_SPEED_MIN, VALID_SPEED_MAX)
                    for value in direct_speed
                ):
                    speed = direct_speed
                    speed_source = "direct_kmh"
                else:
                    latitude = parse_numeric_array(
                        extract_array_bytes(line, "latitude"), cast=float
                    )
                    longitude = parse_numeric_array(
                        extract_array_bytes(line, "longitude"), cast=float
                    )
                    speed = derive_gps_speed(timestamps, latitude, longitude)
                    speed_source = "gps_derived_kmh"
                row = build_series_row(
                    dataset,
                    session_key,
                    timestamps,
                    heart_rate,
                    altitude,
                    speed,
                    speed_source,
                )
                insert_series(connection, row)
            except Exception as exc:
                errors += 1
                insert_error(connection, dataset, session_key, exc)
            new_sessions += 1
            if new_sessions % 250 == 0:
                connection.commit()
                connection.execute("BEGIN")
                print(f"Endomondo series: {new_sessions:,}", flush=True)
    connection.commit()
    return {
        "dataset": dataset,
        "sessions_with_origins": len(all_sessions),
        "sessions_selected_for_scope": len(selected),
        "previously_processed": len(completed),
        "newly_processed": new_sessions,
        "errors": errors,
    }


def parse_golden_features(
    path: Path,
) -> tuple[list[float], list[float], list[float], list[float]]:
    timestamps: list[float] = []
    heart_rate: list[float] = []
    altitude: list[float] = []
    distance_km: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        index = {name.strip(): position for position, name in enumerate(header)}
        required = {"secs", "hr", "km", "alt"}
        if not required.issubset(index):
            raise ValueError(f"missing required columns: {required - set(index)}")
        for row in reader:
            try:
                timestamp = float(row[index["secs"]])
            except (ValueError, IndexError):
                continue
            if not math.isfinite(timestamp):
                continue
            timestamps.append(timestamp)
            values: list[float] = []
            for field in ("hr", "alt", "km"):
                try:
                    value = float(row[index[field]])
                    values.append(value if math.isfinite(value) else math.nan)
                except (ValueError, IndexError):
                    values.append(math.nan)
            heart_rate.append(values[0])
            altitude.append(values[1])
            distance_km.append(values[2])
    speed = derive_distance_speed(timestamps, distance_km)
    return timestamps, heart_rate, altitude, speed


def process_golden(
    connection: sqlite3.Connection,
    root: Path,
    origin_database: Path,
    limit_sessions: int | None,
) -> dict[str, object]:
    dataset = "GoldenCheetah"
    all_sessions = eligible_origin_sessions(origin_database, dataset)
    selected = evenly_select_mapping(all_sessions, limit_sessions)
    completed = processed_keys(connection, dataset)
    new_sessions = 0
    errors = 0
    connection.execute("BEGIN")
    for session_key in selected:
        if session_key in completed:
            continue
        try:
            timestamps, heart_rate, altitude, speed = parse_golden_features(
                root / Path(session_key)
            )
            row = build_series_row(
                dataset,
                session_key,
                timestamps,
                heart_rate,
                altitude,
                speed,
                "distance_derived_kmh",
            )
            insert_series(connection, row)
        except Exception as exc:
            errors += 1
            insert_error(connection, dataset, session_key, exc)
        new_sessions += 1
        if new_sessions % 250 == 0:
            connection.commit()
            connection.execute("BEGIN")
            print(f"GoldenCheetah series: {new_sessions:,}", flush=True)
    connection.commit()
    return {
        "dataset": dataset,
        "sessions_with_origins": len(all_sessions),
        "sessions_selected_for_scope": len(selected),
        "previously_processed": len(completed),
        "newly_processed": new_sessions,
        "errors": errors,
    }


def audit_database(connection: sqlite3.Connection) -> dict[str, object]:
    dataset_summary = [
        {
            "dataset": row[0],
            "sessions": row[1],
            "bins": row[2],
            "mean_hr_coverage": row[3],
            "mean_altitude_coverage": row[4],
            "mean_speed_coverage": row[5],
        }
        for row in connection.execute(
            """
            SELECT dataset, COUNT(*), SUM(n_bins), AVG(hr_coverage),
                   AVG(altitude_coverage), AVG(speed_coverage)
            FROM session_series GROUP BY dataset ORDER BY dataset
            """
        )
    ]
    speed_sources = [
        {"dataset": row[0], "speed_source": row[1], "sessions": row[2]}
        for row in connection.execute(
            """
            SELECT dataset, speed_source, COUNT(*) FROM session_series
            GROUP BY dataset, speed_source ORDER BY dataset, COUNT(*) DESC
            """
        )
    ]
    errors = connection.execute(
        "SELECT COUNT(*) FROM processed_sessions WHERE status = 'error'"
    ).fetchone()[0]
    length_mismatches = 0
    sample_rows = connection.execute(
        """
        SELECT n_bins, hr_values_zlib, hr_mask_zlib,
               altitude_values_zlib, altitude_mask_zlib,
               speed_values_zlib, speed_mask_zlib
        FROM session_series ORDER BY dataset, session_key LIMIT 200
        """
    )
    for row in sample_rows:
        n_bins = row[0]
        try:
            decompress_float32(row[1], n_bins)
            decompress_uint8(row[2], n_bins)
            decompress_float32(row[3], n_bins)
            decompress_uint8(row[4], n_bins)
            decompress_float32(row[5], n_bins)
            decompress_uint8(row[6], n_bins)
        except Exception:
            length_mismatches += 1
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    return {
        "generated_at_utc": utc_now(),
        "feature_version": FEATURE_VERSION,
        "dataset_summary": dataset_summary,
        "speed_sources": speed_sources,
        "assertion_failure_counts": {
            "processing_errors": errors,
            "sampled_blob_length_mismatches": length_mismatches,
            "sqlite_integrity_failure": int(integrity != "ok"),
        },
        "sqlite_integrity_check": integrity,
        "all_assertions_pass": errors == 0 and length_mismatches == 0 and integrity == "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a resumable per-session causal 10-second feature-series cache."
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    endomondo = subparsers.add_parser("endomondo")
    endomondo.add_argument("--source", type=Path, required=True)
    endomondo.add_argument("--origins", type=Path, required=True)
    endomondo.add_argument("--database", type=Path, required=True)
    endomondo.add_argument("--summary-output", type=Path, required=True)
    endomondo.add_argument("--limit-sessions", type=int)

    golden = subparsers.add_parser("goldencheetah")
    golden.add_argument("--root", type=Path, required=True)
    golden.add_argument("--origins", type=Path, required=True)
    golden.add_argument("--database", type=Path, required=True)
    golden.add_argument("--summary-output", type=Path, required=True)
    golden.add_argument("--limit-sessions", type=int)
    args = parser.parse_args()

    connection = connect_database(args.database)
    try:
        if args.dataset == "endomondo":
            run = process_endomondo(
                connection, args.source, args.origins, args.limit_sessions
            )
        else:
            run = process_golden(
                connection, args.root, args.origins, args.limit_sessions
            )
        audit = audit_database(connection)
        audit["run"] = run
        audit["database"] = str(args.database)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        audit["database_bytes"] = args.database.stat().st_size
        atomic_json(args.summary_output, audit)
        print(json.dumps(audit, ensure_ascii=False)[:10000], flush=True)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
