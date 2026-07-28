from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sqlite3
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from build_session_series import (
    GRID_SECONDS,
    decompress_float32,
    decompress_uint8,
)


ARRAY_VERSION = "0.6.0"
CONTEXT_BINS = 30
CHANNELS = ("heart_rate_bpm", "speed_kmh", "altitude_m")
SPORT_CODES = {
    "other_unknown": 0,
    "outdoor_cycling": 1,
    "indoor_virtual_cycling": 2,
    "running": 3,
    "walking_hiking": 4,
    "swimming": 5,
    "skiing": 6,
    "strength_cross_training": 7,
}
PARTITION_CODES = {
    "": 0,
    "train": 1,
    "validation": 2,
    "calibration": 3,
    "test": 4,
    "insufficient_user_history": 5,
}
EXTERNAL_CODES = {"": 0, "frozen_external_test": 1, "not_primary_external": 2}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def grouped_rows(
    cursor: Iterable[sqlite3.Row],
) -> Iterator[tuple[tuple[str, str], Iterator[sqlite3.Row]]]:
    return itertools.groupby(cursor, key=lambda row: (row["dataset"], row["session_key"]))


def context_positions(
    origin_time: float, grid_start_bin: int, n_bins: int
) -> tuple[int, int]:
    origin_grid = origin_time / GRID_SECONDS
    origin_bin = int(round(origin_grid))
    if not math.isclose(origin_grid, origin_bin, abs_tol=1e-7):
        raise ValueError("origin is not aligned to the 10-second grid")
    end = origin_bin - grid_start_bin + 1
    start = end - CONTEXT_BINS
    if start < 0 or end > n_bins:
        raise IndexError("context lies outside cached session series")
    return start, end


def load_session_series(
    connection: sqlite3.Connection, dataset: str, session_key: str
) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT grid_start_bin, n_bins, session_start_time, session_end_time,
               hr_values_zlib, hr_mask_zlib,
               speed_values_zlib, speed_mask_zlib,
               altitude_values_zlib, altitude_mask_zlib
        FROM session_series WHERE dataset = ? AND session_key = ?
        """,
        (dataset, session_key),
    ).fetchone()
    if row is None:
        raise KeyError(f"missing session series: {dataset}/{session_key}")
    grid_start_bin, n_bins, session_start, session_end, *blobs = row
    n_bins = int(n_bins)
    return {
        "grid_start_bin": int(grid_start_bin),
        "n_bins": n_bins,
        "session_start_time": float(session_start),
        "session_end_time": float(session_end),
        "hr_values": decompress_float32(blobs[0], n_bins),
        "hr_mask": decompress_uint8(blobs[1], n_bins),
        "speed_values": decompress_float32(blobs[2], n_bins),
        "speed_mask": decompress_uint8(blobs[3], n_bins),
        "altitude_values": decompress_float32(blobs[4], n_bins),
        "altitude_mask": decompress_uint8(blobs[5], n_bins),
    }


def masked_summary(values: Sequence[float], mask: Sequence[int]) -> tuple[float, float, int]:
    observed = [float(value) for value, valid in zip(values, mask) if valid]
    if not observed:
        return 0.0, 0.0, 0
    mean = sum(observed) / len(observed)
    variance = sum((value - mean) ** 2 for value in observed) / len(observed)
    return mean, math.sqrt(variance), len(observed)


def open_arrays(output_dir: Path, n_rows: int) -> dict[str, np.memmap]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes = {
        "sequence_values": (np.float16, (n_rows, CONTEXT_BINS, len(CHANNELS))),
        "sequence_masks": (np.uint8, (n_rows, CONTEXT_BINS, len(CHANNELS))),
        "targets": (np.float32, (n_rows, 3)),
        "origin_offset_seconds": (np.float32, (n_rows,)),
        "origin_time": (np.float64, (n_rows,)),
        "dataset_code": (np.uint8, (n_rows,)),
        "sport_code": (np.uint8, (n_rows,)),
        "user_index": (np.int32, (n_rows,)),
        "session_index": (np.int32, (n_rows,)),
        "evaluation_origin": (np.uint8, (n_rows,)),
        "unseen_user_partition": (np.uint8, (n_rows,)),
        "temporal_partition": (np.uint8, (n_rows,)),
        "joint_user_partition": (np.uint8, (n_rows,)),
        "sport_shift_candidate": (np.uint8, (n_rows,)),
        "primary_external_partition": (np.uint8, (n_rows,)),
    }
    return {
        name: np.lib.format.open_memmap(
            output_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in shapes.items()
    }


def build_arrays(origins_path: Path, features_path: Path, output_dir: Path) -> dict[str, object]:
    origins = sqlite3.connect(origins_path)
    origins.row_factory = sqlite3.Row
    features = sqlite3.connect(features_path)
    selection = "dataset = 'Endomondo' OR (dataset = 'GoldenCheetah' AND evaluation_origin = 1)"
    expected_rows = int(
        origins.execute(f"SELECT COUNT(*) FROM origins WHERE {selection}").fetchone()[0]
    )
    arrays = open_arrays(output_dir, expected_rows)
    user_mapping: dict[tuple[str, str], int] = {}
    session_count = 0
    row_index = 0
    mask_mismatches = 0
    missing_series = 0
    target_range_failures = 0
    session_manifest = output_dir / "sessions.csv"
    query = origins.execute(
        f"""
        SELECT dataset, session_key, user_id, sport_family, origin_time,
               origin_offset_seconds, context_valid_bins,
               target_hr_60, target_hr_180, target_hr_300,
               evaluation_origin, unseen_user_partition,
               within_user_temporal_partition, sport_shift_candidate,
               joint_shift_user_partition, primary_external_partition
        FROM origins WHERE {selection}
        ORDER BY dataset, session_key, origin_time
        """
    )
    with session_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "session_index",
                "dataset",
                "session_key",
                "user_index",
                "sport_code",
                "session_start_time",
                "session_end_time",
                "duration_seconds",
                "hr_mean",
                "hr_std",
                "hr_observed_bins",
                "speed_mean",
                "speed_std",
                "speed_observed_bins",
                "altitude_mean",
                "altitude_std",
                "altitude_observed_bins",
            ]
        )
        for (dataset, session_key), row_iterator in grouped_rows(query):
            rows = list(row_iterator)
            try:
                series = load_session_series(features, dataset, session_key)
            except KeyError:
                missing_series += 1
                continue
            user_key = (dataset, str(rows[0]["user_id"]))
            if user_key not in user_mapping:
                user_mapping[user_key] = len(user_mapping)
            user_index = user_mapping[user_key]
            sport_code = SPORT_CODES[str(rows[0]["sport_family"])]
            hr_summary = masked_summary(series["hr_values"], series["hr_mask"])
            speed_summary = masked_summary(series["speed_values"], series["speed_mask"])
            altitude_summary = masked_summary(
                series["altitude_values"], series["altitude_mask"]
            )
            writer.writerow(
                [
                    session_count,
                    dataset,
                    session_key,
                    user_index,
                    sport_code,
                    series["session_start_time"],
                    series["session_end_time"],
                    series["session_end_time"] - series["session_start_time"],
                    *hr_summary,
                    *speed_summary,
                    *altitude_summary,
                ]
            )
            for row in rows:
                start, end = context_positions(
                    float(row["origin_time"]),
                    int(series["grid_start_bin"]),
                    int(series["n_bins"]),
                )
                value_channels = (
                    series["hr_values"],
                    series["speed_values"],
                    series["altitude_values"],
                )
                mask_channels = (
                    series["hr_mask"],
                    series["speed_mask"],
                    series["altitude_mask"],
                )
                for channel, (values, mask) in enumerate(zip(value_channels, mask_channels)):
                    arrays["sequence_values"][row_index, :, channel] = np.asarray(
                        values[start:end], dtype=np.float16
                    )
                    arrays["sequence_masks"][row_index, :, channel] = np.asarray(
                        mask[start:end], dtype=np.uint8
                    )
                valid_hr_bins = int(arrays["sequence_masks"][row_index, :, 0].sum())
                if valid_hr_bins != int(row["context_valid_bins"]):
                    mask_mismatches += 1
                targets = (
                    float(row["target_hr_60"]),
                    float(row["target_hr_180"]),
                    float(row["target_hr_300"]),
                )
                if not all(30 <= value <= 240 and math.isfinite(value) for value in targets):
                    target_range_failures += 1
                arrays["targets"][row_index] = targets
                arrays["origin_offset_seconds"][row_index] = float(
                    row["origin_offset_seconds"]
                )
                arrays["origin_time"][row_index] = float(row["origin_time"])
                arrays["dataset_code"][row_index] = 0 if dataset == "Endomondo" else 1
                arrays["sport_code"][row_index] = sport_code
                arrays["user_index"][row_index] = user_index
                arrays["session_index"][row_index] = session_count
                arrays["evaluation_origin"][row_index] = int(row["evaluation_origin"])
                arrays["unseen_user_partition"][row_index] = PARTITION_CODES[
                    str(row["unseen_user_partition"])
                ]
                arrays["temporal_partition"][row_index] = PARTITION_CODES[
                    str(row["within_user_temporal_partition"])
                ]
                arrays["joint_user_partition"][row_index] = PARTITION_CODES[
                    str(row["joint_shift_user_partition"])
                ]
                arrays["sport_shift_candidate"][row_index] = int(
                    row["sport_shift_candidate"]
                )
                arrays["primary_external_partition"][row_index] = EXTERNAL_CODES[
                    str(row["primary_external_partition"])
                ]
                row_index += 1
            session_count += 1
            if session_count % 1000 == 0:
                print(
                    f"Model arrays: {session_count:,} sessions; {row_index:,}/{expected_rows:,} rows",
                    flush=True,
                )
    for memmap in arrays.values():
        memmap.flush()
    user_rows = [
        {"user_index": index, "dataset": key[0], "user_id": key[1]}
        for key, index in sorted(user_mapping.items(), key=lambda item: item[1])
    ]
    with (output_dir / "users.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_index", "dataset", "user_id"])
        writer.writeheader()
        writer.writerows(user_rows)
    file_bytes = {
        path.name: path.stat().st_size
        for path in sorted(output_dir.glob("*.npy"))
    }
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "array_version": ARRAY_VERSION,
        "origin_selection": "all Endomondo origins plus GoldenCheetah evaluation origins",
        "expected_rows": expected_rows,
        "rows_written": row_index,
        "sessions_written": session_count,
        "users_written": len(user_mapping),
        "channels": list(CHANNELS),
        "context_bins": CONTEXT_BINS,
        "grid_seconds": GRID_SECONDS,
        "value_dtype": "float16",
        "mask_dtype": "uint8",
        "sport_codes": SPORT_CODES,
        "partition_codes": PARTITION_CODES,
        "external_codes": EXTERNAL_CODES,
        "context_mask_mismatches": mask_mismatches,
        "missing_feature_series": missing_series,
        "target_range_failures": target_range_failures,
        "array_file_bytes": file_bytes,
    }
    payload["all_assertions_pass"] = (
        row_index == expected_rows
        and mask_mismatches == 0
        and missing_series == 0
        and target_range_failures == 0
    )
    origins.close()
    features.close()
    atomic_json(output_dir / "metadata.json", payload)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact model-ready NumPy arrays.")
    parser.add_argument("--origins", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_arrays(args.origins, args.features, args.output_dir),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
