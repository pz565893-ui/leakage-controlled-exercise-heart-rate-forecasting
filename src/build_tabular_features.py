from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TABULAR_VERSION = "0.7.0"
GRID_SECONDS = 10.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def channel_features(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    mask_float = mask.astype(np.float32, copy=False)
    count = mask_float.sum(axis=1)
    safe_count = np.maximum(count, 1.0)
    mean = (values * mask_float).sum(axis=1) / safe_count
    variance = (((values - mean[:, None]) ** 2) * mask_float).sum(axis=1) / safe_count
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    minimum = np.where(mask, values, np.inf).min(axis=1)
    maximum = np.where(mask, values, -np.inf).max(axis=1)
    first_index = mask.argmax(axis=1)
    last_index = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    first = np.take_along_axis(values, first_index[:, None], axis=1)[:, 0]
    last = np.take_along_axis(values, last_index[:, None], axis=1)[:, 0]
    time = np.arange(-(mask.shape[1] - 1), 1, dtype=np.float32) * GRID_SECONDS
    sum_x = (mask_float * time[None, :]).sum(axis=1)
    sum_xx = (mask_float * (time[None, :] ** 2)).sum(axis=1)
    sum_xy = (mask_float * values * time[None, :]).sum(axis=1)
    denominator = count * sum_xx - sum_x * sum_x
    slope = np.divide(
        count * sum_xy - sum_x * (values * mask_float).sum(axis=1),
        denominator,
        out=np.zeros_like(mean),
        where=np.abs(denominator) > 1e-8,
    )

    def recent_mean(n_bins: int) -> np.ndarray:
        recent_mask = mask_float[:, -n_bins:]
        recent_count = recent_mask.sum(axis=1)
        return np.divide(
            (values[:, -n_bins:] * recent_mask).sum(axis=1),
            recent_count,
            out=np.zeros_like(mean),
            where=recent_count > 0,
        )

    no_observation = count == 0
    for feature in (mean, standard_deviation, minimum, maximum, first, last, slope):
        feature[no_observation] = 0.0
    return np.column_stack(
        [
            last,
            mean,
            standard_deviation,
            minimum,
            maximum,
            last - first,
            slope,
            count / mask.shape[1],
            recent_mean(6),
            recent_mean(18),
        ]
    ).astype(np.float32, copy=False)


def feature_names() -> list[str]:
    statistics = (
        "last",
        "mean",
        "std",
        "min",
        "max",
        "last_minus_first",
        "slope_per_second",
        "coverage",
        "recent_60s_mean",
        "recent_180s_mean",
    )
    names = [
        f"{channel}__{statistic}"
        for channel in ("hr", "speed", "altitude")
        for statistic in statistics
    ]
    names.extend(["log1p_elapsed_seconds"])
    names.extend([f"sport_code_{code}" for code in range(8)])
    return names


def build(array_dir: Path, output_path: Path, chunk_size: int) -> dict[str, object]:
    sequence_values = np.load(array_dir / "sequence_values.npy", mmap_mode="r")
    sequence_masks = np.load(array_dir / "sequence_masks.npy", mmap_mode="r")
    elapsed = np.load(array_dir / "origin_offset_seconds.npy", mmap_mode="r")
    sport = np.load(array_dir / "sport_code.npy", mmap_mode="r")
    n_rows = sequence_values.shape[0]
    names = feature_names()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(n_rows, len(names))
    )
    for start in range(0, n_rows, chunk_size):
        end = min(n_rows, start + chunk_size)
        chunks = [
            channel_features(
                np.asarray(sequence_values[start:end, :, channel], dtype=np.float32),
                np.asarray(sequence_masks[start:end, :, channel], dtype=bool),
            )
            for channel in range(3)
        ]
        output[start:end, :30] = np.concatenate(chunks, axis=1)
        output[start:end, 30] = np.log1p(
            np.asarray(elapsed[start:end], dtype=np.float32)
        )
        sport_chunk = np.asarray(sport[start:end], dtype=np.int64)
        output[start:end, 31:] = np.eye(8, dtype=np.float32)[sport_chunk]
        if start == 0 or end % 500_000 < chunk_size:
            print(f"Tabular features: {end:,}/{n_rows:,}", flush=True)
    output.flush()
    finite_failures = 0
    for start in range(0, n_rows, chunk_size):
        end = min(n_rows, start + chunk_size)
        finite_failures += int((~np.isfinite(output[start:end])).sum())
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "tabular_version": TABULAR_VERSION,
        "rows": n_rows,
        "features": len(names),
        "feature_names": names,
        "dtype": "float32",
        "file": str(output_path),
        "file_bytes": output_path.stat().st_size,
        "nonfinite_values": finite_failures,
        "all_assertions_pass": finite_failures == 0,
    }
    metadata_path = output_path.with_suffix(".json")
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build causal tabular lag features.")
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    print(json.dumps(build(args.array_dir, args.output, args.chunk_size), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
