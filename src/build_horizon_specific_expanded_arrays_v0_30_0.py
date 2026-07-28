"""Build identifier-free model inputs for non-common target-eligible origins.

The original complete-three-target rows remain in the authoritative v0.6.0
arrays.  This builder stores only additional fixed-session origins for which at
least one horizon target is available but the complete-three-target rule fails.
It reconstructs inputs from the same raw HR target stream and v0.4.0 causal
10-second feature cache used by the primary models.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from build_forecast_origins import interpolate_target, valid_hr
from build_model_arrays import context_positions, load_session_series
from evaluate_horizon_specific_eligibility_v0_29_0 import (
    HORIZONS,
    SessionSpec,
    candidate_origins,
    context_is_eligible,
    iter_endomondo,
    iter_golden,
    load_session_specs,
)


VERSION = "0.30.0"
REGIME_BITS = {"strict_temporal": 1, "unseen_user": 2, "external": 4}
ARRAY_DTYPES = {
    "sequence_values": np.float16,
    "sequence_masks": np.uint8,
    "targets": np.float32,
    "target_mask": np.uint8,
    "origin_offset_seconds": np.float32,
    "origin_time": np.float64,
    "dataset_code": np.uint8,
    "sport_code": np.uint8,
    "user_index": np.int32,
    "session_index": np.int32,
    "regime_flags": np.uint8,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def valid_pairs(
    timestamps: Sequence[float], heart_rate: Sequence[float]
) -> tuple[list[float], list[float]] | None:
    if len(timestamps) != len(heart_rate) or len(timestamps) < 2:
        return None
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        return None
    pairs = [
        (float(timestamp), float(hr))
        for timestamp, hr in zip(timestamps, heart_rate)
        if valid_hr(float(hr))
    ]
    if len(pairs) < 2:
        return None
    return [item[0] for item in pairs], [item[1] for item in pairs]


def extra_origins(
    timestamps: Sequence[float], heart_rate: Sequence[float]
) -> list[tuple[float, tuple[float, float, float], tuple[int, int, int]]]:
    prepared = valid_pairs(timestamps, heart_rate)
    if prepared is None:
        return []
    times, values = prepared
    rows = []
    for origin in candidate_origins(timestamps):
        origin_time = float(origin)
        if not context_is_eligible(times, origin_time):
            continue
        results = [
            interpolate_target(times, values, origin_time + horizon)
            for horizon in HORIZONS
        ]
        mask = tuple(int(item is not None) for item in results)
        if not any(mask) or all(mask):
            continue
        targets = tuple(
            float(item[0]) if item is not None else math.nan for item in results
        )
        rows.append((origin_time, targets, mask))
    return rows


def session_regime_flags(spec: SessionSpec) -> int:
    return sum(REGIME_BITS[item] for item in spec.regimes)


def expected_added_counts(result_path: Path) -> dict[tuple[str, int], int]:
    import pandas as pd

    frame = pd.read_csv(result_path)
    frame = frame[frame.cohort == "horizon_specific"]
    reverse = {
        "within_user_temporal_test": "strict_temporal",
        "unseen_user_test": "unseen_user",
        "goldencheetah_frozen_external": "external",
    }
    return {
        (reverse[str(row.regime)], int(row.horizon_seconds)): int(
            row.added_origins_vs_common
        )
        for _, row in frame.iterrows()
    }


def save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    specs = load_session_specs(args.array_dir)
    feature_db = sqlite3.connect(args.session_series)
    collected: defaultdict[str, list[object]] = defaultdict(list)
    processed = defaultdict(int)
    context_mask_mismatches = 0
    invalid_target_values = 0

    streams = (
        iter_endomondo(args.endomondo_hr, specs),
        iter_golden(args.golden_root, specs),
    )
    for stream in streams:
        for spec, timestamps, heart_rate in stream:
            rows = extra_origins(timestamps, heart_rate)
            if rows:
                series = load_session_series(feature_db, spec.dataset, spec.session_key)
                prepared = valid_pairs(timestamps, heart_rate)
                if prepared is None:
                    raise AssertionError("extra origins emitted for an invalid session")
                raw_times, _ = prepared
                for origin_time, targets, target_mask in rows:
                    start, end = context_positions(
                        origin_time,
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
                    values = np.stack(
                        [
                            np.asarray(channel[start:end], dtype=np.float16)
                            for channel in value_channels
                        ],
                        axis=1,
                    )
                    masks = np.stack(
                        [
                            np.asarray(channel[start:end], dtype=np.uint8)
                            for channel in mask_channels
                        ],
                        axis=1,
                    )
                    context_start = origin_time - 300
                    left = bisect.bisect_left(raw_times, context_start)
                    right = bisect.bisect_right(raw_times, origin_time)
                    occupied = {
                        min(29, int(math.ceil((item - context_start) / 10) - 1))
                        for item in raw_times[left:right]
                        if 0 < item - context_start <= 300
                    }
                    if int(masks[:, 0].sum()) != len(occupied):
                        context_mask_mismatches += 1
                    for value, observed in zip(targets, target_mask):
                        if observed and not (math.isfinite(value) and 30 <= value <= 240):
                            invalid_target_values += 1

                    collected["sequence_values"].append(values)
                    collected["sequence_masks"].append(masks)
                    collected["targets"].append(targets)
                    collected["target_mask"].append(target_mask)
                    collected["origin_offset_seconds"].append(
                        origin_time - float(timestamps[0])
                    )
                    collected["origin_time"].append(origin_time)
                    collected["dataset_code"].append(
                        0 if spec.dataset == "Endomondo" else 1
                    )
                    collected["sport_code"].append(spec.sport_code)
                    collected["user_index"].append(spec.user_index)
                    collected["session_index"].append(spec.session_index)
                    collected["regime_flags"].append(session_regime_flags(spec))
            processed[spec.dataset] += 1
            total = sum(processed.values())
            if total % 1000 == 0:
                print(
                    f"sessions processed: {total:,}; extra origins: "
                    f"{len(collected['origin_time']):,}",
                    flush=True,
                )
    feature_db.close()

    arrays: dict[str, np.ndarray] = {}
    for name, dtype in ARRAY_DTYPES.items():
        if name in {"sequence_values", "sequence_masks"}:
            arrays[name] = np.stack(collected[name]).astype(dtype, copy=False)
        else:
            arrays[name] = np.asarray(collected[name], dtype=dtype)

    n_rows = len(arrays["origin_time"])
    if arrays["sequence_values"].shape != (n_rows, 30, 3):
        raise AssertionError("unexpected sequence-values shape")
    if arrays["sequence_masks"].shape != (n_rows, 30, 3):
        raise AssertionError("unexpected sequence-masks shape")
    if arrays["targets"].shape != (n_rows, 3):
        raise AssertionError("unexpected target shape")
    if arrays["target_mask"].shape != (n_rows, 3):
        raise AssertionError("unexpected target-mask shape")
    if np.any(arrays["target_mask"].sum(axis=1) == 0) or np.any(
        arrays["target_mask"].sum(axis=1) == 3
    ):
        raise AssertionError("extra arrays contain empty or common-three-target rows")
    composite = np.rec.fromarrays(
        [arrays["session_index"], arrays["origin_time"]],
        names=["session", "origin"],
    )
    duplicate_rows = n_rows - len(np.unique(composite))
    if duplicate_rows:
        raise AssertionError(f"duplicate extra origins: {duplicate_rows}")

    expected = expected_added_counts(args.v029_results)
    observed: dict[tuple[str, int], int] = {}
    for regime, bit in REGIME_BITS.items():
        regime_mask = (arrays["regime_flags"] & bit) != 0
        for position, horizon in enumerate(HORIZONS):
            observed[(regime, horizon)] = int(
                np.count_nonzero(regime_mask & (arrays["target_mask"][:, position] == 1))
            )
    count_checks = [
        {
            "regime": regime,
            "horizon_seconds": horizon,
            "observed": observed[(regime, horizon)],
            "expected": expected[(regime, horizon)],
            "pass": observed[(regime, horizon)] == expected[(regime, horizon)],
        }
        for regime in REGIME_BITS
        for horizon in HORIZONS
    ]
    if not all(item["pass"] for item in count_checks):
        raise AssertionError(f"v0.29 added-origin counts not reproduced: {count_checks}")
    if context_mask_mismatches or invalid_target_values:
        raise AssertionError(
            f"input reconstruction failures: masks={context_mask_mismatches}, "
            f"targets={invalid_target_values}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        save_array(args.output_dir / f"{name}.npy", array)
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(args.output_dir.glob("*.npy"))
    }
    audit: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "version": VERSION,
        "grain": "one additional fixed-session evaluation origin",
        "scope": "origins with one or two available targets; common-three-target rows excluded",
        "rows": n_rows,
        "sessions_processed": dict(processed),
        "regime_bits": REGIME_BITS,
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "array_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        "count_checks_against_v0_29": count_checks,
        "duplicate_session_origin_rows": duplicate_rows,
        "context_mask_mismatches": context_mask_mismatches,
        "invalid_observed_targets": invalid_target_values,
        "privacy": (
            "private inference input: contains internal user/session indices but no raw "
            "identifiers; excluded from public release"
        ),
        "sha256": hashes,
        "all_assertions_pass": True,
    }
    atomic_json(args.audit, audit)
    print(json.dumps(audit, indent=2))
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    result.add_argument(
        "--session-series",
        type=Path,
        default=Path("outputs/features/session_series_v0_4_0.sqlite"),
    )
    result.add_argument(
        "--endomondo-hr",
        type=Path,
        default=Path("../endomondoHR.json/endomondoHR.json"),
    )
    result.add_argument(
        "--golden-root", type=Path, default=Path("../GoldenCheetah_extracted")
    )
    result.add_argument(
        "--v029-results",
        type=Path,
        default=Path("outputs/results/horizon_specific_eligibility_v0_29_0.csv"),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/features/horizon_specific_extra_v0_30_0"),
    )
    result.add_argument(
        "--audit",
        type=Path,
        default=Path("outputs/audit/horizon_specific_extra_arrays_v0_30_0.json"),
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
