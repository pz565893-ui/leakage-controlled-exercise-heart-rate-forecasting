from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import struct
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


AUDIT_VERSION = "0.20.0"
EXACT_FINGERPRINT_VERSION = "normalized-10s-float32-v1"
PROFILE_FINGERPRINT_VERSION = "trimmed-normalized-profile-v1"

MIN_PROFILE_SPAN_BINS = 180
MIN_PROFILE_COVERAGE = 0.80
MAX_PROFILE_MISSING_GAP_BINS = 6
MIN_PROFILE_HR_STD_BPM = 5.0
PROFILE_POINTS = 48
PROFILE_QUANTIZATION_BPM = 4.0
PROFILE_QUANTIZATION_OFFSETS_BPM = (0.0, 2.0)
PROFILE_CROPS_BINS = ((0, 0), (1, 0), (0, 1))
MAX_DURATION_RELATIVE_DIFFERENCE = 0.05

LSH_RANDOM_SEED = 20260722
LSH_TABLES = 12
LSH_BITS_PER_TABLE = 24
LSH_PREFILTER_MIN_HR_PEARSON_R = 0.98
LSH_PREFILTER_MAX_HR_MAE_BPM = 4.0
LSH_PREFILTER_MAX_HR_P95_ABS_ERROR_BPM = 8.0

VERIFICATION_POINTS = 256
MAX_HR_MAE_BPM = 2.0
MAX_HR_P95_ABS_ERROR_BPM = 4.0
MIN_HR_PEARSON_R = 0.99
MIN_AUXILIARY_COVERAGE = 0.70
MAX_SPEED_MAE_KMH = 1.5
MIN_SPEED_PEARSON_R = 0.98
MAX_CENTERED_ALTITUDE_MAE_M = 5.0
MIN_ALTITUDE_PEARSON_R = 0.98


@dataclass(frozen=True)
class ProfileSummary:
    eligible: bool
    reason: str
    first_observed_bin: int
    last_observed_bin: int
    span_bins: int
    valid_bins: int
    coverage: float
    max_missing_gap_bins: int
    hr_std_bpm: float


@dataclass(frozen=True)
class ProfileVariant:
    digest: bytes
    crop_left_bins: int
    crop_right_bins: int
    quantization_offset_bpm: float
    span_bins: int


@dataclass(frozen=True)
class VariantReference:
    session_key: str
    crop_left_bins: int
    crop_right_bins: int
    quantization_offset_bpm: float
    span_bins: int


@dataclass
class DecodedSeries:
    session_key: str
    grid_seconds: int
    n_bins: int
    hr_values: np.ndarray
    hr_mask: np.ndarray
    altitude_values: np.ndarray
    altitude_mask: np.ndarray
    speed_values: np.ndarray
    speed_mask: np.ndarray


SERIES_QUERY = """
SELECT session_key, grid_seconds, n_bins,
       hr_values_zlib, hr_mask_zlib,
       altitude_values_zlib, altitude_mask_zlib,
       speed_values_zlib, speed_mask_zlib
FROM session_series
WHERE dataset = ?
ORDER BY session_key
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def decode_float32(blob: bytes, expected_length: int) -> np.ndarray:
    values = np.frombuffer(zlib.decompress(blob), dtype="<f4")
    if values.size != expected_length:
        raise ValueError("decoded float32 array has the wrong length")
    return values


def decode_uint8(blob: bytes, expected_length: int) -> np.ndarray:
    values = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
    if values.size != expected_length:
        raise ValueError("decoded uint8 mask has the wrong length")
    return values


def decode_series_row(row: Sequence[object]) -> DecodedSeries:
    session_key = str(row[0])
    grid_seconds = int(row[1])
    n_bins = int(row[2])
    return DecodedSeries(
        session_key=session_key,
        grid_seconds=grid_seconds,
        n_bins=n_bins,
        hr_values=decode_float32(row[3], n_bins),
        hr_mask=decode_uint8(row[4], n_bins),
        altitude_values=decode_float32(row[5], n_bins),
        altitude_mask=decode_uint8(row[6], n_bins),
        speed_values=decode_float32(row[7], n_bins),
        speed_mask=decode_uint8(row[8], n_bins),
    )


def update_framed(digest: "hashlib._Hash", label: bytes, body: bytes) -> None:
    digest.update(struct.pack("<I", len(label)))
    digest.update(label)
    digest.update(struct.pack("<Q", len(body)))
    digest.update(body)


def exact_normalized_signal_digest(
    grid_seconds: int,
    n_bins: int,
    hr_values_blob: bytes,
    hr_mask_blob: bytes,
    altitude_values_blob: bytes,
    altitude_mask_blob: bytes,
    speed_values_blob: bytes,
    speed_mask_blob: bytes,
) -> str:
    """Hash common-grid signal content while excluding absolute timestamps and IDs."""
    digest = hashlib.sha256()
    digest.update(EXACT_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(struct.pack("<II", grid_seconds, n_bins))
    fields = (
        (b"hr_values_f32_le", zlib.decompress(hr_values_blob)),
        (b"hr_mask_u8", zlib.decompress(hr_mask_blob)),
        (b"altitude_values_f32_le", zlib.decompress(altitude_values_blob)),
        (b"altitude_mask_u8", zlib.decompress(altitude_mask_blob)),
        (b"speed_values_f32_le", zlib.decompress(speed_values_blob)),
        (b"speed_mask_u8", zlib.decompress(speed_mask_blob)),
    )
    for label, body in fields:
        update_framed(digest, label, body)
    return digest.hexdigest()


def exact_digest_from_decoded(series: DecodedSeries) -> str:
    digest = hashlib.sha256()
    digest.update(EXACT_FINGERPRINT_VERSION.encode("ascii"))
    digest.update(struct.pack("<II", series.grid_seconds, series.n_bins))
    fields = (
        (b"hr_values_f32_le", series.hr_values.astype("<f4", copy=False).tobytes()),
        (b"hr_mask_u8", series.hr_mask.astype(np.uint8, copy=False).tobytes()),
        (
            b"altitude_values_f32_le",
            series.altitude_values.astype("<f4", copy=False).tobytes(),
        ),
        (
            b"altitude_mask_u8",
            series.altitude_mask.astype(np.uint8, copy=False).tobytes(),
        ),
        (b"speed_values_f32_le", series.speed_values.astype("<f4", copy=False).tobytes()),
        (b"speed_mask_u8", series.speed_mask.astype(np.uint8, copy=False).tobytes()),
    )
    for label, body in fields:
        update_framed(digest, label, body)
    return digest.hexdigest()


def summarize_hr_profile(values: np.ndarray, mask: np.ndarray) -> ProfileSummary:
    observed = np.flatnonzero(mask.astype(bool))
    if observed.size == 0:
        return ProfileSummary(False, "no_valid_hr", -1, -1, 0, 0, 0.0, 0, 0.0)
    first = int(observed[0])
    last = int(observed[-1])
    span = last - first + 1
    valid = int(observed.size)
    coverage = valid / span
    missing_gaps = np.diff(observed) - 1
    max_gap = int(missing_gaps.max()) if missing_gaps.size else 0
    hr_std = float(np.std(values[observed], ddof=0))
    if span < MIN_PROFILE_SPAN_BINS:
        reason = "span_below_minimum"
    elif coverage < MIN_PROFILE_COVERAGE:
        reason = "coverage_below_minimum"
    elif max_gap > MAX_PROFILE_MISSING_GAP_BINS:
        reason = "gap_above_maximum"
    elif not math.isfinite(hr_std) or hr_std < MIN_PROFILE_HR_STD_BPM:
        reason = "hr_variability_below_minimum"
    else:
        reason = "eligible"
    return ProfileSummary(
        eligible=reason == "eligible",
        reason=reason,
        first_observed_bin=first,
        last_observed_bin=last,
        span_bins=span,
        valid_bins=valid,
        coverage=coverage,
        max_missing_gap_bins=max_gap,
        hr_std_bpm=hr_std,
    )


def interpolate_profile(
    values: np.ndarray,
    mask: np.ndarray,
    summary: ProfileSummary,
    crop_left_bins: int,
    crop_right_bins: int,
    points: int,
) -> np.ndarray | None:
    start = summary.first_observed_bin + crop_left_bins
    end = summary.last_observed_bin - crop_right_bins
    if start >= end:
        return None
    positions = np.flatnonzero(mask[start : end + 1].astype(bool)) + start
    if positions.size < 2:
        return None
    targets = np.linspace(start, end, num=points, dtype=np.float64)
    return np.interp(targets, positions, values[positions]).astype(np.float64)


def profile_variants(
    values: np.ndarray,
    mask: np.ndarray,
    summary: ProfileSummary,
) -> list[ProfileVariant]:
    if not summary.eligible:
        return []
    variants: list[ProfileVariant] = []
    seen: set[bytes] = set()
    for crop_left, crop_right in PROFILE_CROPS_BINS:
        profile = interpolate_profile(
            values,
            mask,
            summary,
            crop_left,
            crop_right,
            PROFILE_POINTS,
        )
        if profile is None:
            continue
        for offset in PROFILE_QUANTIZATION_OFFSETS_BPM:
            quantized = np.rint(
                (profile + offset) / PROFILE_QUANTIZATION_BPM
            ).astype("<i2")
            digest = hashlib.sha256()
            digest.update(PROFILE_FINGERPRINT_VERSION.encode("ascii"))
            digest.update(struct.pack("<I", PROFILE_POINTS))
            digest.update(quantized.tobytes())
            fingerprint = digest.digest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            variants.append(
                ProfileVariant(
                    digest=fingerprint,
                    crop_left_bins=crop_left,
                    crop_right_bins=crop_right,
                    quantization_offset_bpm=offset,
                    span_bins=summary.span_bins,
                )
            )
    return variants


def make_lsh_projections() -> np.ndarray:
    generator = np.random.default_rng(LSH_RANDOM_SEED)
    projections = generator.standard_normal(
        (LSH_TABLES, LSH_BITS_PER_TABLE, PROFILE_POINTS)
    ).astype(np.float32)
    norms = np.linalg.norm(projections, axis=2, keepdims=True)
    return projections / norms


def lsh_signatures(profile: np.ndarray, projections: np.ndarray) -> list[int]:
    if profile.shape != (PROFILE_POINTS,):
        raise ValueError("LSH profile has the wrong shape")
    standard_deviation = float(np.std(profile))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("LSH profile must have positive finite variability")
    normalized = ((profile - np.mean(profile)) / standard_deviation).astype(np.float32)
    bits = np.einsum("tbp,p->tb", projections, normalized) >= 0.0
    powers = np.left_shift(
        np.uint32(1), np.arange(LSH_BITS_PER_TABLE, dtype=np.uint32)
    )
    return [int(value) for value in np.sum(bits.astype(np.uint32) * powers, axis=1)]


def trimmed_exact_hr_digest(
    values: np.ndarray, mask: np.ndarray, summary: ProfileSummary
) -> str:
    if not summary.eligible:
        raise ValueError("trimmed HR digest requires an eligible profile")
    start = summary.first_observed_bin
    stop = summary.last_observed_bin + 1
    digest = hashlib.sha256()
    digest.update(b"trimmed-exact-hr-f32-v1")
    digest.update(struct.pack("<I", stop - start))
    update_framed(
        digest,
        b"hr_values_f32_le",
        values[start:stop].astype("<f4", copy=False).tobytes(),
    )
    update_framed(
        digest,
        b"hr_mask_u8",
        mask[start:stop].astype(np.uint8, copy=False).tobytes(),
    )
    return digest.hexdigest()


def relative_duration_difference(span_a: int, span_b: int) -> float:
    return abs(span_a - span_b) / max(span_a, span_b)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 2:
        return math.nan
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def resample_auxiliary(
    values: np.ndarray,
    mask: np.ndarray,
    start: int,
    end: int,
    points: int,
) -> tuple[np.ndarray | None, float]:
    if start >= end:
        return None, 0.0
    local_mask = mask[start : end + 1].astype(bool)
    coverage = float(np.mean(local_mask))
    positions = np.flatnonzero(local_mask) + start
    if coverage < MIN_AUXILIARY_COVERAGE or positions.size < 2:
        return None, coverage
    targets = np.linspace(start, end, num=points, dtype=np.float64)
    profile = np.interp(targets, positions, values[positions]).astype(np.float64)
    return profile, coverage


def verify_near_candidate(
    endomondo: DecodedSeries,
    golden: DecodedSeries,
    matches: Sequence[tuple[int, int, float, int, int, float]],
) -> dict[str, object]:
    endo_summary = summarize_hr_profile(endomondo.hr_values, endomondo.hr_mask)
    golden_summary = summarize_hr_profile(golden.hr_values, golden.hr_mask)
    best: dict[str, object] | None = None
    for (
        endo_left,
        endo_right,
        endo_offset,
        golden_left,
        golden_right,
        golden_offset,
    ) in matches:
        endo_profile = interpolate_profile(
            endomondo.hr_values,
            endomondo.hr_mask,
            endo_summary,
            endo_left,
            endo_right,
            VERIFICATION_POINTS,
        )
        golden_profile = interpolate_profile(
            golden.hr_values,
            golden.hr_mask,
            golden_summary,
            golden_left,
            golden_right,
            VERIFICATION_POINTS,
        )
        if endo_profile is None or golden_profile is None:
            continue
        difference = np.abs(endo_profile - golden_profile)
        candidate = {
            "endo_crop_left_bins": endo_left,
            "endo_crop_right_bins": endo_right,
            "endo_quantization_offset_bpm": endo_offset,
            "golden_crop_left_bins": golden_left,
            "golden_crop_right_bins": golden_right,
            "golden_quantization_offset_bpm": golden_offset,
            "hr_mae_bpm": float(np.mean(difference)),
            "hr_p95_abs_error_bpm": float(np.percentile(difference, 95)),
            "hr_pearson_r": safe_correlation(endo_profile, golden_profile),
        }
        score = (
            float(candidate["hr_mae_bpm"]),
            float(candidate["hr_p95_abs_error_bpm"]),
            -float(candidate["hr_pearson_r"])
            if math.isfinite(float(candidate["hr_pearson_r"]))
            else math.inf,
        )
        if best is None or score < best["_score"]:
            candidate["_score"] = score
            best = candidate
    if best is None:
        raise ValueError("no matched profile variant could be verified")

    endo_start = endo_summary.first_observed_bin + int(best["endo_crop_left_bins"])
    endo_end = endo_summary.last_observed_bin - int(best["endo_crop_right_bins"])
    golden_start = golden_summary.first_observed_bin + int(best["golden_crop_left_bins"])
    golden_end = golden_summary.last_observed_bin - int(best["golden_crop_right_bins"])

    endo_speed, endo_speed_coverage = resample_auxiliary(
        endomondo.speed_values,
        endomondo.speed_mask,
        endo_start,
        endo_end,
        VERIFICATION_POINTS,
    )
    golden_speed, golden_speed_coverage = resample_auxiliary(
        golden.speed_values,
        golden.speed_mask,
        golden_start,
        golden_end,
        VERIFICATION_POINTS,
    )
    if endo_speed is not None and golden_speed is not None:
        speed_mae = float(np.mean(np.abs(endo_speed - golden_speed)))
        speed_r = safe_correlation(endo_speed, golden_speed)
    else:
        speed_mae = math.nan
        speed_r = math.nan
    speed_support = (
        math.isfinite(speed_mae)
        and speed_mae <= MAX_SPEED_MAE_KMH
        and math.isfinite(speed_r)
        and speed_r >= MIN_SPEED_PEARSON_R
    )

    endo_altitude, endo_altitude_coverage = resample_auxiliary(
        endomondo.altitude_values,
        endomondo.altitude_mask,
        endo_start,
        endo_end,
        VERIFICATION_POINTS,
    )
    golden_altitude, golden_altitude_coverage = resample_auxiliary(
        golden.altitude_values,
        golden.altitude_mask,
        golden_start,
        golden_end,
        VERIFICATION_POINTS,
    )
    if endo_altitude is not None and golden_altitude is not None:
        centered_endo = endo_altitude - np.median(endo_altitude)
        centered_golden = golden_altitude - np.median(golden_altitude)
        altitude_mae = float(np.mean(np.abs(centered_endo - centered_golden)))
        altitude_r = safe_correlation(centered_endo, centered_golden)
    else:
        altitude_mae = math.nan
        altitude_r = math.nan
    altitude_support = (
        math.isfinite(altitude_mae)
        and altitude_mae <= MAX_CENTERED_ALTITUDE_MAE_M
        and math.isfinite(altitude_r)
        and altitude_r >= MIN_ALTITUDE_PEARSON_R
    )

    duration_difference = relative_duration_difference(
        endo_summary.span_bins, golden_summary.span_bins
    )
    hr_pass = (
        duration_difference <= MAX_DURATION_RELATIVE_DIFFERENCE
        and float(best["hr_mae_bpm"]) <= MAX_HR_MAE_BPM
        and float(best["hr_p95_abs_error_bpm"]) <= MAX_HR_P95_ABS_ERROR_BPM
        and math.isfinite(float(best["hr_pearson_r"]))
        and float(best["hr_pearson_r"]) >= MIN_HR_PEARSON_R
    )
    auxiliary_support = bool(speed_support or altitude_support)
    verification_status = (
        "near_candidate_hr_plus_auxiliary"
        if hr_pass and auxiliary_support
        else "near_candidate_hr_only"
        if hr_pass
        else "screened_out_after_continuous_verification"
    )
    best.pop("_score")
    best.update(
        {
            "verification_status": verification_status,
            "duration_relative_difference": duration_difference,
            "endo_span_bins": endo_summary.span_bins,
            "golden_span_bins": golden_summary.span_bins,
            "endo_hr_coverage": endo_summary.coverage,
            "golden_hr_coverage": golden_summary.coverage,
            "endo_hr_std_bpm": endo_summary.hr_std_bpm,
            "golden_hr_std_bpm": golden_summary.hr_std_bpm,
            "endo_speed_coverage": endo_speed_coverage,
            "golden_speed_coverage": golden_speed_coverage,
            "speed_mae_kmh": speed_mae,
            "speed_pearson_r": speed_r,
            "speed_support": speed_support,
            "endo_altitude_coverage": endo_altitude_coverage,
            "golden_altitude_coverage": golden_altitude_coverage,
            "centered_altitude_mae_m": altitude_mae,
            "altitude_pearson_r": altitude_r,
            "altitude_support": altitude_support,
            "auxiliary_support": auxiliary_support,
        }
    )
    return best


def connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=120)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -200000")
    return connection


def iter_series_rows(
    connection: sqlite3.Connection, dataset: str
) -> Iterator[Sequence[object]]:
    yield from connection.execute(SERIES_QUERY, (dataset,))


def fetch_series(
    connection: sqlite3.Connection, dataset: str, session_key: str
) -> DecodedSeries:
    row = connection.execute(
        SERIES_QUERY.replace("ORDER BY session_key", "AND session_key = ?"),
        (dataset, session_key),
    ).fetchone()
    if row is None:
        raise KeyError(f"missing series: {dataset}/{session_key}")
    return decode_series_row(row)


def load_manifest(path: Path, dataset: str) -> dict[str, dict[str, str]]:
    key_column = "record_index" if dataset == "Endomondo" else "csv_relative_path"
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row[key_column]
            if key in result:
                raise ValueError(f"duplicate session key in {path}: {key}")
            result[key] = row
    return result


def manifest_fields(
    endomondo_manifest: dict[str, dict[str, str]],
    golden_manifest: dict[str, dict[str, str]],
    endomondo_key: str,
    golden_key: str,
) -> dict[str, object]:
    endomondo = endomondo_manifest[endomondo_key]
    golden = golden_manifest[golden_key]

    def pseudonym(label: str, value: str) -> str:
        digest = hashlib.sha256()
        digest.update(b"cross-source-duplicate-audit-v0.20.0\x00")
        digest.update(label.encode("ascii"))
        digest.update(b"\x00")
        digest.update(value.encode("utf-8"))
        return digest.hexdigest()[:20]

    return {
        "endomondo_session_ref": pseudonym("endomondo_session", endomondo_key),
        "endomondo_user_ref": pseudonym(
            "endomondo_user", endomondo.get("user_id", "")
        ),
        "endomondo_sport_family": endomondo.get("sport_family", ""),
        "endomondo_unseen_user_partition": endomondo.get(
            "unseen_user_partition", ""
        ),
        "endomondo_temporal_partition": endomondo.get(
            "within_user_temporal_partition", ""
        ),
        "endomondo_joint_shift_partition": endomondo.get(
            "joint_shift_user_partition", ""
        ),
        "golden_session_ref": pseudonym("golden_session", golden_key),
        "golden_user_ref": pseudonym("golden_user", golden.get("user_id", "")),
        "golden_sport_family": golden.get("sport_family", ""),
        "golden_primary_external_partition": golden.get(
            "primary_external_partition", ""
        ),
        "golden_secondary_adaptation_partition": golden.get(
            "secondary_adaptation_partition", ""
        ),
        "same_sport_family": endomondo.get("sport_family", "")
        == golden.get("sport_family", ""),
    }


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def clean_for_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(item) for item in value]
    return value


def audit(
    database: Path,
    endomondo_manifest_path: Path,
    golden_manifest_path: Path,
    exact_csv: Path,
    near_csv: Path,
    json_output: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    endomondo_manifest = load_manifest(endomondo_manifest_path, "Endomondo")
    golden_manifest = load_manifest(golden_manifest_path, "GoldenCheetah")
    connection = connect_read_only(database)
    try:
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        source_counts = dict(
            connection.execute(
                "SELECT dataset, COUNT(*) FROM session_series GROUP BY dataset"
            )
        )
        grid_seconds_values = [
            int(row[0])
            for row in connection.execute(
                "SELECT DISTINCT grid_seconds FROM session_series ORDER BY grid_seconds"
            )
        ]

        exact_index: dict[str, list[str]] = defaultdict(list)
        exact_hr_index: dict[str, list[str]] = defaultdict(list)
        profile_index: dict[bytes, list[VariantReference]] = defaultdict(list)
        lsh_projections = make_lsh_projections()
        lsh_projection_sha256 = hashlib.sha256(lsh_projections.tobytes()).hexdigest()
        lsh_index: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
        endomondo_base_profiles: dict[str, np.ndarray] = {}
        endomondo_profile_reasons: Counter[str] = Counter()
        endomondo_profile_variants = 0
        endomondo_keys: set[str] = set()

        for index, row in enumerate(iter_series_rows(connection, "Endomondo"), start=1):
            session_key = str(row[0])
            endomondo_keys.add(session_key)
            exact_digest = exact_normalized_signal_digest(
                int(row[1]), int(row[2]), *row[3:]
            )
            exact_index[exact_digest].append(session_key)
            decoded = decode_series_row(row)
            summary = summarize_hr_profile(decoded.hr_values, decoded.hr_mask)
            endomondo_profile_reasons[summary.reason] += 1
            if summary.eligible:
                base_profile = interpolate_profile(
                    decoded.hr_values,
                    decoded.hr_mask,
                    summary,
                    0,
                    0,
                    PROFILE_POINTS,
                )
                if base_profile is None:
                    raise AssertionError("eligible profile could not be interpolated")
                endomondo_base_profiles[session_key] = base_profile.astype(np.float32)
                for table_id, signature in enumerate(
                    lsh_signatures(base_profile, lsh_projections)
                ):
                    lsh_index[(table_id, signature)].append(
                        (session_key, summary.span_bins)
                    )
                exact_hr_index[
                    trimmed_exact_hr_digest(
                        decoded.hr_values, decoded.hr_mask, summary
                    )
                ].append(session_key)
                for variant in profile_variants(
                    decoded.hr_values, decoded.hr_mask, summary
                ):
                    profile_index[variant.digest].append(
                        VariantReference(
                            session_key=session_key,
                            crop_left_bins=variant.crop_left_bins,
                            crop_right_bins=variant.crop_right_bins,
                            quantization_offset_bpm=variant.quantization_offset_bpm,
                            span_bins=variant.span_bins,
                        )
                    )
                    endomondo_profile_variants += 1
            if index % 25_000 == 0:
                print(f"Endomondo normalized sessions fingerprinted: {index:,}", flush=True)

        exact_pairs: list[dict[str, object]] = []
        exact_hr_pairs: list[dict[str, object]] = []
        candidate_matches: dict[
            tuple[str, str], set[tuple[int, int, float, int, int, float]]
        ] = defaultdict(set)
        candidate_methods: dict[tuple[str, str], set[str]] = defaultdict(set)
        golden_profile_reasons: Counter[str] = Counter()
        golden_profile_variants = 0
        golden_keys: set[str] = set()
        raw_profile_signature_cross_products = 0
        raw_lsh_bucket_cross_products = 0
        unique_lsh_pairs_after_duration_filter: set[tuple[str, str]] = set()
        lsh_pairs_passing_continuous_prefilter = 0

        for index, row in enumerate(iter_series_rows(connection, "GoldenCheetah"), start=1):
            golden_key = str(row[0])
            golden_keys.add(golden_key)
            exact_digest = exact_normalized_signal_digest(
                int(row[1]), int(row[2]), *row[3:]
            )
            for endomondo_key in exact_index.get(exact_digest, []):
                exact_pairs.append(
                    {
                        "match_type": "all_signal_float32_exact",
                        "interpretation": "confirmed_exact_normalized_signal_match",
                        "fingerprint_sha256": exact_digest,
                        **manifest_fields(
                            endomondo_manifest,
                            golden_manifest,
                            endomondo_key,
                            golden_key,
                        ),
                    }
                )

            decoded = decode_series_row(row)
            summary = summarize_hr_profile(decoded.hr_values, decoded.hr_mask)
            golden_profile_reasons[summary.reason] += 1
            if summary.eligible:
                hr_digest = trimmed_exact_hr_digest(
                    decoded.hr_values, decoded.hr_mask, summary
                )
                for endomondo_key in exact_hr_index.get(hr_digest, []):
                    exact_hr_pairs.append(
                        {
                            "match_type": "hr_trimmed_float32_exact",
                            "interpretation": "exact_hr_subset_candidate_not_full_signal_confirmation",
                            "fingerprint_sha256": hr_digest,
                            **manifest_fields(
                                endomondo_manifest,
                                golden_manifest,
                                endomondo_key,
                                golden_key,
                            ),
                        }
                    )

                variants = profile_variants(
                    decoded.hr_values, decoded.hr_mask, summary
                )
                golden_profile_variants += len(variants)
                for variant in variants:
                    references = profile_index.get(variant.digest, [])
                    raw_profile_signature_cross_products += len(references)
                    for reference in references:
                        if (
                            relative_duration_difference(
                                reference.span_bins, variant.span_bins
                            )
                            > MAX_DURATION_RELATIVE_DIFFERENCE
                        ):
                            continue
                        candidate_matches[(reference.session_key, golden_key)].add(
                            (
                                reference.crop_left_bins,
                                reference.crop_right_bins,
                                reference.quantization_offset_bpm,
                                variant.crop_left_bins,
                                variant.crop_right_bins,
                                variant.quantization_offset_bpm,
                            )
                        )
                        candidate_methods[(reference.session_key, golden_key)].add(
                            "quantized_profile_equality"
                        )

                base_profile = interpolate_profile(
                    decoded.hr_values,
                    decoded.hr_mask,
                    summary,
                    0,
                    0,
                    PROFILE_POINTS,
                )
                if base_profile is None:
                    raise AssertionError("eligible profile could not be interpolated")
                for table_id, signature in enumerate(
                    lsh_signatures(base_profile, lsh_projections)
                ):
                    references = lsh_index.get((table_id, signature), [])
                    raw_lsh_bucket_cross_products += len(references)
                    for endomondo_key, endomondo_span_bins in references:
                        if (
                            relative_duration_difference(
                                endomondo_span_bins, summary.span_bins
                            )
                            > MAX_DURATION_RELATIVE_DIFFERENCE
                        ):
                            continue
                        pair_key = (endomondo_key, golden_key)
                        if pair_key in unique_lsh_pairs_after_duration_filter:
                            continue
                        unique_lsh_pairs_after_duration_filter.add(pair_key)
                        endomondo_profile = endomondo_base_profiles[endomondo_key]
                        absolute_error = np.abs(endomondo_profile - base_profile)
                        profile_mae = float(np.mean(absolute_error))
                        profile_p95 = float(np.percentile(absolute_error, 95))
                        profile_r = safe_correlation(endomondo_profile, base_profile)
                        if not (
                            profile_mae <= LSH_PREFILTER_MAX_HR_MAE_BPM
                            and profile_p95
                            <= LSH_PREFILTER_MAX_HR_P95_ABS_ERROR_BPM
                            and math.isfinite(profile_r)
                            and profile_r >= LSH_PREFILTER_MIN_HR_PEARSON_R
                        ):
                            continue
                        lsh_pairs_passing_continuous_prefilter += 1
                        candidate_matches[pair_key].add((0, 0, 0.0, 0, 0, 0.0))
                        candidate_methods[pair_key].add("random_hyperplane_lsh")
            if index % 10_000 == 0:
                print(f"GoldenCheetah normalized sessions fingerprinted: {index:,}", flush=True)

        missing_endomondo_manifest_keys = endomondo_keys - set(endomondo_manifest)
        missing_golden_manifest_keys = golden_keys - set(golden_manifest)
        if missing_endomondo_manifest_keys or missing_golden_manifest_keys:
            raise AssertionError("session-series keys are not fully covered by split manifests")

        near_rows: list[dict[str, object]] = []
        for index, ((endomondo_key, golden_key), matches) in enumerate(
            sorted(candidate_matches.items()), start=1
        ):
            endomondo_series = fetch_series(connection, "Endomondo", endomondo_key)
            golden_series = fetch_series(connection, "GoldenCheetah", golden_key)
            verification = verify_near_candidate(
                endomondo_series, golden_series, sorted(matches)
            )
            row = {
                "candidate_pair_id": index,
                "candidate_generation_methods": ";".join(
                    sorted(candidate_methods[(endomondo_key, golden_key)])
                ),
                "matched_profile_variant_pairs": len(matches),
                **manifest_fields(
                    endomondo_manifest,
                    golden_manifest,
                    endomondo_key,
                    golden_key,
                ),
                **verification,
            }
            near_rows.append(row)

        exact_rows = exact_pairs + exact_hr_pairs
        exact_fields = [
            "match_type",
            "interpretation",
            "fingerprint_sha256",
            "endomondo_session_ref",
            "endomondo_user_ref",
            "endomondo_sport_family",
            "endomondo_unseen_user_partition",
            "endomondo_temporal_partition",
            "endomondo_joint_shift_partition",
            "golden_session_ref",
            "golden_user_ref",
            "golden_sport_family",
            "golden_primary_external_partition",
            "golden_secondary_adaptation_partition",
            "same_sport_family",
        ]
        near_fields = [
            "candidate_pair_id",
            "verification_status",
            "candidate_generation_methods",
            "matched_profile_variant_pairs",
            "endomondo_session_ref",
            "endomondo_user_ref",
            "endomondo_sport_family",
            "endomondo_unseen_user_partition",
            "endomondo_temporal_partition",
            "endomondo_joint_shift_partition",
            "golden_session_ref",
            "golden_user_ref",
            "golden_sport_family",
            "golden_primary_external_partition",
            "golden_secondary_adaptation_partition",
            "same_sport_family",
            "duration_relative_difference",
            "endo_span_bins",
            "golden_span_bins",
            "endo_hr_coverage",
            "golden_hr_coverage",
            "endo_hr_std_bpm",
            "golden_hr_std_bpm",
            "endo_crop_left_bins",
            "endo_crop_right_bins",
            "endo_quantization_offset_bpm",
            "golden_crop_left_bins",
            "golden_crop_right_bins",
            "golden_quantization_offset_bpm",
            "hr_mae_bpm",
            "hr_p95_abs_error_bpm",
            "hr_pearson_r",
            "endo_speed_coverage",
            "golden_speed_coverage",
            "speed_mae_kmh",
            "speed_pearson_r",
            "speed_support",
            "endo_altitude_coverage",
            "golden_altitude_coverage",
            "centered_altitude_mae_m",
            "altitude_pearson_r",
            "altitude_support",
            "auxiliary_support",
        ]
        atomic_csv(exact_csv, exact_fields, exact_rows)
        atomic_csv(near_csv, near_fields, near_rows)

        status_counts = Counter(str(row["verification_status"]) for row in near_rows)

        def finite_metric_summary(field: str) -> dict[str, float] | None:
            values = np.asarray(
                [float(row[field]) for row in near_rows if math.isfinite(float(row[field]))],
                dtype=np.float64,
            )
            if values.size == 0:
                return None
            return {
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            }

        exact_full_groups = len(
            {str(row["fingerprint_sha256"]) for row in exact_pairs}
        )
        exact_hr_groups = len(
            {str(row["fingerprint_sha256"]) for row in exact_hr_pairs}
        )
        all_assertions_pass = (
            integrity == "ok"
            and source_counts.get("Endomondo", 0) > 0
            and source_counts.get("GoldenCheetah", 0) > 0
            and grid_seconds_values == [10]
            and not missing_endomondo_manifest_keys
            and not missing_golden_manifest_keys
            and len(exact_pairs)
            == sum(row["match_type"] == "all_signal_float32_exact" for row in exact_rows)
            and len(near_rows) == len(candidate_matches)
        )
        if not all_assertions_pass:
            raise AssertionError("cross-source duplicate audit assertions failed")

        summary: dict[str, object] = {
            "generated_at_utc": utc_now(),
            "audit_version": AUDIT_VERSION,
            "scope": {
                "purpose": (
                    "detect possible Endomondo-to-GoldenCheetah session overlap in the "
                    "processed modeling universe before external validation"
                ),
                "grain": "one processed session-series row",
                "source_session_counts": source_counts,
                "grid_seconds_values": grid_seconds_values,
                "database_feature_version": metadata.get("feature_version"),
                "database_value_encoding": metadata.get("value_encoding"),
                "database_mask_encoding": metadata.get("mask_encoding"),
            },
            "exact_normalized_signal_audit": {
                "fingerprint_version": EXACT_FINGERPRINT_VERSION,
                "fields": [
                    "heart-rate float32 values and masks",
                    "altitude float32 values and masks",
                    "speed float32 values and masks",
                    "grid_seconds and n_bins",
                ],
                "excluded_from_fingerprint": [
                    "dataset label",
                    "session/user identifiers",
                    "absolute timestamps and grid_start_bin",
                    "sport labels and partitions",
                ],
                "cross_source_hash_groups": exact_full_groups,
                "cross_source_pairs": len(exact_pairs),
                "endomondo_sessions_in_pairs": len(
                    {str(row["endomondo_session_ref"]) for row in exact_pairs}
                ),
                "goldencheetah_sessions_in_pairs": len(
                    {str(row["golden_session_ref"]) for row in exact_pairs}
                ),
                "interpretation": (
                    "Cryptographic equality of the full normalized signal payload is treated "
                    "as a confirmed exact normalized-signal match."
                ),
            },
            "exact_hr_subset_screen": {
                "cross_source_hash_groups": exact_hr_groups,
                "cross_source_pairs": len(exact_hr_pairs),
                "interpretation": (
                    "Exact equality of only the trimmed HR values/mask is a candidate screen, "
                    "not confirmation that the full sessions are duplicates."
                ),
            },
            "near_duplicate_candidate_screen": {
                "fingerprint_version": PROFILE_FINGERPRINT_VERSION,
                "eligibility_thresholds": {
                    "minimum_span_bins": MIN_PROFILE_SPAN_BINS,
                    "minimum_span_minutes": MIN_PROFILE_SPAN_BINS * 10 / 60,
                    "minimum_hr_coverage": MIN_PROFILE_COVERAGE,
                    "maximum_missing_gap_bins": MAX_PROFILE_MISSING_GAP_BINS,
                    "minimum_hr_std_bpm": MIN_PROFILE_HR_STD_BPM,
                },
                "endomondo_profile_reasons": dict(endomondo_profile_reasons),
                "goldencheetah_profile_reasons": dict(golden_profile_reasons),
                "profile_points": PROFILE_POINTS,
                "quantization_bpm": PROFILE_QUANTIZATION_BPM,
                "quantization_offsets_bpm": list(PROFILE_QUANTIZATION_OFFSETS_BPM),
                "endpoint_crop_variants_bins": [list(item) for item in PROFILE_CROPS_BINS],
                "endomondo_unique_profile_variants_indexed": endomondo_profile_variants,
                "goldencheetah_unique_profile_variants_queried": golden_profile_variants,
                "raw_matched_signature_cross_products": raw_profile_signature_cross_products,
                "quantized_signature_candidate_pairs_after_duration_filter": sum(
                    "quantized_profile_equality" in methods
                    for methods in candidate_methods.values()
                ),
                "random_hyperplane_lsh": {
                    "seed": LSH_RANDOM_SEED,
                    "tables": LSH_TABLES,
                    "bits_per_table": LSH_BITS_PER_TABLE,
                    "projection_sha256": lsh_projection_sha256,
                    "raw_bucket_cross_products": raw_lsh_bucket_cross_products,
                    "unique_pairs_after_duration_filter": len(
                        unique_lsh_pairs_after_duration_filter
                    ),
                    "continuous_prefilter": {
                        "minimum_hr_pearson_r": LSH_PREFILTER_MIN_HR_PEARSON_R,
                        "maximum_hr_mae_bpm": LSH_PREFILTER_MAX_HR_MAE_BPM,
                        "maximum_hr_p95_abs_error_bpm": (
                            LSH_PREFILTER_MAX_HR_P95_ABS_ERROR_BPM
                        ),
                    },
                    "pairs_passing_continuous_prefilter": (
                        lsh_pairs_passing_continuous_prefilter
                    ),
                },
                "unique_candidate_pairs_after_duration_filter": len(candidate_matches),
                "maximum_duration_relative_difference": MAX_DURATION_RELATIVE_DIFFERENCE,
                "continuous_verification_thresholds": {
                    "points": VERIFICATION_POINTS,
                    "maximum_hr_mae_bpm": MAX_HR_MAE_BPM,
                    "maximum_hr_p95_abs_error_bpm": MAX_HR_P95_ABS_ERROR_BPM,
                    "minimum_hr_pearson_r": MIN_HR_PEARSON_R,
                    "minimum_auxiliary_coverage": MIN_AUXILIARY_COVERAGE,
                    "maximum_speed_mae_kmh": MAX_SPEED_MAE_KMH,
                    "minimum_speed_pearson_r": MIN_SPEED_PEARSON_R,
                    "maximum_centered_altitude_mae_m": MAX_CENTERED_ALTITUDE_MAE_M,
                    "minimum_altitude_pearson_r": MIN_ALTITUDE_PEARSON_R,
                },
                "verification_status_counts": dict(status_counts),
                "continuous_verification_metric_summaries": {
                    "hr_mae_bpm": finite_metric_summary("hr_mae_bpm"),
                    "hr_p95_abs_error_bpm": finite_metric_summary(
                        "hr_p95_abs_error_bpm"
                    ),
                    "hr_pearson_r": finite_metric_summary("hr_pearson_r"),
                    "duration_relative_difference": finite_metric_summary(
                        "duration_relative_difference"
                    ),
                },
                "individual_final_threshold_pass_counts": {
                    "hr_mae": sum(
                        float(row["hr_mae_bpm"]) <= MAX_HR_MAE_BPM
                        for row in near_rows
                    ),
                    "hr_p95_abs_error": sum(
                        float(row["hr_p95_abs_error_bpm"])
                        <= MAX_HR_P95_ABS_ERROR_BPM
                        for row in near_rows
                    ),
                    "hr_pearson_r": sum(
                        math.isfinite(float(row["hr_pearson_r"]))
                        and float(row["hr_pearson_r"]) >= MIN_HR_PEARSON_R
                        for row in near_rows
                    ),
                    "all_hr_thresholds": sum(
                        str(row["verification_status"])
                        in {
                            "near_candidate_hr_only",
                            "near_candidate_hr_plus_auxiliary",
                        }
                        for row in near_rows
                    ),
                },
                "verified_hr_candidate_pairs": (
                    status_counts["near_candidate_hr_only"]
                    + status_counts["near_candidate_hr_plus_auxiliary"]
                ),
                "verified_hr_plus_auxiliary_candidate_pairs": status_counts[
                    "near_candidate_hr_plus_auxiliary"
                ],
                "interpretation": (
                    "The coarse fingerprint stage is an exhaustive equality join for the "
                    "declared deterministic profile signatures. Deterministic random-hyperplane "
                    "LSH adds a broader approximate-shape screen but is probabilistic rather "
                    "than exhaustive. All matches remain near-duplicate candidates unless the "
                    "full exact fingerprint also matches."
                ),
            },
            "risk_assessment": {
                "confirmed_cross_source_exact_overlap": len(exact_pairs) > 0,
                "high_confidence_near_overlap_candidate": status_counts[
                    "near_candidate_hr_plus_auxiliary"
                ]
                > 0,
                "external_validation_contamination_action": (
                    "Exclude or adjudicate every implicated GoldenCheetah session before "
                    "reporting external validation."
                    if len(exact_pairs) > 0
                    or status_counts["near_candidate_hr_plus_auxiliary"] > 0
                    else "No session exclusion is triggered by this declared audit."
                ),
            },
            "limitations": [
                (
                    "Exact fingerprints operate on the processed 10-second cache, not on raw "
                    "files; preprocessing collisions are theoretically possible but full "
                    "float32 values and masks across three signals make accidental equality "
                    "unlikely."
                ),
                (
                    "The near screen can miss duplicates with substantial cropping, clock drift, "
                    "long gaps, different HR smoothing, or differences larger than the declared "
                    "quantization/alignment variants; the LSH stage also has non-zero false-"
                    "negative probability."
                ),
                (
                    "HR-only similarity can occur by chance; only full exact matches are called "
                    "confirmed duplicates, and auxiliary agreement is reported separately."
                ),
                (
                    "A zero verified near-candidate count lowers but cannot prove absence of all "
                    "possible cross-platform duplicate recordings."
                ),
            ],
            "inputs": {
                "session_series_database": str(database),
                "session_series_database_sha256": sha256_file(database),
                "endomondo_split_manifest": str(endomondo_manifest_path),
                "endomondo_split_manifest_sha256": sha256_file(
                    endomondo_manifest_path
                ),
                "goldencheetah_split_manifest": str(golden_manifest_path),
                "goldencheetah_split_manifest_sha256": sha256_file(
                    golden_manifest_path
                ),
            },
            "outputs": {
                "exact_pair_csv": str(exact_csv),
                "near_candidate_csv": str(near_csv),
                "audit_json": str(json_output),
                "identifier_policy": (
                    "CSV session and user references are deterministic domain-separated "
                    "SHA-256 pseudonyms truncated to 20 hexadecimal characters; raw IDs and "
                    "file names are not serialized."
                ),
            },
            "assertions": {
                "sqlite_quick_check": integrity,
                "manifest_coverage_missing_endomondo_keys": len(
                    missing_endomondo_manifest_keys
                ),
                "manifest_coverage_missing_golden_keys": len(
                    missing_golden_manifest_keys
                ),
                "all_assertions_pass": all_assertions_pass,
            },
            "runtime_seconds": time.perf_counter() - started,
        }
        summary = clean_for_json(summary)  # type: ignore[assignment]
        atomic_json(json_output, summary)
        return summary
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact and high-precision near-duplicate exercise signals across "
            "Endomondo and GoldenCheetah."
        )
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--endomondo-manifest", type=Path, required=True)
    parser.add_argument("--goldencheetah-manifest", type=Path, required=True)
    parser.add_argument("--exact-csv", type=Path, required=True)
    parser.add_argument("--near-csv", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(
        database=args.database,
        endomondo_manifest_path=args.endomondo_manifest,
        golden_manifest_path=args.goldencheetah_manifest,
        exact_csv=args.exact_csv,
        near_csv=args.near_csv,
        json_output=args.json_output,
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
