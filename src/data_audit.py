from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


USER_RE = re.compile(rb"'userId':\s*(\d+)")
GENDER_RE = re.compile(rb"'gender':\s*'([^']*)'")
SPORT_RE = re.compile(rb"'sport':\s*'([^']*)'")


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p10": quantile(values, 0.10),
        "median": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "max": max(values) if values else None,
    }


def decode_match(pattern: re.Pattern[bytes], line: bytes) -> str | None:
    match = pattern.search(line)
    if match is None:
        return None
    return match.group(1).decode("utf-8", errors="replace")


def extract_array_bytes(line: bytes, field: str) -> bytes | None:
    marker = f"'{field}': [".encode("ascii")
    start = line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = line.find(b"]", start)
    if end < 0:
        return None
    return line[start:end]


def parse_numeric_array(body: bytes | None, cast=float) -> list[float]:
    if body is None or not body.strip():
        return []
    values: list[float] = []
    for token in body.split(b","):
        token = token.strip()
        if not token or token in {b"None", b"nan", b"NaN"}:
            continue
        try:
            values.append(cast(token))
        except (TypeError, ValueError):
            continue
    return values


def evenly_sample_lines(path: Path, requested: int) -> Iterable[tuple[int, bytes]]:
    size = path.stat().st_size
    if requested <= 0:
        return
    with path.open("rb") as handle:
        seen_offsets: set[int] = set()
        for index in range(requested):
            offset = int((index + 0.5) * size / requested)
            handle.seek(offset)
            if offset:
                handle.readline()
            line_offset = handle.tell()
            if line_offset in seen_offsets:
                continue
            seen_offsets.add(line_offset)
            line = handle.readline()
            if line:
                yield line_offset, line


def audit_endomondo_sample(path: Path, requested: int = 5000) -> dict[str, object]:
    sport_counts: Counter[str] = Counter()
    gender_counts: Counter[str] = Counter()
    key_presence: Counter[str] = Counter()
    users: set[str] = set()
    spans: list[float] = []
    sampling_intervals: list[float] = []
    hr_coverages: list[float] = []
    array_length_consistent = 0
    eligible_provisional = 0
    sampled = 0

    for sample_index, (_, line) in enumerate(evenly_sample_lines(path, requested)):
        sampled += 1
        user_id = decode_match(USER_RE, line)
        sport = decode_match(SPORT_RE, line) or "<missing>"
        gender = decode_match(GENDER_RE, line) or "<missing>"
        if user_id:
            users.add(user_id)
        sport_counts[sport] += 1
        gender_counts[gender] += 1

        arrays: dict[str, list[float]] = {}
        for field, cast in (
            ("timestamp", int),
            ("heart_rate", float),
            ("speed", float),
            ("altitude", float),
            ("latitude", float),
            ("longitude", float),
        ):
            body = extract_array_bytes(line, field)
            if body is not None:
                key_presence[field] += 1
                arrays[field] = parse_numeric_array(body, cast=cast)

        timestamps = arrays.get("timestamp", [])
        heart_rate = arrays.get("heart_rate", [])
        if len(timestamps) >= 2:
            span = float(timestamps[-1] - timestamps[0])
            if span >= 0:
                spans.append(span)
        valid_hr = sum(30 <= value <= 240 for value in heart_rate)
        coverage = valid_hr / len(heart_rate) if heart_rate else 0.0
        hr_coverages.append(coverage)

        present_lengths = [
            len(arrays[field])
            for field in ("timestamp", "heart_rate", "altitude", "latitude", "longitude")
            if field in arrays
        ]
        if present_lengths and len(set(present_lengths)) == 1:
            array_length_consistent += 1

        if spans and spans[-1] >= 600 and coverage >= 0.80 and len(timestamps) >= 3:
            eligible_provisional += 1

        if sample_index % max(1, requested // 250) == 0 and len(timestamps) >= 2:
            sampling_intervals.extend(
                float(current - previous)
                for previous, current in zip(timestamps[:-1], timestamps[1:])
                if 0 < current - previous <= 600
            )

    return {
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "sampling_method": "deterministic evenly spaced byte-offset sample",
        "requested_records": requested,
        "sampled_records": sampled,
        "sampled_distinct_users": len(users),
        "sport_counts": dict(sport_counts.most_common()),
        "gender_counts": dict(gender_counts.most_common()),
        "field_presence": dict(key_presence),
        "session_span_seconds": summarize_numeric(spans),
        "sampling_interval_seconds": summarize_numeric(sampling_intervals),
        "hr_valid_coverage": summarize_numeric(hr_coverages),
        "records_with_consistent_core_array_lengths": array_length_consistent,
        "provisionally_eligible_records": eligible_provisional,
        "provisional_eligibility_rate": eligible_provisional / sampled if sampled else 0.0,
        "limitations": [
            "This is a deterministic cross-file sample, not a full Endomondo census.",
            "Eligibility does not yet enforce all forecast-origin gap and target-alignment rules.",
        ],
    }


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(character for character in normalized if not unicodedata.combining(character)).lower()


def provisional_sport_family(raw_label: str) -> str:
    label = normalize_text(raw_label)
    if raw_label == "<missing>" or not label.strip():
        return "other_unknown"
    indoor_terms = (
        "virtualride", "trainer", "kickr", "zwift", "rolle", "rullo", "rodillo",
        "tacx", "bkool", "indoor", "turbo", "spinning", "ftp",
    )
    if any(term in label for term in indoor_terms):
        return "indoor_virtual_cycling"
    swimming_terms = ("swim", "natacion", "nataci", "plavan", "pool")
    if any(term in label for term in swimming_terms):
        return "swimming"
    skiing_terms = ("ski", "sci fondo", "nordic", "langlauf", "randonee")
    if any(term in label for term in skiing_terms):
        return "skiing"
    walking_terms = (
        "walk", "hike", "hiking", "caminata", "caminar", "mountaineer", "snowshoe",
    )
    if any(term in label for term in walking_terms):
        return "walking_hiking"
    running_terms = (
        "run", "running", "carrera", "corsa", "corsetta", "trail run", "jog",
        "orienteer",
    )
    if any(term in label for term in running_terms):
        return "running"
    strength_terms = (
        "weight", "strength", "kraft", "fuerza", "gym", "yoga", "cross train",
        "elliptical", "stairstepper", "core stability", "circuit training",
        "aerobic", "pilates", "stair clim",
    )
    if any(term in label for term in strength_terms):
        return "strength_cross_training"
    cycling_terms = (
        "bike", "cycling", "cycli", "ride", "ciclismo", "ciclista", "giro",
        "pedal", "rennrad", "mtb", "enduro", "commute", "biker",
        "bicycle", "cycle",
    )
    if any(term in label for term in cycling_terms):
        return "outdoor_cycling"
    if label.strip() in {"road", "tt"}:
        return "outdoor_cycling"
    return "other_unknown"


def evenly_select_paths(paths: list[Path], requested: int) -> list[Path]:
    if requested <= 0 or not paths:
        return []
    if requested >= len(paths):
        return paths
    selected: list[Path] = []
    seen: set[int] = set()
    for index in range(requested):
        position = min(len(paths) - 1, int((index + 0.5) * len(paths) / requested))
        if position not in seen:
            seen.add(position)
            selected.append(paths[position])
    return selected


def audit_goldencheetah(root: Path, requested_csv_files: int = 3000) -> dict[str, object]:
    user_dirs = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    raw_sports: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    family_hr_mask_counts: Counter[str] = Counter()
    genders: Counter[str] = Counter()
    invalid_json: list[dict[str, str]] = []
    ride_records = 0

    for user_dir in user_dirs:
        json_paths = sorted(user_dir.glob("*.json"))
        if not json_paths:
            invalid_json.append({"user_id": user_dir.name, "error": "missing JSON"})
            continue
        try:
            metadata = json.loads(json_paths[0].read_text(encoding="utf-8-sig"))
        except Exception as exc:
            invalid_json.append(
                {"user_id": user_dir.name, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        gender = str(metadata.get("ATHLETE", {}).get("gender") or "<missing>")
        genders[gender] += 1
        for ride in metadata.get("RIDES", []):
            ride_records += 1
            raw_sport = str(ride.get("sport") or "<missing>")
            family = provisional_sport_family(raw_sport)
            raw_sports[raw_sport] += 1
            family_counts[family] += 1
            if "H" in str(ride.get("data") or ""):
                family_hr_mask_counts[family] += 1

    csv_paths = sorted(root.glob("*/*.csv"))
    selected_csv = evenly_select_paths(csv_paths, requested_csv_files)
    column_presence: Counter[str] = Counter()
    rows_per_file: list[float] = []
    duration_seconds: list[float] = []
    hr_coverages: list[float] = []
    files_with_hr_column = 0
    files_with_shared_core = 0
    provisionally_eligible = 0
    read_errors: list[dict[str, str]] = []

    for csv_path in selected_csv:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                fields = [field.strip() for field in header]
                field_index = {field: index for index, field in enumerate(fields)}
                column_presence.update(fields)
                has_hr = "hr" in field_index
                if has_hr:
                    files_with_hr_column += 1
                if {"secs", "hr", "km", "alt"}.issubset(field_index):
                    files_with_shared_core += 1
                row_count = 0
                valid_hr = 0
                first_second: float | None = None
                last_second: float | None = None
                for row in reader:
                    row_count += 1
                    if "secs" in field_index and field_index["secs"] < len(row):
                        try:
                            second = float(row[field_index["secs"]])
                            if first_second is None:
                                first_second = second
                            last_second = second
                        except (TypeError, ValueError):
                            pass
                    if has_hr and field_index["hr"] < len(row):
                        try:
                            heart_rate = float(row[field_index["hr"]])
                            valid_hr += 30 <= heart_rate <= 240
                        except (TypeError, ValueError):
                            pass
                duration = (
                    last_second - first_second
                    if first_second is not None and last_second is not None
                    else 0.0
                )
                coverage = valid_hr / row_count if row_count else 0.0
                rows_per_file.append(float(row_count))
                duration_seconds.append(float(duration))
                hr_coverages.append(coverage)
                if (
                    duration >= 600
                    and coverage >= 0.80
                    and {"secs", "hr"}.issubset(field_index)
                ):
                    provisionally_eligible += 1
        except Exception as exc:
            read_errors.append(
                {
                    "file": str(csv_path.relative_to(root)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    family_summary = []
    for family, count in family_counts.most_common():
        hr_count = family_hr_mask_counts[family]
        family_summary.append(
            {
                "sport_family": family,
                "ride_records": count,
                "metadata_hr_mask_records": hr_count,
                "metadata_hr_mask_rate": hr_count / count if count else 0.0,
            }
        )

    return {
        "source": str(root),
        "users": len(user_dirs),
        "json_valid_users": len(user_dirs) - len(invalid_json),
        "json_invalid_users": invalid_json,
        "gender_counts_valid_json": dict(genders.most_common()),
        "ride_records_valid_json": ride_records,
        "raw_sport_label_count": len(raw_sports),
        "top_raw_sports": [
            {"sport": sport, "ride_records": count}
            for sport, count in raw_sports.most_common(30)
        ],
        "provisional_sport_family_summary": family_summary,
        "csv_files_total": len(csv_paths),
        "csv_sampling_method": "deterministic evenly spaced selection over user/path-sorted CSV files",
        "csv_files_requested": requested_csv_files,
        "csv_files_sampled": len(selected_csv),
        "csv_read_errors": read_errors,
        "sample_column_presence": dict(column_presence.most_common()),
        "sample_files_with_hr_column": files_with_hr_column,
        "sample_files_with_shared_core_secs_hr_km_alt": files_with_shared_core,
        "sample_rows_per_file": summarize_numeric(rows_per_file),
        "sample_duration_seconds": summarize_numeric(duration_seconds),
        "sample_hr_valid_coverage": summarize_numeric(hr_coverages),
        "sample_provisionally_eligible_files": provisionally_eligible,
        "sample_provisional_eligibility_rate": (
            provisionally_eligible / len(selected_csv) if selected_csv else 0.0
        ),
        "limitations": [
            "Sport-family rules are provisional and require manual review of frequent unmatched labels.",
            "Metadata HR masks indicate recorded channels but do not replace row-level CSV validation.",
            "CSV eligibility is estimated from a deterministic sample, not yet a full census.",
            "CSV files are not yet linked one-to-one to metadata ride records and sport labels.",
        ],
    }


def run_audit(
    endomondo_path: Path,
    goldencheetah_root: Path,
    endomondo_samples: int,
    golden_csv_files: int,
) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_mode": "bounded feasibility audit",
        "endomondo": audit_endomondo_sample(endomondo_path, requested=endomondo_samples),
        "goldencheetah": audit_goldencheetah(
            goldencheetah_root, requested_csv_files=golden_csv_files
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit HR forecasting dataset feasibility.")
    parser.add_argument("--endomondo", type=Path, required=True)
    parser.add_argument("--goldencheetah", type=Path, required=True)
    parser.add_argument("--endomondo-samples", type=int, default=5000)
    parser.add_argument("--golden-csv-files", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_audit(
        args.endomondo,
        args.goldencheetah,
        endomondo_samples=args.endomondo_samples,
        golden_csv_files=args.golden_csv_files,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)

    print(json.dumps({
        "output": str(args.output),
        "endomondo_sampled_records": result["endomondo"]["sampled_records"],
        "goldencheetah_users": result["goldencheetah"]["users"],
        "goldencheetah_sampled_csv": result["goldencheetah"]["csv_files_sampled"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
