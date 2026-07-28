from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from data_audit import (
    GENDER_RE,
    SPORT_RE,
    USER_RE,
    decode_match,
    extract_array_bytes,
    parse_numeric_array,
    provisional_sport_family,
    summarize_numeric,
)


ID_RE = re.compile(rb"'id':\s*(\d+)")
PRIMARY_EXTERNAL_FAMILIES = {
    "outdoor_cycling",
    "running",
    "indoor_virtual_cycling",
}


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def gap_statistics(timestamps: list[float]) -> tuple[int, float | None, float | None, int]:
    gaps: list[float] = []
    nonpositive = 0
    for previous, current in zip(timestamps[:-1], timestamps[1:]):
        gap = current - previous
        if gap > 0:
            gaps.append(float(gap))
        else:
            nonpositive += 1
    if not gaps:
        return nonpositive, None, None, 0
    ordered = sorted(gaps)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return nonpositive, float(median), float(max(gaps)), sum(gap > 60 for gap in gaps)


def build_endomondo_manifest(source: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    durations: list[float] = []
    coverages: list[float] = []
    users: set[str] = set()
    family_records: Counter[str] = Counter()
    eligible_family_records: Counter[str] = Counter()
    eligible_family_users: dict[str, set[str]] = defaultdict(set)

    with source.open("rb") as handle:
        record_index = 0
        while True:
            byte_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            record_index += 1
            user_id = decode_match(USER_RE, line) or "<missing>"
            activity_id = decode_match(ID_RE, line) or ""
            raw_sport = decode_match(SPORT_RE, line) or "<missing>"
            gender = decode_match(GENDER_RE, line) or "<missing>"
            family = provisional_sport_family(raw_sport)
            timestamps = parse_numeric_array(extract_array_bytes(line, "timestamp"), cast=int)
            heart_rate = parse_numeric_array(extract_array_bytes(line, "heart_rate"), cast=float)
            speed_present = extract_array_bytes(line, "speed") is not None
            altitude_present = extract_array_bytes(line, "altitude") is not None
            latitude_present = extract_array_bytes(line, "latitude") is not None
            longitude_present = extract_array_bytes(line, "longitude") is not None

            duration = (
                float(timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0
            )
            valid_hr = sum(30 <= value <= 240 for value in heart_rate)
            hr_coverage = valid_hr / len(heart_rate) if heart_rate else 0.0
            nonpositive_gaps, median_gap, max_gap, gaps_over_60 = gap_statistics(timestamps)
            aligned_timestamp_hr = len(timestamps) == len(heart_rate) and len(timestamps) > 0
            signal_eligible = (
                600 <= duration <= 86_400
                and hr_coverage >= 0.80
                and aligned_timestamp_hr
                and nonpositive_gaps == 0
            )
            model_eligible = signal_eligible and family != "other_unknown"

            rows.append(
                {
                    "record_index": record_index,
                    "byte_offset": byte_offset,
                    "activity_id": activity_id,
                    "user_id": user_id,
                    "gender": gender,
                    "raw_sport": raw_sport,
                    "provisional_sport_family": family,
                    "timestamp_count": len(timestamps),
                    "heart_rate_count": len(heart_rate),
                    "valid_hr_count": valid_hr,
                    "valid_hr_coverage": hr_coverage,
                    "start_timestamp": timestamps[0] if timestamps else "",
                    "end_timestamp": timestamps[-1] if timestamps else "",
                    "duration_seconds": duration,
                    "median_positive_gap_seconds": "" if median_gap is None else median_gap,
                    "max_positive_gap_seconds": "" if max_gap is None else max_gap,
                    "nonpositive_timestamp_gaps": nonpositive_gaps,
                    "gaps_over_60_seconds": gaps_over_60,
                    "timestamp_hr_lengths_aligned": aligned_timestamp_hr,
                    "speed_field_present": speed_present,
                    "altitude_field_present": altitude_present,
                    "latitude_field_present": latitude_present,
                    "longitude_field_present": longitude_present,
                    "provisional_signal_eligible": signal_eligible,
                    "provisional_model_eligible": model_eligible,
                }
            )
            users.add(user_id)
            family_records[family] += 1
            durations.append(duration)
            coverages.append(hr_coverage)
            if model_eligible:
                eligible_family_records[family] += 1
                eligible_family_users[family].add(user_id)
            if record_index % 25_000 == 0:
                print(f"Endomondo manifest records: {record_index:,}", flush=True)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Endomondo",
        "source": str(source),
        "records": len(rows),
        "users": len(users),
        "family_records": dict(family_records.most_common()),
        "provisional_model_eligible_records": sum(eligible_family_records.values()),
        "provisional_model_eligible_rate": (
            sum(eligible_family_records.values()) / len(rows) if rows else 0.0
        ),
        "eligible_family_support": [
            {
                "sport_family": family,
                "eligible_records": count,
                "eligible_users": len(eligible_family_users[family]),
            }
            for family, count in eligible_family_records.most_common()
        ],
        "duration_seconds": summarize_numeric(durations),
        "valid_hr_coverage": summarize_numeric(coverages),
        "eligibility_note": (
            "Session-level screening uses a 10-minute to 24-hour duration range. Final forecast "
            "origins still require local context, gap, interpolation, and horizon-target checks."
        ),
    }
    return rows, summary


def as_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def scan_golden_csv(path: Path) -> dict[str, object]:
    row_count = 0
    valid_hr = 0
    valid_secs = 0
    valid_km = 0
    valid_alt = 0
    valid_power = 0
    valid_cad = 0
    first_second: float | None = None
    last_second: float | None = None
    previous_second: float | None = None
    previous_km: float | None = None
    positive_gaps: list[float] = []
    nonpositive_gaps = 0
    negative_distance_increments = 0
    read_error = ""
    header: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            for row in reader:
                row_count += 1
                second = as_float(row.get("secs"))
                if second is not None:
                    valid_secs += 1
                    if first_second is None:
                        first_second = second
                    last_second = second
                    if previous_second is not None:
                        gap = second - previous_second
                        if gap > 0:
                            positive_gaps.append(gap)
                        else:
                            nonpositive_gaps += 1
                    previous_second = second
                hr = as_float(row.get("hr"))
                valid_hr += hr is not None and 30 <= hr <= 240
                km = as_float(row.get("km"))
                if km is not None:
                    valid_km += 1
                    if previous_km is not None and km < previous_km:
                        negative_distance_increments += 1
                    previous_km = km
                valid_alt += as_float(row.get("alt")) is not None
                valid_power += as_float(row.get("power")) is not None
                valid_cad += as_float(row.get("cad")) is not None
    except Exception as exc:
        read_error = f"{type(exc).__name__}: {exc}"

    duration = (
        last_second - first_second
        if first_second is not None and last_second is not None
        else 0.0
    )
    ordered_gaps = sorted(positive_gaps)
    median_gap: float | str = ""
    if ordered_gaps:
        middle = len(ordered_gaps) // 2
        median_gap = (
            ordered_gaps[middle]
            if len(ordered_gaps) % 2
            else (ordered_gaps[middle - 1] + ordered_gaps[middle]) / 2
        )
    return {
        "read_error": read_error,
        "header": "|".join(header),
        "row_count": row_count,
        "valid_secs_count": valid_secs,
        "valid_hr_count": valid_hr,
        "valid_hr_coverage": valid_hr / row_count if row_count else 0.0,
        "valid_km_coverage": valid_km / row_count if row_count else 0.0,
        "valid_alt_coverage": valid_alt / row_count if row_count else 0.0,
        "valid_power_coverage": valid_power / row_count if row_count else 0.0,
        "valid_cad_coverage": valid_cad / row_count if row_count else 0.0,
        "start_second": "" if first_second is None else first_second,
        "end_second": "" if last_second is None else last_second,
        "duration_seconds": duration,
        "median_positive_gap_seconds": median_gap,
        "max_positive_gap_seconds": max(positive_gaps) if positive_gaps else "",
        "nonpositive_timestamp_gaps": nonpositive_gaps,
        "gaps_over_60_seconds": sum(gap > 60 for gap in positive_gaps),
        "negative_distance_increments": negative_distance_increments,
    }


def build_golden_manifest(
    root: Path, linkage_csv: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    with linkage_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        linkage_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    durations: list[float] = []
    coverages: list[float] = []
    eligible_family_records: Counter[str] = Counter()
    eligible_family_users: dict[str, set[str]] = defaultdict(set)
    read_errors: list[dict[str, str]] = []

    for index, linkage in enumerate(linkage_rows, start=1):
        relative = Path(linkage["csv_relative_path"])
        metrics = scan_golden_csv(root / relative)
        family = linkage["provisional_sport_family"]
        signal_eligible = (
            not metrics["read_error"]
            and 600 <= float(metrics["duration_seconds"]) <= 86_400
            and float(metrics["valid_hr_coverage"]) >= 0.80
            and int(metrics["nonpositive_timestamp_gaps"]) == 0
        )
        model_eligible = (
            signal_eligible
            and linkage["link_status"] == "linked"
            and family not in {"", "other_unknown"}
        )
        primary_external_candidate = model_eligible and family in PRIMARY_EXTERNAL_FAMILIES
        row = dict(linkage)
        row.update(metrics)
        row["provisional_signal_eligible"] = signal_eligible
        row["provisional_model_eligible"] = model_eligible
        row["primary_external_family_candidate"] = primary_external_candidate
        rows.append(row)
        durations.append(float(metrics["duration_seconds"]))
        coverages.append(float(metrics["valid_hr_coverage"]))
        if model_eligible:
            eligible_family_records[family] += 1
            eligible_family_users[family].add(linkage["user_id"])
        if metrics["read_error"]:
            read_errors.append(
                {"csv_relative_path": str(relative), "error": str(metrics["read_error"])}
            )
        if index % 2_500 == 0:
            print(f"GoldenCheetah manifest files: {index:,}/{len(linkage_rows):,}", flush=True)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "GoldenCheetah",
        "source": str(root),
        "csv_files": len(rows),
        "read_errors": read_errors,
        "provisional_model_eligible_records": sum(eligible_family_records.values()),
        "provisional_model_eligible_rate": (
            sum(eligible_family_records.values()) / len(rows) if rows else 0.0
        ),
        "eligible_family_support": [
            {
                "sport_family": family,
                "eligible_records": count,
                "eligible_users": len(eligible_family_users[family]),
            }
            for family, count in eligible_family_records.most_common()
        ],
        "duration_seconds": summarize_numeric(durations),
        "valid_hr_coverage": summarize_numeric(coverages),
        "eligibility_note": (
            "Session-level screening uses a 10-minute to 24-hour duration range. Final forecast "
            "origins still require local context, gap, interpolation, and horizon-target checks."
        ),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build full session-level quality manifests.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    endomondo = subparsers.add_parser("endomondo")
    endomondo.add_argument("--source", type=Path, required=True)
    endomondo.add_argument("--csv-output", type=Path, required=True)
    endomondo.add_argument("--json-output", type=Path, required=True)

    golden = subparsers.add_parser("goldencheetah")
    golden.add_argument("--root", type=Path, required=True)
    golden.add_argument("--linkage", type=Path, required=True)
    golden.add_argument("--csv-output", type=Path, required=True)
    golden.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset == "endomondo":
        rows, summary = build_endomondo_manifest(args.source)
    else:
        rows, summary = build_golden_manifest(args.root, args.linkage)
    atomic_csv(args.csv_output, rows)
    summary["manifest_csv"] = str(args.csv_output)
    atomic_json(args.json_output, summary)
    print(json.dumps(summary, ensure_ascii=False)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
