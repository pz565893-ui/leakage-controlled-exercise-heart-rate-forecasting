from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from data_audit import summarize_numeric


PRIMARY_EXTERNAL_FAMILIES = {
    "outdoor_cycling",
    "running",
    "indoor_virtual_cycling",
}

METRIC_FIELDS = [
    "read_error",
    "header",
    "row_count",
    "valid_secs_count",
    "valid_hr_count",
    "valid_hr_coverage",
    "valid_km_coverage",
    "valid_alt_coverage",
    "valid_power_coverage",
    "valid_cad_coverage",
    "start_second",
    "end_second",
    "duration_seconds",
    "median_positive_gap_seconds",
    "max_positive_gap_seconds",
    "nonpositive_timestamp_gaps",
    "gaps_over_60_seconds",
    "negative_distance_increments",
    "provisional_signal_eligible",
    "provisional_model_eligible",
    "primary_external_family_candidate",
]


def finite_float(row: list[str], index: int | None) -> float | None:
    if index is None or index >= len(row):
        return None
    value = row[index]
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def counter_median(counts: Counter[float], total: int) -> float | str:
    if total <= 0:
        return ""
    lower_position = (total - 1) // 2
    upper_position = total // 2
    cumulative = 0
    lower_value: float | None = None
    upper_value: float | None = None
    for value in sorted(counts):
        cumulative += counts[value]
        if lower_value is None and cumulative > lower_position:
            lower_value = value
        if cumulative > upper_position:
            upper_value = value
            break
    if lower_value is None or upper_value is None:
        return ""
    return (lower_value + upper_value) / 2


def scan_csv(path: Path) -> dict[str, object]:
    row_count = 0
    valid_hr = valid_secs = valid_km = valid_alt = valid_power = valid_cad = 0
    first_second: float | None = None
    last_second: float | None = None
    previous_second: float | None = None
    previous_km: float | None = None
    gap_counts: Counter[float] = Counter()
    positive_gap_count = 0
    max_gap: float | None = None
    gaps_over_60 = 0
    nonpositive_gaps = 0
    negative_distance_increments = 0
    header: list[str] = []
    read_error = ""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            index = {name.strip(): position for position, name in enumerate(header)}
            secs_i = index.get("secs")
            km_i = index.get("km")
            power_i = index.get("power")
            hr_i = index.get("hr")
            cad_i = index.get("cad")
            alt_i = index.get("alt")
            for row in reader:
                row_count += 1
                second = finite_float(row, secs_i)
                if second is not None:
                    valid_secs += 1
                    if first_second is None:
                        first_second = second
                    last_second = second
                    if previous_second is not None:
                        gap = second - previous_second
                        if gap > 0:
                            gap_counts[gap] += 1
                            positive_gap_count += 1
                            max_gap = gap if max_gap is None or gap > max_gap else max_gap
                            gaps_over_60 += gap > 60
                        else:
                            nonpositive_gaps += 1
                    previous_second = second

                hr = finite_float(row, hr_i)
                valid_hr += hr is not None and 30 <= hr <= 240

                km = finite_float(row, km_i)
                if km is not None:
                    valid_km += 1
                    if previous_km is not None and km < previous_km:
                        negative_distance_increments += 1
                    previous_km = km

                valid_power += finite_float(row, power_i) is not None
                valid_cad += finite_float(row, cad_i) is not None
                valid_alt += finite_float(row, alt_i) is not None
    except Exception as exc:
        read_error = f"{type(exc).__name__}: {exc}"

    duration = (
        last_second - first_second
        if first_second is not None and last_second is not None
        else 0.0
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
        "median_positive_gap_seconds": counter_median(gap_counts, positive_gap_count),
        "max_positive_gap_seconds": "" if max_gap is None else max_gap,
        "nonpositive_timestamp_gaps": nonpositive_gaps,
        "gaps_over_60_seconds": gaps_over_60,
        "negative_distance_increments": negative_distance_increments,
    }


def count_part_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def write_part(
    part_path: Path,
    root: Path,
    linkage_chunk: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    temporary = part_path.with_suffix(part_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for linkage in linkage_chunk:
            metrics = scan_csv(root / Path(linkage["csv_relative_path"]))
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
            metrics["provisional_signal_eligible"] = signal_eligible
            metrics["provisional_model_eligible"] = model_eligible
            metrics["primary_external_family_candidate"] = (
                model_eligible and family in PRIMARY_EXTERNAL_FAMILIES
            )
            row: dict[str, object] = dict(linkage)
            row.update(metrics)
            writer.writerow(row)
    temporary.replace(part_path)


def merge_parts(parts: list[Path], output: Path, fieldnames: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for part in parts:
            with part.open("r", encoding="utf-8-sig", newline="") as source:
                writer.writerows(csv.DictReader(source))
    temporary.replace(output)


def summarize_manifest(path: Path) -> dict[str, object]:
    records = 0
    model_eligible = 0
    primary_candidates = 0
    durations: list[float] = []
    coverages: list[float] = []
    eligible_family_records: Counter[str] = Counter()
    eligible_family_users: dict[str, set[str]] = defaultdict(set)
    read_errors: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            records += 1
            durations.append(float(row["duration_seconds"]))
            coverages.append(float(row["valid_hr_coverage"]))
            if row["read_error"]:
                read_errors.append(
                    {
                        "csv_relative_path": row["csv_relative_path"],
                        "error": row["read_error"],
                    }
                )
            if row["provisional_model_eligible"] == "True":
                model_eligible += 1
                family = row["provisional_sport_family"]
                eligible_family_records[family] += 1
                eligible_family_users[family].add(row["user_id"])
            primary_candidates += row["primary_external_family_candidate"] == "True"
    return {
        "csv_files": records,
        "read_errors": read_errors,
        "provisional_model_eligible_records": model_eligible,
        "provisional_model_eligible_rate": model_eligible / records if records else 0.0,
        "primary_external_family_candidate_records": primary_candidates,
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a restartable, chunked GoldenCheetah session-quality manifest."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--linkage", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--part-size", type=int, default=500)
    args = parser.parse_args()

    with args.linkage.open("r", encoding="utf-8-sig", newline="") as handle:
        linkage_rows = list(csv.DictReader(handle))
    linkage_fields = list(linkage_rows[0])
    fieldnames = linkage_fields + METRIC_FIELDS
    parts_dir = args.csv_output.parent / f"{args.csv_output.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []

    for start in range(0, len(linkage_rows), args.part_size):
        chunk = linkage_rows[start : start + args.part_size]
        part_index = start // args.part_size
        part_path = parts_dir / f"part_{part_index:05d}.csv"
        parts.append(part_path)
        if part_path.exists() and count_part_rows(part_path) == len(chunk):
            print(
                f"GoldenCheetah part {part_index + 1}/{(len(linkage_rows) - 1) // args.part_size + 1}: reused",
                flush=True,
            )
            continue
        write_part(part_path, args.root, chunk, fieldnames)
        print(
            f"GoldenCheetah part {part_index + 1}/{(len(linkage_rows) - 1) // args.part_size + 1}: completed",
            flush=True,
        )

    merge_parts(parts, args.csv_output, fieldnames)
    summary = summarize_manifest(args.csv_output)
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "GoldenCheetah",
            "source": str(args.root),
            "manifest_csv": str(args.csv_output),
            "part_size": args.part_size,
            "parts_directory": str(parts_dir),
            "eligibility_note": (
                "Session-level screening uses a 10-minute to 24-hour duration range. Final "
                "forecast origins still require local context, gap, interpolation, and "
                "horizon-target checks."
            ),
        }
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.json_output)
    print(json.dumps(summary, ensure_ascii=False)[:5000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
