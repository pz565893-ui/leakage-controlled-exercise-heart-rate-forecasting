from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_audit import provisional_sport_family


CSV_TIME_FORMAT = "%Y_%m_%d_%H_%M_%S"
METADATA_TIME_FORMAT = "%Y/%m/%d %H:%M:%S UTC"
OFFSET_MINUTES = tuple(range(-14 * 60, 14 * 60 + 1, 15))


def parse_csv_time(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.stem, CSV_TIME_FORMAT)
    except ValueError:
        return None


def rows_for_user(user_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    csv_paths = sorted(user_dir.glob("*.csv"))
    csv_by_time: dict[datetime, Path] = {}
    unparsable_csv: list[Path] = []
    duplicate_csv_times: set[datetime] = set()
    for path in csv_paths:
        parsed = parse_csv_time(path)
        if parsed is None:
            unparsable_csv.append(path)
        elif parsed in csv_by_time:
            duplicate_csv_times.add(parsed)
        else:
            csv_by_time[parsed] = path

    json_paths = sorted(user_dir.glob("*.json"))
    metadata_error: str | None = None
    metadata: dict[str, object] | None = None
    if not json_paths:
        metadata_error = "missing JSON"
    else:
        try:
            metadata = json.loads(json_paths[0].read_text(encoding="utf-8-sig"))
        except Exception as exc:
            metadata_error = f"{type(exc).__name__}: {exc}"

    assignments: dict[Path, list[tuple[int, int, dict[str, object], datetime]]] = defaultdict(list)
    ambiguous_candidates: set[Path] = set()
    unparsable_ride_dates = 0
    rides = [] if metadata is None else list(metadata.get("RIDES", []))
    if metadata is not None:
        for ride_index, ride in enumerate(rides):
            try:
                utc_time = datetime.strptime(str(ride.get("date")), METADATA_TIME_FORMAT)
            except (TypeError, ValueError):
                unparsable_ride_dates += 1
                continue
            hits: list[tuple[int, Path]] = []
            for offset_minutes in OFFSET_MINUTES:
                candidate_time = utc_time + timedelta(minutes=offset_minutes)
                candidate_path = csv_by_time.get(candidate_time)
                if candidate_path is not None:
                    hits.append((offset_minutes, candidate_path))
            if len(hits) == 1:
                offset_minutes, candidate_path = hits[0]
                assignments[candidate_path].append((ride_index, offset_minutes, ride, utc_time))
            elif len(hits) > 1:
                ambiguous_candidates.update(path for _, path in hits)

    rows: list[dict[str, object]] = []
    for csv_path in csv_paths:
        parsed_time = parse_csv_time(csv_path)
        status = "unmatched_timestamp"
        linked: tuple[int, int, dict[str, object], datetime] | None = None
        if metadata_error is not None:
            status = "metadata_invalid_or_missing"
        elif parsed_time is None:
            status = "unparsable_csv_timestamp"
        elif parsed_time in duplicate_csv_times:
            status = "duplicate_csv_timestamp"
        elif csv_path in ambiguous_candidates:
            status = "ambiguous_timestamp_match"
        elif len(assignments[csv_path]) == 1:
            status = "linked"
            linked = assignments[csv_path][0]
        elif len(assignments[csv_path]) > 1:
            status = "duplicate_metadata_match"

        ride_index = ""
        offset_minutes: int | str = ""
        utc_time_text = ""
        raw_sport = ""
        family = ""
        data_mask = ""
        hr_flag: bool | str = ""
        if linked is not None:
            ride_index, offset_minutes, ride, utc_time = linked
            utc_time_text = utc_time.strftime(METADATA_TIME_FORMAT)
            raw_sport = str(ride.get("sport") or "<missing>")
            family = provisional_sport_family(raw_sport)
            data_mask = str(ride.get("data") or "")
            hr_flag = "H" in data_mask

        rows.append(
            {
                "user_id": user_dir.name,
                "csv_relative_path": f"{user_dir.name}/{csv_path.name}",
                "csv_local_datetime": "" if parsed_time is None else parsed_time.isoformat(),
                "link_status": status,
                "metadata_ride_index": ride_index,
                "metadata_utc_datetime": utc_time_text,
                "inferred_utc_offset_minutes": offset_minutes,
                "raw_sport": raw_sport,
                "provisional_sport_family": family,
                "data_mask": data_mask,
                "metadata_hr_flag": hr_flag,
            }
        )

    return rows, {
        "user_id": user_dir.name,
        "csv_files": len(csv_paths),
        "metadata_rides": len(rides),
        "metadata_error": metadata_error,
        "unparsable_csv_timestamps": len(unparsable_csv),
        "duplicate_csv_timestamps": len(duplicate_csv_times),
        "unparsable_metadata_dates": unparsable_ride_dates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Link GoldenCheetah CSV sessions to ride metadata without using row outcomes."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    user_summaries: list[dict[str, object]] = []
    for user_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        rows, summary = rows_for_user(user_dir)
        all_rows.extend(rows)
        user_summaries.append(summary)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = args.csv_output.with_suffix(args.csv_output.suffix + ".tmp")
    with csv_temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    csv_temp.replace(args.csv_output)

    status_counts = Counter(str(row["link_status"]) for row in all_rows)
    linked_family_counts = Counter(
        str(row["provisional_sport_family"])
        for row in all_rows
        if row["link_status"] == "linked"
    )
    linked_hr_family_counts = Counter(
        str(row["provisional_sport_family"])
        for row in all_rows
        if row["link_status"] == "linked" and row["metadata_hr_flag"] is True
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "linkage_rule": (
            "Exact timestamp matching within user after testing UTC offsets from -14:00 "
            "to +14:00 in 15-minute increments; ambiguous and duplicate matches are excluded."
        ),
        "csv_files": len(all_rows),
        "status_counts": dict(status_counts.most_common()),
        "linked_rate": status_counts["linked"] / len(all_rows) if all_rows else 0.0,
        "linked_family_counts": dict(linked_family_counts.most_common()),
        "linked_hr_flag_family_counts": dict(linked_hr_family_counts.most_common()),
        "users": user_summaries,
        "manifest_csv": str(args.csv_output),
    }
    json_temp = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
    json_temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    json_temp.replace(args.json_output)

    print(json.dumps({key: summary[key] for key in ("csv_files", "status_counts", "linked_rate")}, ensure_ascii=False))
    print(json.dumps(summary["linked_family_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
