from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from data_audit import SPORT_RE, USER_RE, decode_match, normalize_text, provisional_sport_family


ONTOLOGY_VERSION = "0.2.0-adjudicated-rules"
RULE_MAPPED_STATUS = "rule_mapped_locked"
UNRESOLVED_STATUS = "retained_unknown_locked"
LOCKED_MAPPING_STATUS = (
    "locked outcome-blind rule mapping; unresolved labels retained as other_unknown"
)


def mapping_fingerprint(rows: list[dict[str, object]]) -> str:
    """Hash only the ordered source-label-to-family mapping, excluding metadata."""
    digest = hashlib.sha256()
    for row in rows:
        for field in ("source", "raw_label", "provisional_family"):
            value = str(row[field]).encode("utf-8")
            digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
            digest.update(value)
    return digest.hexdigest()


def scan_endomondo(path: Path) -> tuple[dict[str, dict[str, object]], int, int]:
    labels: dict[str, dict[str, object]] = defaultdict(
        lambda: {"records": 0, "users": set(), "hr_mask_records": None}
    )
    all_users: set[str] = set()
    total = 0
    with path.open("rb") as handle:
        for total, line in enumerate(handle, start=1):
            raw_label = decode_match(SPORT_RE, line) or "<missing>"
            user_id = decode_match(USER_RE, line)
            labels[raw_label]["records"] += 1
            if user_id:
                labels[raw_label]["users"].add(user_id)
                all_users.add(user_id)
            if total % 50_000 == 0:
                print(f"Endomondo records scanned: {total:,}", flush=True)
    return labels, total, len(all_users)


def scan_goldencheetah(
    root: Path,
) -> tuple[dict[str, dict[str, object]], int, int, list[dict[str, str]]]:
    labels: dict[str, dict[str, object]] = defaultdict(
        lambda: {"records": 0, "users": set(), "hr_mask_records": 0}
    )
    valid_users = 0
    total_rides = 0
    errors: list[dict[str, str]] = []
    for user_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        json_paths = sorted(user_dir.glob("*.json"))
        if not json_paths:
            errors.append({"user_id": user_dir.name, "error": "missing JSON"})
            continue
        try:
            metadata = json.loads(json_paths[0].read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append(
                {"user_id": user_dir.name, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        valid_users += 1
        for ride in metadata.get("RIDES", []):
            raw_label = str(ride.get("sport") or "<missing>")
            labels[raw_label]["records"] += 1
            labels[raw_label]["users"].add(user_dir.name)
            if "H" in str(ride.get("data") or ""):
                labels[raw_label]["hr_mask_records"] += 1
            total_rides += 1
    return labels, total_rides, valid_users, errors


def ontology_rows(
    source: str, labels: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_label, stats in labels.items():
        family = provisional_sport_family(raw_label)
        hr_records = stats["hr_mask_records"]
        records = int(stats["records"])
        rows.append(
            {
                "ontology_version": ONTOLOGY_VERSION,
                "source": source,
                "raw_label": raw_label,
                "normalized_label": normalize_text(raw_label),
                "provisional_family": family,
                "records": records,
                "users": len(stats["users"]),
                "metadata_hr_mask_records": "" if hr_records is None else int(hr_records),
                "metadata_hr_mask_rate": (
                    "" if hr_records is None or records == 0 else int(hr_records) / records
                ),
                "review_status": (
                    UNRESOLVED_STATUS
                    if family == "other_unknown"
                    else RULE_MAPPED_STATUS
                ),
            }
        )
    return sorted(rows, key=lambda row: (str(row["source"]), -int(row["records"]), str(row["raw_label"])))


def source_family_summary(
    source: str, labels: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    records: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    users: dict[str, set[str]] = defaultdict(set)
    hr_records: Counter[str] = Counter()
    has_hr_metadata = False
    for raw_label, stats in labels.items():
        family = provisional_sport_family(raw_label)
        records[family] += int(stats["records"])
        label_counts[family] += 1
        users[family].update(stats["users"])
        if stats["hr_mask_records"] is not None:
            has_hr_metadata = True
            hr_records[family] += int(stats["hr_mask_records"])
    return [
        {
            "source": source,
            "provisional_family": family,
            "records": count,
            "users": len(users[family]),
            "raw_labels": label_counts[family],
            "metadata_hr_mask_records": hr_records[family] if has_hr_metadata else None,
            "metadata_hr_mask_rate": (
                hr_records[family] / count if has_hr_metadata and count else None
            ),
        }
        for family, count in records.most_common()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the versioned raw-label sport ontology.")
    parser.add_argument("--endomondo", type=Path, required=True)
    parser.add_argument("--goldencheetah", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    endomondo, endomondo_records, endomondo_users = scan_endomondo(args.endomondo)
    golden, golden_rides, golden_users, golden_errors = scan_goldencheetah(args.goldencheetah)
    rows = ontology_rows("Endomondo", endomondo) + ontology_rows("GoldenCheetah", golden)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = args.csv_output.with_suffix(args.csv_output.suffix + ".tmp")
    with csv_temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_temp.replace(args.csv_output)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ontology_version": ONTOLOGY_VERSION,
        "status": LOCKED_MAPPING_STATUS,
        "mapping_fingerprint_sha256": mapping_fingerprint(rows),
        "mapping_fingerprint_scope": (
            "ordered UTF-8 length-prefixed source, raw_label, provisional_family tuples; "
            "review status and generated metadata excluded"
        ),
        "lock_semantics": {
            "analysis_mapping": "locked",
            "mapped_label_status": RULE_MAPPED_STATUS,
            "unresolved_label_status": UNRESOLVED_STATUS,
            "future_semantic_review": (
                "may inform a new ontology version only; it cannot alter the reported "
                "v0.2.0 analyses in place"
            ),
        },
        "endomondo": {
            "records": endomondo_records,
            "users": endomondo_users,
            "raw_labels": len(endomondo),
        },
        "goldencheetah": {
            "ride_records": golden_rides,
            "valid_json_users": golden_users,
            "raw_labels": len(golden),
            "json_errors": golden_errors,
        },
        "family_summary": (
            source_family_summary("Endomondo", endomondo)
            + source_family_summary("GoldenCheetah", golden)
        ),
        "manual_review_queue": [
            row
            for row in rows
            if row["provisional_family"] == "other_unknown"
        ],
        # Keep generated audit metadata portable and avoid embedding a local path.
        "mapping_csv": args.csv_output.name,
    }
    json_temp = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
    json_temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    json_temp.replace(args.json_output)

    print(
        json.dumps(
            {
                "csv_output": str(args.csv_output),
                "json_output": str(args.json_output),
                "endomondo_records": endomondo_records,
                "goldencheetah_rides": golden_rides,
                "mapping_rows": len(rows),
                "manual_review_labels": len(summary["manual_review_queue"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
