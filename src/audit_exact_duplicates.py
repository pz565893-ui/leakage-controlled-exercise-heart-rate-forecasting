from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from data_audit import extract_array_bytes


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_rows(groups: dict[str, list[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    duplicate_group_index = 0
    for digest, keys in groups.items():
        if len(keys) < 2:
            continue
        duplicate_group_index += 1
        for key in keys:
            rows.append(
                {
                    "duplicate_group": duplicate_group_index,
                    "sha256": digest,
                    "group_size": len(keys),
                    "source_key": key,
                }
            )
    return rows


def audit_endomondo(source: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: dict[str, list[str]] = defaultdict(list)
    records = 0
    with source.open("rb") as handle:
        for records, line in enumerate(handle, start=1):
            groups[hashlib.sha256(line).hexdigest()].append(str(records))
            if records % 50_000 == 0:
                print(f"Endomondo lines hashed: {records:,}", flush=True)
    rows = duplicate_rows(groups)
    return rows, {
        "dataset": "Endomondo",
        "source": str(source),
        "records_hashed": records,
        "unique_content_hashes": len(groups),
        "exact_duplicate_groups": len({row["duplicate_group"] for row in rows}),
        "records_in_exact_duplicate_groups": len(rows),
    }


def audit_endomondo_signal(
    source: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: dict[str, list[str]] = defaultdict(list)
    records = 0
    fields = ("timestamp", "heart_rate", "speed", "altitude", "latitude", "longitude")
    with source.open("rb") as handle:
        for records, line in enumerate(handle, start=1):
            digest = hashlib.sha256()
            for field in fields:
                body = extract_array_bytes(line, field)
                digest.update(field.encode("ascii"))
                digest.update(b"\x00")
                digest.update(body if body is not None else b"<missing>")
                digest.update(b"\x1f")
            groups[digest.hexdigest()].append(str(records))
            if records % 50_000 == 0:
                print(f"Endomondo signal fingerprints: {records:,}", flush=True)
    rows = duplicate_rows(groups)
    return rows, {
        "dataset": "Endomondo",
        "source": str(source),
        "records_hashed": records,
        "unique_content_hashes": len(groups),
        "exact_duplicate_groups": len({row["duplicate_group"] for row in rows}),
        "records_in_exact_duplicate_groups": len(rows),
        "fingerprint_fields": list(fields),
        "fingerprint_note": "IDs, URLs, user identifiers, gender, and sport labels are excluded.",
    }


def audit_golden(
    root: Path, manifest: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    for index, row in enumerate(manifest_rows, start=1):
        relative = row["csv_relative_path"]
        groups[hash_file(root / Path(relative))].append(relative)
        if index % 5_000 == 0:
            print(f"GoldenCheetah files hashed: {index:,}/{len(manifest_rows):,}", flush=True)
    rows = duplicate_rows(groups)
    return rows, {
        "dataset": "GoldenCheetah",
        "source": str(root),
        "records_hashed": len(manifest_rows),
        "unique_content_hashes": len(groups),
        "exact_duplicate_groups": len({row["duplicate_group"] for row in rows}),
        "records_in_exact_duplicate_groups": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit exact duplicate exercise records by SHA-256.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    endomondo = subparsers.add_parser("endomondo")
    endomondo.add_argument("--source", type=Path, required=True)
    endomondo.add_argument("--csv-output", type=Path, required=True)
    endomondo.add_argument("--json-output", type=Path, required=True)

    endomondo_signal = subparsers.add_parser("endomondo-signal")
    endomondo_signal.add_argument("--source", type=Path, required=True)
    endomondo_signal.add_argument("--csv-output", type=Path, required=True)
    endomondo_signal.add_argument("--json-output", type=Path, required=True)

    golden = subparsers.add_parser("goldencheetah")
    golden.add_argument("--root", type=Path, required=True)
    golden.add_argument("--manifest", type=Path, required=True)
    golden.add_argument("--csv-output", type=Path, required=True)
    golden.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset == "endomondo":
        rows, summary = audit_endomondo(args.source)
    elif args.dataset == "endomondo-signal":
        rows, summary = audit_endomondo_signal(args.source)
    else:
        rows, summary = audit_golden(args.root, args.manifest)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = args.csv_output.with_suffix(args.csv_output.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["duplicate_group", "sha256", "group_size", "source_key"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(args.csv_output)

    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "hash_algorithm": "SHA-256",
            "duplicate_manifest_csv": str(args.csv_output),
            "scope_note": (
                "Exact byte-identical records or exact signal fingerprints only; approximate "
                "near-duplicate detection is separate."
            ),
        }
    )
    temporary_json = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
    temporary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(args.json_output)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
