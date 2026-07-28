from __future__ import annotations

import argparse
import csv
from pathlib import Path

from data_audit import provisional_sport_family


PRIMARY_EXTERNAL_FAMILIES = {
    "outdoor_cycling",
    "running",
    "indoor_virtual_cycling",
}


def as_bool(value: str) -> bool:
    return value == "True"


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply an updated sport ontology to existing manifests.")
    parser.add_argument(
        "--kind",
        choices=("endomondo-quality", "golden-linkage", "golden-quality"),
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for row in rows:
        raw_sport = row.get("raw_sport", "")
        family = provisional_sport_family(raw_sport) if raw_sport else ""
        if args.kind == "endomondo-quality":
            row["provisional_sport_family"] = family
            row["provisional_model_eligible"] = str(
                as_bool(row["provisional_signal_eligible"]) and family != "other_unknown"
            )
        elif args.kind == "golden-linkage":
            row["provisional_sport_family"] = (
                family if row["link_status"] == "linked" else ""
            )
        else:
            row["provisional_sport_family"] = family
            model_eligible = (
                as_bool(row["provisional_signal_eligible"])
                and row["link_status"] == "linked"
                and family not in {"", "other_unknown"}
            )
            row["provisional_model_eligible"] = str(model_eligible)
            row["primary_external_family_candidate"] = str(
                model_eligible and family in PRIMARY_EXTERNAL_FAMILIES
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"{args.kind}: {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
