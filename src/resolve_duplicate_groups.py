from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create conservative, deterministic actions for exact duplicate groups."
    )
    parser.add_argument("--dataset", choices=("endomondo", "goldencheetah"), required=True)
    parser.add_argument("--duplicates", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    duplicate_rows = read_csv(args.duplicates)
    quality_rows = read_csv(args.quality_manifest)
    quality_key = "record_index" if args.dataset == "endomondo" else "csv_relative_path"
    quality = {row[quality_key]: row for row in quality_rows}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in duplicate_rows:
        groups[row["duplicate_group"]].append(row)

    output_rows: list[dict[str, object]] = []
    group_decisions: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for group_id, members in groups.items():
        enriched = [(member, quality[member["source_key"]]) for member in members]
        users = {row["user_id"] for _, row in enriched}
        families = {row["provisional_sport_family"] for _, row in enriched}
        if len(users) > 1:
            group_decision = "exclude_all_cross_user_identity_conflict"
            canonical_key = ""
        elif len(families) > 1:
            group_decision = "exclude_all_sport_family_conflict"
            canonical_key = ""
        else:
            group_decision = "keep_one_within_user_family_consistent"
            canonical_key = min(
                (member["source_key"] for member, _ in enriched),
                key=(int if args.dataset == "endomondo" else str),
            )
        group_decisions[group_decision] += 1

        for member, row in enriched:
            if group_decision == "exclude_all_cross_user_identity_conflict":
                action = "exclude"
                reason = "exact_signal_duplicate_across_users"
            elif group_decision == "exclude_all_sport_family_conflict":
                action = "exclude"
                reason = "exact_signal_duplicate_with_family_conflict"
            elif member["source_key"] == canonical_key:
                action = "keep_canonical"
                reason = "first_deterministic_key_within_user_and_family"
            else:
                action = "exclude"
                reason = "redundant_exact_duplicate_of_kept_canonical"
            action_counts[action] += 1
            output_rows.append(
                {
                    "dataset": args.dataset,
                    "duplicate_group": group_id,
                    "sha256": member["sha256"],
                    "group_size": member["group_size"],
                    "source_key": member["source_key"],
                    "user_id": row["user_id"],
                    "sport_family": row["provisional_sport_family"],
                    "group_decision": group_decision,
                    "action": action,
                    "reason": reason,
                }
            )

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = args.csv_output.with_suffix(args.csv_output.suffix + ".tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary_csv.replace(args.csv_output)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "duplicate_groups": len(groups),
        "records_in_duplicate_groups": len(output_rows),
        "group_decisions": dict(group_decisions.most_common()),
        "record_actions": dict(action_counts.most_common()),
        "policy": [
            "Exclude every exact signal duplicate group spanning more than one user.",
            "Exclude every exact signal duplicate group with conflicting canonical sport families.",
            "Otherwise retain the first deterministic source key and exclude redundant copies.",
        ],
        "resolution_manifest_csv": str(args.csv_output),
    }
    temporary_json = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
    temporary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(args.json_output)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
