from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SEED = 20260722
USER_PARTITIONS = (
    (0.70, "train"),
    (0.80, "validation"),
    (0.90, "calibration"),
    (1.00, "test"),
)
PRIMARY_EXTERNAL_FAMILIES = {
    "outdoor_cycling",
    "running",
    "indoor_virtual_cycling",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_fraction(namespace: str, key: str) -> float:
    payload = f"{SEED}|{namespace}|{key}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def stable_partition(namespace: str, key: str) -> str:
    value = stable_fraction(namespace, key)
    for upper, label in USER_PARTITIONS:
        if value < upper:
            return label
    raise AssertionError("partition thresholds do not cover [0, 1)")


def duplicate_actions(path: Path) -> dict[str, str]:
    return {row["source_key"]: row["action"] for row in read_csv(path)}


def temporal_partitions(rows: list[dict[str, object]]) -> dict[str, str]:
    by_user: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["analysis_eligible"] is True:
            by_user[str(row["user_id"])].append(row)
    assignment: dict[str, str] = {}
    for user_id, user_rows in by_user.items():
        if len(user_rows) < 10:
            for row in user_rows:
                assignment[row["record_index"]] = "insufficient_user_history"
            continue
        ordered = sorted(
            user_rows,
            key=lambda row: (float(row["start_timestamp"]), int(row["record_index"])),
        )
        count = len(ordered)
        train_end = max(1, math.floor(0.70 * count))
        validation_end = max(train_end + 1, math.floor(0.80 * count))
        calibration_end = max(validation_end + 1, math.floor(0.90 * count))
        calibration_end = min(calibration_end, count - 1)
        validation_end = min(validation_end, calibration_end - 1)
        train_end = min(train_end, validation_end - 1)
        for index, row in enumerate(ordered):
            if index < train_end:
                label = "train"
            elif index < validation_end:
                label = "validation"
            elif index < calibration_end:
                label = "calibration"
            else:
                label = "test"
            assignment[row["record_index"]] = label
    return assignment


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_endomondo(
    quality_path: Path, duplicate_path: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    quality = read_csv(quality_path)
    actions = duplicate_actions(duplicate_path)
    rows: list[dict[str, object]] = []
    family_records: Counter[str] = Counter()
    family_users: dict[str, set[str]] = defaultdict(set)
    for source in quality:
        key = source["record_index"]
        duplicate_action = actions.get(key, "not_in_duplicate_group")
        analysis_eligible = (
            source["provisional_model_eligible"] == "True"
            and duplicate_action != "exclude"
        )
        family = source["provisional_sport_family"]
        if analysis_eligible:
            family_records[family] += 1
            family_users[family].add(source["user_id"])
        rows.append(
            {
                "record_index": key,
                "activity_id": source["activity_id"],
                "user_id": source["user_id"],
                "start_timestamp": source["start_timestamp"],
                "sport_family": family,
                "duplicate_action": duplicate_action,
                "analysis_eligible": analysis_eligible,
                "unseen_user_partition": stable_partition(
                    "endomondo_unseen_user", source["user_id"]
                ),
                "within_user_temporal_partition": "",
                "sport_shift_candidate": False,
                "joint_shift_user_partition": stable_partition(
                    "endomondo_unseen_user", source["user_id"]
                ),
            }
        )

    qualifying_families = {
        family
        for family, count in family_records.items()
        if count >= 1_000 and len(family_users[family]) >= 50
    }
    temporal = temporal_partitions(rows)
    for row in rows:
        row["within_user_temporal_partition"] = temporal.get(
            str(row["record_index"]), "ineligible"
        )
        row["sport_shift_candidate"] = (
            row["analysis_eligible"] is True and row["sport_family"] in qualifying_families
        )

    unseen_partition_users: dict[str, set[str]] = defaultdict(set)
    unseen_partition_records: Counter[str] = Counter()
    temporal_records: Counter[str] = Counter()
    for row in rows:
        if row["analysis_eligible"] is True:
            partition = str(row["unseen_user_partition"])
            unseen_partition_users[partition].add(str(row["user_id"]))
            unseen_partition_records[partition] += 1
            temporal_records[str(row["within_user_temporal_partition"])] += 1

    summary = {
        "dataset": "Endomondo",
        "seed": SEED,
        "analysis_eligible_records": sum(
            row["analysis_eligible"] is True for row in rows
        ),
        "qualifying_sport_shift_families": sorted(qualifying_families),
        "eligible_family_support_after_duplicate_control": [
            {
                "sport_family": family,
                "records": count,
                "users": len(family_users[family]),
            }
            for family, count in family_records.most_common()
        ],
        "unseen_user_partition_records": dict(unseen_partition_records),
        "unseen_user_partition_users": {
            partition: len(users) for partition, users in unseen_partition_users.items()
        },
        "within_user_temporal_partition_records": dict(temporal_records),
    }
    return rows, summary


def build_golden(
    quality_path: Path, duplicate_path: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    quality = read_csv(quality_path)
    actions = duplicate_actions(duplicate_path)
    family_primary_users: dict[str, set[str]] = defaultdict(set)
    for source in quality:
        key = source["csv_relative_path"]
        family = source["provisional_sport_family"]
        analysis_eligible = (
            source["provisional_model_eligible"] == "True"
            and actions.get(key, "not_in_duplicate_group") != "exclude"
        )
        if analysis_eligible and family in PRIMARY_EXTERNAL_FAMILIES:
            family_primary_users[family].add(source["user_id"])
    calibration_users: set[str] = set()
    for family, users in family_primary_users.items():
        ordered = sorted(
            users,
            key=lambda user: stable_fraction(
                f"golden_secondary_adaptation|{family}", user
            ),
        )
        calibration_users.update(ordered[: math.ceil(0.20 * len(ordered))])

    rows: list[dict[str, object]] = []
    family_records: Counter[str] = Counter()
    family_users: dict[str, set[str]] = defaultdict(set)
    secondary_partition_users: dict[str, set[str]] = defaultdict(set)
    for source in quality:
        key = source["csv_relative_path"]
        duplicate_action = actions.get(key, "not_in_duplicate_group")
        family = source["provisional_sport_family"]
        analysis_eligible = (
            source["provisional_model_eligible"] == "True"
            and duplicate_action != "exclude"
        )
        primary_external = analysis_eligible and family in PRIMARY_EXTERNAL_FAMILIES
        secondary = (
            "external_calibration"
            if source["user_id"] in calibration_users
            else "external_test"
        )
        if analysis_eligible:
            family_records[family] += 1
            family_users[family].add(source["user_id"])
        if primary_external:
            secondary_partition_users[secondary].add(source["user_id"])
        rows.append(
            {
                "csv_relative_path": key,
                "user_id": source["user_id"],
                "metadata_utc_datetime": source["metadata_utc_datetime"],
                "sport_family": family,
                "duplicate_action": duplicate_action,
                "analysis_eligible": analysis_eligible,
                "primary_external_partition": (
                    "frozen_external_test" if primary_external else "not_primary_external"
                ),
                "secondary_adaptation_partition": (
                    secondary if primary_external else "not_primary_external"
                ),
            }
        )

    summary = {
        "dataset": "GoldenCheetah",
        "seed": SEED,
        "analysis_eligible_records": sum(
            row["analysis_eligible"] is True for row in rows
        ),
        "primary_external_records": sum(
            row["primary_external_partition"] == "frozen_external_test" for row in rows
        ),
        "eligible_family_support_after_duplicate_control": [
            {
                "sport_family": family,
                "records": count,
                "users": len(family_users[family]),
            }
            for family, count in family_records.most_common()
        ],
        "secondary_adaptation_partition_users": {
            partition: len(users)
            for partition, users in secondary_partition_users.items()
        },
    }
    return rows, summary


def assert_user_disjoint(rows: list[dict[str, object]], partition_field: str) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        if row["analysis_eligible"] is not True:
            continue
        user = str(row["user_id"])
        partition = str(row[partition_field])
        if partition == "not_primary_external":
            continue
        previous = seen.setdefault(user, partition)
        if previous != partition:
            raise AssertionError(f"user {user} appears in {previous} and {partition}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build leakage-controlled session split manifests.")
    parser.add_argument("--endomondo-quality", type=Path, required=True)
    parser.add_argument("--endomondo-duplicates", type=Path, required=True)
    parser.add_argument("--golden-quality", type=Path, required=True)
    parser.add_argument("--golden-duplicates", type=Path, required=True)
    parser.add_argument("--endomondo-output", type=Path, required=True)
    parser.add_argument("--golden-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    endomondo_rows, endomondo_summary = build_endomondo(
        args.endomondo_quality, args.endomondo_duplicates
    )
    golden_rows, golden_summary = build_golden(
        args.golden_quality, args.golden_duplicates
    )
    assert_user_disjoint(endomondo_rows, "unseen_user_partition")
    assert_user_disjoint(golden_rows, "secondary_adaptation_partition")
    write_csv_atomic(args.endomondo_output, endomondo_rows)
    write_csv_atomic(args.golden_output, golden_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_version": "0.2.0",
        "endomondo": endomondo_summary,
        "goldencheetah": golden_summary,
        "assertions": {
            "endomondo_unseen_user_disjoint": True,
            "goldencheetah_secondary_adaptation_user_disjoint": True,
            "split_before_window_generation": True,
        },
        "manifests": {
            "endomondo": str(args.endomondo_output),
            "goldencheetah": str(args.golden_output),
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json_output.with_suffix(args.json_output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.json_output)
    print(json.dumps(summary, ensure_ascii=False)[:10000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
