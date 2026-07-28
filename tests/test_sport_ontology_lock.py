from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_sport_ontology import (  # noqa: E402
    LOCKED_MAPPING_STATUS,
    RULE_MAPPED_STATUS,
    UNRESOLVED_STATUS,
    mapping_fingerprint,
    ontology_rows,
)


# Captured from the reported v0.2.0 CSV before the status-only lock migration.
PRE_LOCK_MAPPING_FINGERPRINT = (
    "e6253d2f9af73b47daece92a1f5d427863fe4cc041f91dcfb2ca4ed544d85475"
)


class SportOntologyLockTests(unittest.TestCase):
    def test_generated_rows_use_locked_statuses(self) -> None:
        rows = ontology_rows(
            "Synthetic",
            {
                "run": {"records": 1, "users": {"u1"}, "hr_mask_records": None},
                "ambiguous": {
                    "records": 1,
                    "users": {"u2"},
                    "hr_mask_records": None,
                },
            },
        )
        by_label = {str(row["raw_label"]): row for row in rows}
        self.assertEqual(by_label["run"]["review_status"], RULE_MAPPED_STATUS)
        self.assertEqual(
            by_label["ambiguous"]["review_status"], UNRESOLVED_STATUS
        )

    def test_reported_artifacts_are_locked_without_mapping_change(self) -> None:
        csv_path = ROOT / "configs" / "sport_ontology_v0_2_0.csv"
        summary_path = ROOT / "outputs" / "audit" / "sport_ontology_v0_2_0_summary.json"
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 396)
        self.assertEqual(
            len({(row["source"], row["raw_label"]) for row in rows}), len(rows)
        )
        self.assertEqual(summary["status"], LOCKED_MAPPING_STATUS)
        self.assertEqual(summary["mapping_csv"], csv_path.name)
        self.assertEqual(summary["lock_semantics"]["analysis_mapping"], "locked")
        self.assertEqual(
            {row["review_status"] for row in rows},
            {RULE_MAPPED_STATUS, UNRESOLVED_STATUS},
        )
        self.assertTrue(
            all(
                row["review_status"]
                == (
                    UNRESOLVED_STATUS
                    if row["provisional_family"] == "other_unknown"
                    else RULE_MAPPED_STATUS
                )
                for row in rows
            )
        )

        fingerprint = mapping_fingerprint(rows)
        self.assertEqual(fingerprint, PRE_LOCK_MAPPING_FINGERPRINT)
        self.assertEqual(summary["mapping_fingerprint_sha256"], fingerprint)
        self.assertEqual(
            len(summary["manual_review_queue"]),
            sum(row["provisional_family"] == "other_unknown" for row in rows),
        )


if __name__ == "__main__":
    unittest.main()
