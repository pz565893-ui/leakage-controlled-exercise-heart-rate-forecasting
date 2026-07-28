from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_public_release_integrity import (  # noqa: E402
    AllowRule,
    DEFAULT_ALLOW_RULES,
    ReleaseSafetyError,
    collect_entries,
    inspect_candidate,
)


class PublicReleaseIntegrityTests(unittest.TestCase):
    def test_safe_file_has_relative_path_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("safe\n", encoding="utf-8")
            entries = collect_entries(
                root, (AllowRule("release_metadata", ("README.md",)),)
            )
            self.assertEqual(entries[0]["relative_path"], "README.md")
            self.assertEqual(entries[0]["size_bytes"], (root / "README.md").stat().st_size)
            self.assertEqual(len(str(entries[0]["sha256"])), 64)

    def test_missing_allowlisted_pattern_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ReleaseSafetyError, "allowlist pattern has no files"):
                collect_entries(
                    root, (AllowRule("release_metadata", ("README.md",)),)
                )

    def test_sensitive_csv_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "outputs" / "results"
            results.mkdir(parents=True)
            path = results / "unsafe.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["user_id", "mae_bpm"])
                writer.writerow(["example", "1.0"])
            with self.assertRaisesRegex(ReleaseSafetyError, "sensitive tabular field"):
                inspect_candidate(path, root)

    def test_small_aggregate_user_group_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "subgroup.csv"
            path.write_text(
                "recorded_group,users,mae_bpm\nunknown,4,7.1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ReleaseSafetyError, "below k=10"):
                inspect_candidate(path, root)

    def test_raw_label_ontology_is_blocked_even_if_explicitly_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "configs" / "sport_ontology_v0_2_0.csv"
            path.parent.mkdir(parents=True)
            path.write_text(
                "source,raw_label,provisional_family,users\n"
                "GoldenCheetah,Example route,running,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ReleaseSafetyError, "privacy- or rights-blocked"):
                inspect_candidate(path, root)

    def test_graphical_abstract_is_blocked_until_rights_are_approved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "figures" / "Graphical_Abstract.svg"
            path.parent.mkdir(parents=True)
            path.write_text("<svg></svg>\n", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseSafetyError, "privacy- or rights-blocked"):
                inspect_candidate(path, root)

    def test_pseudonymous_user_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "per_user_summary.csv"
            path.write_text(
                "user_index,mean_paired_loss_difference_bpm\n0,-0.1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ReleaseSafetyError, "sensitive tabular field"):
                inspect_candidate(path, root)

    def test_multiseed_release_paths_are_explicit_and_exclude_private_levels(self) -> None:
        aggregate_rule = next(
            rule for rule in DEFAULT_ALLOW_RULES if rule.category == "aggregate_results"
        )
        patterns = set(aggregate_rule.patterns)
        expected = {
            "outputs/q1_multiseed_v0_21_0/aggregation/main_history_difference_summary_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/main_history_seed_paired_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/main_vs_comparator_seed_paired_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/main_vs_comparator_summary_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/per_seed_metrics_long_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/seed_variability_summary_v0_22_0.csv",
            "outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrasts_per_seed_v0_23_0.csv",
            "outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrast_seed_summary_v0_23_0.csv",
            "outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrast_user_bootstrap_v0_23_0.csv",
            "outputs/results/external_sport_uncertainty_standardization_v0_24_0.csv",
            "outputs/results/multiseed_balanced_calibration_differences_v0_24_0.csv",
            "outputs/results/multiseed_balanced_calibration_difference_summary_v0_24_0.csv",
            "outputs/results/multiseed_balanced_calibration_per_seed_v0_24_0.csv",
            "outputs/results/multiseed_balanced_calibration_summary_v0_24_0.csv",
            "outputs/results/multiseed_paired_model_comparisons_v0_25_0.csv",
            "outputs/results/multiseed_paired_sport_shift_v0_25_0.csv",
        }
        excluded = {
            "outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrast_user_seed_mean_v0_23_0.csv",
            "outputs/independent_zero_history_v0_23_0/aggregation/progress_manifest.json",
            "outputs/independent_zero_history_v0_23_0/aggregation/aggregation_audit_v0_23_0.json",
            "outputs/q1_multiseed_v0_21_0/aggregation/progress_manifest.json",
            "outputs/q1_multiseed_v0_21_0/aggregation/aggregation_audit_v0_22_0.json",
        }
        self.assertTrue(expected <= patterns)
        self.assertTrue(patterns.isdisjoint(excluded))
        self.assertFalse(
            any(
                "q1_multiseed_v0_21_0" in pattern
                or "independent_zero_history_v0_23_0" in pattern
                for pattern in patterns
                if "*" in pattern
            )
        )

    def test_sensitive_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "summary.json"
            path.write_text('{"records": [{"session_key": "example"}]}\n', encoding="utf-8")
            with self.assertRaisesRegex(ReleaseSafetyError, "sensitive JSON key"):
                inspect_candidate(path, root)

    def test_blocked_intermediate_location_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "outputs" / "manifests"
            manifests.mkdir(parents=True)
            path = manifests / "apparently_safe.csv"
            path.write_text("aggregate,value\nall,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseSafetyError, "blocked release location"):
                inspect_candidate(path, root)

    def test_local_absolute_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "README.md"
            separator = chr(92)
            local_path = separator.join(("C:", "Users", "example", "private.csv"))
            path.write_text(f"Do not release {local_path}\n", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseSafetyError, "local absolute path"):
                inspect_candidate(path, root)


if __name__ == "__main__":
    unittest.main()
