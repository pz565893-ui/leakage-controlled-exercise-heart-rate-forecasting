from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_strict_temporal_learned_baseline import (  # noqa: E402
    FORMAL_EPOCH_SAMPLES,
    FORMAL_MAX_EPOCHS,
    FORMAL_PATIENCE,
    pairwise_overlap_counts,
    portable_path,
    resolve_argument_paths,
    resolve_run_paths,
    session_partition_codes,
    split_indices,
    validate_budget,
    validate_existing_outputs,
)


class StrictTemporalLearnedBaselineTests(unittest.TestCase):
    def test_split_indices_use_strict_codes_and_evaluation_filter(self) -> None:
        dataset = np.array([0, 0, 0, 0, 0, 1])
        evaluation = np.array([0, 0, 1, 1, 0, 1])
        temporal = np.array([1, 2, 2, 4, 4, 4])
        result = split_indices(dataset, evaluation, temporal)
        np.testing.assert_array_equal(result["train"], np.array([0]))
        np.testing.assert_array_equal(result["validation"], np.array([2]))
        np.testing.assert_array_equal(result["test"], np.array([3]))
        self.assertEqual(len(result["calibration"]), 0)

    def test_pairwise_overlap_counts_detect_session_leakage(self) -> None:
        result = pairwise_overlap_counts(
            {
                "train": np.array([1, 2]),
                "validation": np.array([2, 3]),
                "test": np.array([4]),
            }
        )
        self.assertEqual(result["train_validation"], 1)
        self.assertEqual(result["train_test"], 0)
        self.assertEqual(result["validation_test"], 0)

    def test_session_partition_codes_reject_cross_partition_session(self) -> None:
        dataset = np.array([0, 0, 0])
        temporal = np.array([1, 2, 4])
        sessions = np.array([10, 10, 20])
        codes, conflicts = session_partition_codes(
            dataset, temporal, sessions, n_sessions=21
        )
        self.assertEqual(conflicts, 1)
        self.assertEqual(codes[10], 2)
        self.assertEqual(codes[20], 4)

    def test_formal_budget_is_locked(self) -> None:
        self.assertTrue(
            validate_budget(
                "formal",
                FORMAL_EPOCH_SAMPLES,
                FORMAL_MAX_EPOCHS,
                FORMAL_PATIENCE,
            )
        )
        with self.assertRaises(ValueError):
            validate_budget("formal", 100, FORMAL_MAX_EPOCHS, FORMAL_PATIENCE)
        self.assertFalse(validate_budget("smoke", 100, 1, 1))

    def test_paths_separate_purpose_model_and_seed(self) -> None:
        paths = resolve_run_paths(Path("out"), "formal", "gru", 123)
        self.assertEqual(paths.run_dir, Path("out/formal/gru/seed_123"))
        self.assertEqual(paths.predictions.name, "strict_temporal_test_predictions.npz")
        smoke = resolve_run_paths(Path("out"), "smoke", "gru", 123)
        self.assertNotEqual(paths.run_dir, smoke.run_dir)

    def test_portable_path_omits_local_absolute_workspace_prefix(self) -> None:
        project_file = ROOT / "src" / "train_temporal_neural_baselines.py"
        self.assertEqual(
            portable_path(project_file),
            "src/train_temporal_neural_baselines.py",
        )
        self.assertNotIn(":", portable_path(project_file))

    def test_explicit_output_paths_match_runner_contract(self) -> None:
        class Arguments:
            model = "tcn"
            output_dir = Path("out/seed_1/temporal_tcn")
            predictions = Path("predictions/tcn_seed_1.npz")
            metrics = Path("results/tcn_seed_1.csv")
            audit = Path("audit/tcn_seed_1.json")

        paths = resolve_argument_paths(Arguments())
        self.assertEqual(paths.run_dir, Arguments.output_dir)
        self.assertEqual(paths.predictions, Arguments.predictions)
        self.assertEqual(paths.metrics, Arguments.metrics)
        self.assertEqual(paths.audit, Arguments.audit)

    def test_incomplete_run_can_restart_but_completed_run_is_protected(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = resolve_run_paths(root, "formal", "gru", 1)
            paths.run_dir.mkdir(parents=True)
            paths.resolved_config.write_text("{}", encoding="utf-8")
            existing = validate_existing_outputs(paths, allow_overwrite=False)
            self.assertEqual(existing, [str(paths.resolved_config)])
            paths.audit.write_text(
                json.dumps({"all_assertions_pass": True}), encoding="utf-8"
            )
            with self.assertRaises(FileExistsError):
                validate_existing_outputs(paths, allow_overwrite=False)
            self.assertTrue(
                validate_existing_outputs(paths, allow_overwrite=True)
            )


if __name__ == "__main__":
    unittest.main()
