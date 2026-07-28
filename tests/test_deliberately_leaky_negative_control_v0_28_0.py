from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aggregate_deliberately_leaky_negative_control_v0_28_0 import (  # noqa: E402
    aggregate,
    hierarchical_user_average,
    hierarchical_user_point_metrics,
    load_prediction_archive,
    percentile_user_bootstrap,
)
from build_deliberately_leaky_temporal_partition_v0_28_0 import (  # noqa: E402
    PARTITION_CALIBRATION,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    construct_partition,
    index_sha256,
    proximity_and_collision_audit,
    stable_window_assignments,
)
from run_deliberately_leaky_negative_control_queue_v0_28_0 import (  # noqa: E402
    EXPECTED_SEEDS,
    PHASES,
    QueueRunner,
)
from train_deliberately_leaky_temporal_negative_control_v0_28_0 import (  # noqa: E402
    load_clean_control_row_index,
    train,
    validate_configuration,
    validate_runtime_budget,
    validate_test_alignment,
)


class DeliberatelyLeakyPartitionTests(unittest.TestCase):
    def test_sha_assignment_is_deterministic_and_namespace_bound(self) -> None:
        sessions = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
        times = np.array([60, 120, 60, 120, 60, 120], dtype=np.float64)
        first = stable_window_assignments(
            sessions,
            times,
            namespace="leaky_negative_control_v0_28_0",
            train_threshold=0.7,
            validation_threshold=0.85,
        )
        second = stable_window_assignments(
            sessions,
            times,
            namespace="leaky_negative_control_v0_28_0",
            train_threshold=0.7,
            validation_threshold=0.85,
        )
        changed = stable_window_assignments(
            sessions,
            times,
            namespace="different_namespace",
            train_threshold=0.7,
            validation_threshold=0.85,
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, changed))
        self.assertTrue(
            set(first.tolist()).issubset(
                {PARTITION_TRAIN, PARTITION_VALIDATION, PARTITION_CALIBRATION}
            )
        )

    def test_fixed_test_is_exact_and_only_same_test_sessions_contaminate(self) -> None:
        rows: list[tuple[int, int, int, int, int, float]] = []
        # dataset, evaluation, strict code, user, session, origin
        for origin in range(60, 660, 60):
            rows.append((0, 0, PARTITION_TRAIN, 1, 10, float(origin)))
        rows.append((0, 1, PARTITION_VALIDATION, 1, 20, 300.0))
        rows.append((0, 0, PARTITION_VALIDATION, 1, 20, 360.0))
        rows.append((0, 1, PARTITION_CALIBRATION, 1, 30, 300.0))
        rows.append((0, 0, PARTITION_CALIBRATION, 1, 30, 360.0))
        for origin in range(60, 3_060, 60):
            evaluation = int(origin % 300 == 0)
            rows.append((0, evaluation, PARTITION_TEST, 1, 40, float(origin)))
        # Strict-test rows from a session with no evaluation origin are excluded.
        rows.extend(
            [
                (0, 0, PARTITION_TEST, 1, 41, 60.0),
                (0, 0, PARTITION_TEST, 1, 41, 120.0),
                (1, 1, PARTITION_TEST, 2, 50, 300.0),
            ]
        )
        matrix = np.asarray(rows, dtype=np.float64)
        dataset = matrix[:, 0].astype(np.uint8)
        evaluation = matrix[:, 1].astype(np.uint8)
        strict = matrix[:, 2].astype(np.uint8)
        users = matrix[:, 3].astype(np.int64)
        sessions = matrix[:, 4].astype(np.int64)
        times = matrix[:, 5]
        partition, audit = construct_partition(
            dataset=dataset,
            evaluation=evaluation,
            strict_partition=strict,
            users=users,
            sessions=sessions,
            origin_times=times,
            namespace="leaky_negative_control_v0_28_0",
            train_threshold=0.7,
            validation_threshold=0.85,
        )
        expected_test = np.flatnonzero(
            (dataset == 0) & (strict == PARTITION_TEST) & (evaluation == 1)
        )
        self.assertTrue(np.array_equal(np.flatnonzero(partition == 4), expected_test))
        self.assertEqual(audit["fixed_test"]["rows"], len(expected_test))
        self.assertFalse(audit["valid_for_generalization"])
        self.assertTrue(audit["all_assertions_pass"])
        self.assertGreater(audit["session_overlap_counts"]["train_test"], 0)
        excluded_session_rows = np.flatnonzero(sessions == 41)
        self.assertTrue(np.all(partition[excluded_session_rows] == 0))
        self.assertFalse(
            any(audit["exact_row_overlap_counts"].values())
        )

    def test_proximity_context_overlap_and_target_collision_are_audited(self) -> None:
        sessions = np.array([7, 7], dtype=np.int64)
        times = np.array([180.0, 300.0])
        audit = proximity_and_collision_audit(
            np.array([1]), np.array([0]), sessions, times
        )
        nearest = audit["nearest_contaminated_train_origin_distance_seconds"]
        self.assertEqual(nearest["minimum"], 120.0)
        overlap = audit["nearest_context_overlap_seconds"]
        self.assertEqual(overlap["median"], 180.0)
        collision = audit["exact_target_timestamp_collisions"]
        self.assertEqual(collision["test_origins_with_any_collision"], 1)
        self.assertEqual(collision["collided_target_slots"], 2)

    def test_row_index_hash_is_explicit_little_endian_int64(self) -> None:
        index = np.array([1, 9, 17], dtype=np.int64)
        self.assertEqual(index_sha256(index), index_sha256(index.astype("<i8")))
        self.assertNotEqual(index_sha256(index), index_sha256(index[::-1]))


class DeliberatelyLeakyTrainingAndQueueTests(unittest.TestCase):
    def config(self) -> dict[str, object]:
        return json.loads(
            (ROOT / "configs" / "leaky_negative_control_v0_28_0.json").read_text(
                encoding="utf-8"
            )
        )

    def test_locked_config_and_budget_reject_mutation(self) -> None:
        config = self.config()
        validate_configuration(config)
        training = config["training"]
        args = argparse.Namespace(
            epoch_samples=500_000,
            batch_size=2_048,
            inference_batch_size=4_096,
            max_epochs=40,
            patience=4,
            learning_rate=0.001,
            history_dropout=0.0,
            seed=20260722,
        )
        validate_runtime_budget(args, training)
        args.max_epochs = 41
        with self.assertRaises(AssertionError):
            validate_runtime_budget(args, training)
        config["valid_for_generalization"] = True
        with self.assertRaises(AssertionError):
            validate_configuration(config)

    def test_test_alignment_rejects_reordering_or_mismatch(self) -> None:
        index = np.array([2, 5, 8], dtype=np.int64)
        expected_hash = index_sha256(index)
        self.assertEqual(validate_test_alignment(index, index.copy(), expected_hash), expected_hash)
        with self.assertRaises(AssertionError):
            validate_test_alignment(index, index[::-1], expected_hash)
        with self.assertRaises(AssertionError):
            validate_test_alignment(index, index.copy(), "0" * 64)

    def test_checkpoint_io_uses_binary_handles_for_non_ascii_windows_paths(self) -> None:
        source = inspect.getsource(train)
        self.assertIn('checkpoint = args.output_dir / "best.pt"', source)
        self.assertIn('checkpoint = args.output_dir / "best.pt"', source)
        self.assertIn('with checkpoint.open("wb") as checkpoint_handle:', source)
        self.assertIn('with checkpoint.open("rb") as checkpoint_handle:', source)
        self.assertIn("torch.save(\n                    {", source)
        self.assertIn("checkpoint_handle,", source)
        self.assertIn("torch.load(\n            checkpoint_handle,", source)
        self.assertNotIn("torch.load(checkpoint,", source)

    @unittest.skipUnless(sys.platform == "win32", "Windows MAX_PATH regression")
    def test_formal_checkpoint_path_stays_below_legacy_windows_limit(self) -> None:
        checkpoint = (
            ROOT
            / "outputs"
            / "deliberately_leaky_negative_control_v0_28_0"
            / f"seed_{EXPECTED_SEEDS[0]}"
            / "m"
            / "best.pt"
        )
        self.assertLess(len(str(checkpoint.resolve())), 260)

    def test_clean_control_loader_requires_zero_history_only_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.npz"
            np.savez_compressed(
                valid,
                row_index=np.array([1, 2]),
                zero_history_quantiles=np.ones((2, 3, 7), dtype=np.float32),
            )
            self.assertTrue(
                np.array_equal(load_clean_control_row_index(valid), [1, 2])
            )
            invalid = root / "invalid.npz"
            np.savez_compressed(
                invalid,
                row_index=np.array([1, 2]),
                zero_history_quantiles=np.ones((2, 3, 7), dtype=np.float32),
                history_quantiles=np.ones((2, 3, 7), dtype=np.float32),
            )
            with self.assertRaises(AssertionError):
                load_clean_control_row_index(invalid)

    def test_queue_requires_acknowledgement_and_dry_run_registers_three_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "queue"
            with self.assertRaises(AssertionError):
                QueueRunner(
                    project_root=ROOT,
                    configuration=ROOT
                    / "configs"
                    / "leaky_negative_control_v0_28_0.json",
                    output_root=output,
                    acknowledge_invalid_generalization=False,
                    dry_run=True,
                )
            runner = QueueRunner(
                project_root=ROOT,
                configuration=ROOT
                / "configs"
                / "leaky_negative_control_v0_28_0.json",
                output_root=output,
                acknowledge_invalid_generalization=True,
                dry_run=True,
            )
            runner.run(list(PHASES), None)
            tasks = runner.manifest["tasks"]
            self.assertEqual(
                set(tasks),
                {"partition", "aggregation"}
                | {f"seed_{seed}" for seed in EXPECTED_SEEDS},
            )
            self.assertIn(tasks["partition"]["status"], {"dry_run", "completed"})
            self.assertTrue(
                all(
                    tasks[key]["status"] == "dry_run"
                    for key in {"aggregation"}
                    | {f"seed_{seed}" for seed in EXPECTED_SEEDS}
                )
            )
            for seed in EXPECTED_SEEDS:
                command = tasks[f"seed_{seed}"]["command"]
                self.assertIn("--acknowledge-invalid-generalization", command)
                self.assertIn("--history-dropout 0.0", command)
            self.assertIn(
                "--acknowledge-invalid-generalization",
                tasks["aggregation"]["command"],
            )


class DeliberatelyLeakyAggregationTests(unittest.TestCase):
    def test_hierarchical_point_metrics_use_session_then_user_weighting(self) -> None:
        prediction = np.array([0.0, 2.0, 10.0, 30.0])
        target = np.zeros(4)
        users = np.array([1, 1, 1, 2])
        sessions = np.array([10, 10, 11, 20])
        user_ids, metrics = hierarchical_user_point_metrics(
            prediction, target, users, sessions
        )
        self.assertTrue(np.array_equal(user_ids, [1, 2]))
        # User 1 session MAEs are 1 and 10, hence equal-session MAE is 5.5.
        self.assertTrue(np.allclose(metrics["mae_bpm"], [5.5, 30.0]))
        self.assertTrue(np.allclose(metrics["bias_bpm"], [5.5, 30.0]))

    def test_hierarchical_average_and_bootstrap_are_deterministic(self) -> None:
        values = np.array([0.0, 2.0, 10.0, 30.0])
        users = np.array([1, 1, 1, 2])
        sessions = np.array([10, 10, 11, 20])
        user_ids, means = hierarchical_user_average(values, users, sessions)
        self.assertTrue(np.array_equal(user_ids, [1, 2]))
        self.assertTrue(np.allclose(means, [5.5, 30.0]))
        first = percentile_user_bootstrap(means, replicates=1_000, seed=9)
        second = percentile_user_bootstrap(means, replicates=1_000, seed=9)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], means.mean())

    def test_prediction_loader_rejects_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.npz"
            np.savez_compressed(
                path,
                row_index=np.array([1, 1]),
                zero_history_quantiles=np.ones((2, 3, 7), dtype=np.float32),
            )
            with self.assertRaises(AssertionError):
                load_prediction_archive(path)

    def test_full_synthetic_three_seed_aggregation_is_paired_and_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = root / "arrays"
            clean_root = root / "clean"
            leaky_root = root / "leaky"
            output = root / "aggregation"
            arrays.mkdir()
            row_index = np.arange(6, dtype=np.int64)
            target = np.column_stack(
                [100.0 + row_index, 102.0 + row_index, 104.0 + row_index]
            ).astype(np.float32)
            np.save(arrays / "targets.npy", target)
            np.save(arrays / "user_index.npy", np.array([0, 0, 0, 1, 1, 1]))
            np.save(arrays / "session_index.npy", np.array([0, 0, 1, 2, 2, 3]))
            offsets = np.array([-10, -5, -2, 0, 2, 5, 10], dtype=np.float32)
            clean_quantiles = target[:, :, None] + offsets[None, None, :] + 2.0
            leaky_quantiles = target[:, :, None] + offsets[None, None, :] + 1.0
            clean_thresholds = {
                "zero_history": {
                    "0.5": [0.0, 0.0, 0.0],
                    "0.8": [0.0, 0.0, 0.0],
                    "0.9": [0.0, 0.0, 0.0],
                }
            }
            leaky_thresholds = {
                "analysis_version": "0.28.0",
                "valid_for_generalization": False,
                "coverage_guarantee_valid": False,
                "thresholds": clean_thresholds["zero_history"],
            }
            for seed in EXPECTED_SEEDS:
                clean_seed = clean_root / f"seed_{seed}" / "strict_temporal"
                leaky_seed = leaky_root / f"seed_{seed}"
                (clean_seed / "m").mkdir(parents=True)
                (leaky_seed / "m").mkdir(parents=True)
                np.savez_compressed(
                    clean_seed / "predictions.npz",
                    row_index=row_index,
                    zero_history_quantiles=clean_quantiles,
                )
                np.savez_compressed(
                    leaky_seed / "predictions.npz",
                    row_index=row_index,
                    zero_history_quantiles=leaky_quantiles,
                )
                (clean_seed / "m" / "conformal_thresholds.json").write_text(
                    json.dumps(clean_thresholds), encoding="utf-8"
                )
                (leaky_seed / "m" / "conformal_thresholds.json").write_text(
                    json.dumps(leaky_thresholds), encoding="utf-8"
                )
                (leaky_seed / "audit.json").write_text(
                    json.dumps(
                        {
                            "all_assertions_pass": True,
                            "valid_for_generalization": False,
                            "protocol": "deliberately_leaky_strict_temporal_negative_control",
                            "seed": seed,
                            "training_history_mode": "always_zero",
                            "clean_checkpoint_reused_or_warm_started": False,
                        }
                    ),
                    encoding="utf-8",
                )
            config = json.loads(
                (
                    ROOT / "configs" / "leaky_negative_control_v0_28_0.json"
                ).read_text(encoding="utf-8")
            )
            config["fixed_test"] = {
                "definition": "synthetic",
                "rows": len(row_index),
                "sessions": 4,
                "users": 2,
                "row_index_sha256_int64_little_endian": index_sha256(row_index),
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            partition_audit = root / "partition_audit.json"
            partition_audit.write_text(
                json.dumps(
                    {
                        "all_assertions_pass": True,
                        "valid_for_generalization": False,
                    }
                ),
                encoding="utf-8",
            )
            result = aggregate(
                argparse.Namespace(
                    acknowledge_invalid_generalization=True,
                    configuration=config_path,
                    array_dir=arrays,
                    leaky_root=leaky_root,
                    clean_root=clean_root,
                    partition_audit=partition_audit,
                    output_dir=output,
                )
            )
            self.assertTrue(result["all_assertions_pass"])
            self.assertFalse(result["valid_for_generalization"])
            self.assertEqual(result["matched_seed_count"], 3)
            with (output / "paired_metrics_per_seed_v0_28_0.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row["valid_for_generalization"] == "False" for row in rows))
            self.assertTrue(
                all(float(row["leaky_minus_clean_mae_bpm"]) < 0 for row in rows)
            )


if __name__ == "__main__":
    unittest.main()
