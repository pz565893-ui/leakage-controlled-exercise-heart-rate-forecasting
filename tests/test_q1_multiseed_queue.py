from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_q1_multiseed_queue import (  # noqa: E402
    PHASES,
    QueueRunner,
    audit_postcondition_passed,
    recoverable_native_shutdown,
)


class AuditPostconditionTests(unittest.TestCase):
    def make_complete_job(self, root: Path) -> tuple[Path, list[Path]]:
        checkpoint = root / "model.pt"
        predictions = root / "predictions.npz"
        metrics = root / "metrics.csv"
        for path in (checkpoint, predictions, metrics):
            path.write_bytes(b"complete")
        audit = root / "audit.json"
        audit.write_text(
            json.dumps(
                {
                    "all_assertions_pass": True,
                    "seed": 17,
                    "model": "gru",
                    "checkpoint": str(checkpoint),
                    "prediction_rows": 100,
                    "metric_rows": 18,
                }
            ),
            encoding="utf-8",
        )
        return audit, [predictions, metrics]

    def test_complete_matching_job_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audit, artifacts = self.make_complete_job(Path(temporary))
            self.assertTrue(
                audit_postcondition_passed(
                    audit,
                    artifacts,
                    {"seed": 17, "model": "gru"},
                )
            )

    def test_missing_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audit, artifacts = self.make_complete_job(Path(temporary))
            artifacts[0].unlink()
            self.assertFalse(audit_postcondition_passed(audit, artifacts))

    def test_non_object_audit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "predictions.npz"
            artifact.write_bytes(b"complete")
            audit = root / "audit.json"
            audit.write_text(
                json.dumps([{"all_assertions_pass": True}]), encoding="utf-8"
            )
            self.assertFalse(audit_postcondition_passed(audit, [artifact]))

    def test_wrong_seed_or_failed_audit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audit, artifacts = self.make_complete_job(Path(temporary))
            self.assertFalse(
                audit_postcondition_passed(audit, artifacts, {"seed": 18})
            )
            payload = json.loads(audit.read_text(encoding="utf-8"))
            payload["all_assertions_pass"] = False
            audit.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(audit_postcondition_passed(audit, artifacts))

    def test_nested_expected_field_and_row_counts_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audit, artifacts = self.make_complete_job(Path(temporary))
            payload = json.loads(audit.read_text(encoding="utf-8"))
            payload["resolved_hyperparameters"] = {"epoch_samples": 500_000}
            payload["point_metric_rows"] = 18
            audit.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(
                audit_postcondition_passed(
                    audit,
                    artifacts,
                    {"resolved_hyperparameters.epoch_samples": 500_000},
                )
            )
            payload["point_metric_rows"] = "not-an-integer"
            audit.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(audit_postcondition_passed(audit, artifacts))

    def test_only_observed_native_shutdown_status_is_recoverable(self) -> None:
        self.assertTrue(recoverable_native_shutdown(3221226505))
        self.assertTrue(recoverable_native_shutdown(-1073740791))
        self.assertFalse(recoverable_native_shutdown(1))
        self.assertFalse(recoverable_native_shutdown(3221225477))

    def make_runner(self, root: Path) -> QueueRunner:
        configuration = root / "config.json"
        configuration.write_text(
            json.dumps({"analysis_version": "0.21.0"}), encoding="utf-8"
        )
        return QueueRunner(
            project_root=root,
            configuration=configuration,
            output_root=root / "queue",
            dry_run=False,
        )

    def test_ordinary_nonzero_exit_is_not_recovered_by_old_passing_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, artifacts = self.make_complete_job(root)
            runner = self.make_runner(root)
            with patch(
                "run_q1_multiseed_queue.subprocess.run",
                return_value=SimpleNamespace(returncode=1),
            ):
                with self.assertRaises(RuntimeError):
                    runner.run_command(
                        task_key="seed_17/unseen_gru",
                        step="training_and_evaluation",
                        command=["python", "trainer.py"],
                        log_path=root / "job.log",
                        postcondition_audit=audit,
                        postcondition_artifacts=artifacts,
                        expected_audit_fields={"seed": 17, "model": "gru"},
                    )
            task = runner.manifest["tasks"]["seed_17/unseen_gru"]
            self.assertEqual(task["status"], "failed")
            self.assertEqual(task["return_code"], 1)

    def test_verified_native_shutdown_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, artifacts = self.make_complete_job(root)
            runner = self.make_runner(root)
            with patch(
                "run_q1_multiseed_queue.subprocess.run",
                return_value=SimpleNamespace(returncode=3221226505),
            ):
                runner.run_command(
                    task_key="seed_17/unseen_gru",
                    step="training_and_evaluation",
                    command=["python", "trainer.py"],
                    log_path=root / "job.log",
                    postcondition_audit=audit,
                    postcondition_artifacts=artifacts,
                    expected_audit_fields={"seed": 17, "model": "gru"},
                )
            task = runner.manifest["tasks"]["seed_17/unseen_gru"]
            self.assertEqual(task["status"], "running")
            self.assertEqual(
                task["recovered_native_shutdown_return_code"], 3221226505
            )

    def test_terminal_success_rejects_unrecognized_prior_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self.make_runner(Path(temporary))
            key = "seed_17/unseen_gru"
            runner.task_update(
                key,
                status="running",
                recovered_native_shutdown_return_code=1,
            )
            with self.assertRaises(AssertionError):
                runner.task_completed(key)

    def test_terminal_success_clears_stale_failure_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = self.make_runner(Path(temporary))
            key = "seed_17/unseen_gru"
            runner.task_update(
                key,
                status="failed",
                current_step="training",
                failed_step="training",
                return_code=3221226505,
                recovered_native_shutdown_return_code=3221226505,
            )
            runner.task_completed(key, resumed_and_verified=True)
            task = runner.manifest["tasks"][key]
            self.assertEqual(task["status"], "completed")
            self.assertIsNone(task["current_step"])
            self.assertIsNone(task["failed_step"])
            self.assertIsNone(task["return_code"])

    def test_formal_configuration_builds_every_phase_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = QueueRunner(
                project_root=ROOT,
                configuration=ROOT / "configs" / "q1_multiseed_v0_21_0.json",
                output_root=Path(temporary) / "queue",
                dry_run=True,
            )
            runner.run(list(PHASES), {20260722})
            expected = {
                "seed_20260722/unseen_main",
                "seed_20260722/unseen_gru",
                "seed_20260722/unseen_tcn",
                "seed_20260722/temporal_main",
                "seed_20260722/temporal_gru",
                "seed_20260722/temporal_tcn",
                "seed_20260722/held_sport/outdoor_cycling",
                "seed_20260722/held_sport/indoor_virtual_cycling",
                "seed_20260722/held_sport/running",
                "seed_20260722/held_sport/walking_hiking",
                "seed_20260722/held_sport/strength_cross_training",
            }
            self.assertEqual(set(runner.manifest["tasks"]), expected)
            self.assertTrue(
                all(
                    task["status"] == "dry_run"
                    for task in runner.manifest["tasks"].values()
                )
            )
            self.assertEqual(
                runner.manifest["last_requested_phases"], list(PHASES)
            )
            self.assertEqual(
                runner.manifest["last_requested_seed_subset"], [20260722]
            )
            self.assertIsNotNone(runner.manifest["last_run_finished_at_utc"])


if __name__ == "__main__":
    unittest.main()
