from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ANALYSIS_VERSION = "0.21.0"
PHASES = (
    "unseen_main",
    "unseen_baselines",
    "temporal_main",
    "temporal_baselines",
    "held_sport",
)

# Windows may surface STATUS_STACK_BUFFER_OVERRUN either as its unsigned NTSTATUS
# value or as the corresponding signed 32-bit integer.  This is the only native
# shutdown status observed after PyTorch had atomically written every declared
# output.  Ordinary Python failures must never be recovered merely because an
# older passing audit happens to be present.
RECOVERABLE_NATIVE_SHUTDOWN_CODES = frozenset({3221226505, -1073740791})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def audit_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            isinstance(payload, dict)
            and payload.get("all_assertions_pass") is True
        )
    except (OSError, json.JSONDecodeError):
        return False


def audit_postcondition_passed(
    audit_path: Path,
    required_artifacts: Iterable[Path],
    expected_fields: dict[str, object] | None = None,
) -> bool:
    """Accept a completed job only when its written audit and artifacts are coherent.

    On Windows, a CUDA process can exceptionally report a native shutdown status
    after it has atomically written a passing audit and all declared outputs.  This
    validator is deliberately stricter than ``audit_passed`` so the queue may
    distinguish that post-completion shutdown event from a training failure.
    """
    if not audit_passed(audit_path):
        return False
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if expected_fields:
        for key, value in expected_fields.items():
            observed: object = payload
            for component in key.split("."):
                if not isinstance(observed, dict) or component not in observed:
                    return False
                observed = observed[component]
            if observed != value:
                return False
    paths = [Path(path) for path in required_artifacts]
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, (str, os.PathLike)) and checkpoint:
        paths.append(Path(checkpoint))
    if not paths:
        return False
    try:
        if any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
            return False
        for key in (
            "prediction_rows",
            "metric_rows",
            "point_metric_rows",
            "interval_metric_rows",
            "external_rows",
            "test_rows",
        ):
            if key in payload and int(payload[key]) <= 0:
                return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def recoverable_native_shutdown(return_code: int) -> bool:
    return return_code in RECOVERABLE_NATIVE_SHUTDOWN_CODES


def argument_list(parameters: dict[str, object], names: Iterable[str]) -> list[str]:
    output: list[str] = []
    for name in names:
        value = parameters[name]
        output.extend([f"--{name.replace('_', '-')}", str(value)])
    return output


class QueueRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        configuration: Path,
        output_root: Path,
        dry_run: bool,
    ) -> None:
        self.project_root = project_root
        self.configuration_path = configuration
        self.config = json.loads(configuration.read_text(encoding="utf-8"))
        if self.config.get("analysis_version") != ANALYSIS_VERSION:
            raise AssertionError("configuration version mismatch")
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = output_root / "queue_manifest.json"
        self.dry_run = dry_run
        configuration_hash = sha256_file(configuration)
        if self.manifest_path.exists():
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
            if self.manifest.get("configuration_sha256") != configuration_hash:
                raise AssertionError(
                    "configuration changed after queue initialization; use a new output root"
                )
        else:
            self.manifest = {
                "analysis_version": ANALYSIS_VERSION,
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "configuration": str(configuration.resolve()),
                "configuration_sha256": configuration_hash,
                "python": sys.executable,
                "project_root": str(project_root),
                "tasks": {},
            }
            self.save_manifest()

    def save_manifest(self) -> None:
        self.manifest["updated_at_utc"] = utc_now()
        atomic_json(self.manifest_path, self.manifest)

    def task_update(self, key: str, **updates: object) -> None:
        tasks = self.manifest.setdefault("tasks", {})
        task = tasks.setdefault(key, {})
        task.update(updates)
        self.save_manifest()

    def task_completed(self, key: str, **updates: object) -> None:
        """Write an unambiguous terminal-success state to the queue manifest."""
        existing = self.manifest.setdefault("tasks", {}).setdefault(key, {})
        recorded_recovery = existing.get("recovered_native_shutdown_return_code")
        if recorded_recovery is not None:
            try:
                recovery_code = int(recorded_recovery)
            except (TypeError, ValueError) as error:
                raise AssertionError(
                    f"{key}: malformed recorded native-shutdown code"
                ) from error
            if not recoverable_native_shutdown(recovery_code):
                raise AssertionError(
                    f"{key}: unrecognized nonzero exit was previously recovered: "
                    f"{recovery_code}"
                )
        terminal: dict[str, object] = {
            "status": "completed",
            "current_step": None,
            "failed_step": None,
            "return_code": None,
            "finished_at_utc": utc_now(),
        }
        terminal.update(updates)
        self.task_update(key, **terminal)

    def run_command(
        self,
        *,
        task_key: str,
        step: str,
        command: list[str],
        log_path: Path,
        postcondition_audit: Path | None = None,
        postcondition_artifacts: Iterable[Path] = (),
        expected_audit_fields: dict[str, object] | None = None,
    ) -> None:
        command_text = subprocess.list2cmdline(command)
        self.task_update(
            task_key,
            status="running",
            current_step=step,
            step_started_at_utc=utc_now(),
            command=command_text,
            log=str(log_path),
            failed_step=None,
            return_code=None,
            finished_at_utc=None,
            recovered_native_shutdown_return_code=None,
            postcondition_audit_sha256=None,
            resumed_and_verified=None,
        )
        if self.dry_run:
            self.task_update(task_key, status="dry_run", current_step=step)
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] {command_text}\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=self.project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log.write(f"[{utc_now()}] return_code={result.returncode}\n")
        if (
            recoverable_native_shutdown(result.returncode)
            and postcondition_audit is not None
            and audit_postcondition_passed(
                postcondition_audit,
                postcondition_artifacts,
                expected_audit_fields,
            )
        ):
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"[{utc_now()}] recovered_native_shutdown_after_verified_outputs="
                    f"{result.returncode}\n"
                )
            self.task_update(
                task_key,
                status="running",
                recovered_native_shutdown_return_code=result.returncode,
                postcondition_audit_sha256=sha256_file(postcondition_audit),
            )
            return
        if result.returncode != 0:
            self.task_update(
                task_key,
                status="failed",
                failed_step=step,
                return_code=result.returncode,
                finished_at_utc=utc_now(),
            )
            raise RuntimeError(f"{task_key}/{step} failed; see {log_path}")

    def relative(self, configured_path: str) -> Path:
        return (self.project_root / configured_path).resolve()

    def main_seed(self, seed: int) -> None:
        key = f"seed_{seed}/unseen_main"
        root = self.output_root / f"seed_{seed}" / "unseen_main"
        model = root / "model"
        development_audit = root / "development_audit.json"
        freeze_record = root / "freeze_record.json"
        external_audit = root / "external_audit.json"
        if audit_passed(development_audit) and audit_passed(external_audit):
            self.task_update(
                key, status="completed", resumed_and_verified=True, finished_at_utc=utc_now()
            )
            return
        parameters = self.config["unseen_user_main"]
        array_dir = self.relative(self.config["data_boundaries"]["array_dir"])
        train_command = [
            sys.executable,
            str(self.project_root / "src" / "train_uncertainty_model.py"),
            "--array-dir",
            str(array_dir),
            "--output-dir",
            str(model),
            "--predictions",
            str(root / "development_predictions.npz"),
            "--point-metrics",
            str(root / "development_point_metrics.csv"),
            "--uncertainty-metrics",
            str(root / "development_interval_metrics.csv"),
            "--audit",
            str(development_audit),
            "--seed",
            str(seed),
            "--development-only",
        ] + argument_list(
            parameters,
            (
                "epoch_samples",
                "batch_size",
                "inference_batch_size",
                "max_epochs",
                "patience",
                "learning_rate",
                "history_dropout",
                "signal_input",
            ),
        )
        if not audit_passed(development_audit):
            if freeze_record.exists():
                raise AssertionError(
                    f"immutable freeze record exists without a valid development audit: {freeze_record}"
                )
            self.run_command(
                task_key=key,
                step="development_training",
                command=train_command,
                log_path=root / "development_training.log",
            )
            if self.dry_run:
                return
            if not audit_passed(development_audit):
                raise AssertionError("development audit did not pass after training")
        if not freeze_record.exists():
            freeze_command = [
                sys.executable,
                str(self.project_root / "src" / "write_external_freeze_record.py"),
                "--seed",
                str(seed),
                "--configuration",
                str(self.configuration_path),
                "--development-audit",
                str(development_audit),
                "--checkpoint",
                str(model / "history_quantile_tcn_best_v0_11_0.pt"),
                "--thresholds",
                str(model / "conformal_thresholds_v0_11_0.json"),
                "--input-normalization",
                str(model / "normalization_unseen_user_train.json"),
                "--history-normalization",
                str(model / "history_normalization_unseen_user_train.json"),
                "--training-command",
                subprocess.list2cmdline(train_command),
                "--output",
                str(freeze_record),
            ]
            self.run_command(
                task_key=key,
                step="freeze_record",
                command=freeze_command,
                log_path=root / "freeze_record.log",
            )
            if self.dry_run:
                return
        if not audit_passed(external_audit):
            external_command = [
                sys.executable,
                str(
                    self.project_root
                    / "src"
                    / "infer_frozen_external_uncertainty.py"
                ),
                "--array-dir",
                str(array_dir),
                "--checkpoint",
                str(model / "history_quantile_tcn_best_v0_11_0.pt"),
                "--thresholds",
                str(model / "conformal_thresholds_v0_11_0.json"),
                "--input-normalization",
                str(model / "normalization_unseen_user_train.json"),
                "--history-normalization",
                str(model / "history_normalization_unseen_user_train.json"),
                "--freeze-record",
                str(freeze_record),
                "--predictions",
                str(root / "external_predictions.npz"),
                "--point-metrics",
                str(root / "external_point_metrics.csv"),
                "--interval-metrics",
                str(root / "external_interval_metrics.csv"),
                "--audit",
                str(external_audit),
                "--inference-batch-size",
                str(parameters["inference_batch_size"]),
                "--signal-input",
                str(parameters["signal_input"]),
            ]
            self.run_command(
                task_key=key,
                step="frozen_external_inference",
                command=external_command,
                log_path=root / "external_inference.log",
            )
            if self.dry_run:
                return
        if not audit_passed(external_audit):
            raise AssertionError("external audit did not pass")
        self.task_update(
            key,
            status="completed",
            current_step=None,
            development_audit_sha256=sha256_file(development_audit),
            freeze_record_sha256=sha256_file(freeze_record),
            external_audit_sha256=sha256_file(external_audit),
            finished_at_utc=utc_now(),
        )

    def unseen_baseline(self, seed: int, model_name: str) -> None:
        key = f"seed_{seed}/unseen_{model_name}"
        root = self.output_root / f"seed_{seed}" / f"unseen_{model_name}"
        audit = root / "audit.json"
        parameters = self.config["learned_comparators"]
        required_artifacts = (
            root / "predictions.npz",
            root / "point_metrics.csv",
            root / "model" / f"{model_name}_best_v0_9_0.pt",
            root / "model" / "normalization_unseen_user_train.json",
        )
        expected_fields = {
            "seed": seed,
            "model": model_name,
            "protocol": "unseen_user_train",
            "epoch_samples": parameters["epoch_samples"],
            "batch_size": parameters["batch_size"],
            "inference_batch_size": parameters["inference_batch_size"],
            "max_epochs": parameters["max_epochs"],
            "patience": parameters["patience"],
            "learning_rate": parameters["learning_rate"],
            "checkpoint": str(
                (root / "model" / f"{model_name}_best_v0_9_0.pt").resolve()
            ),
            "normalization_file": str(
                (root / "model" / "normalization_unseen_user_train.json").resolve()
            ),
        }
        if audit_postcondition_passed(audit, required_artifacts, expected_fields):
            self.task_completed(key, resumed_and_verified=True)
            return
        command = [
            sys.executable,
            str(self.project_root / "src" / "train_neural_baselines.py"),
            "--model",
            model_name,
            "--array-dir",
            str(self.relative(self.config["data_boundaries"]["array_dir"])),
            "--output-dir",
            str(root / "model"),
            "--predictions",
            str(root / "predictions.npz"),
            "--metrics",
            str(root / "point_metrics.csv"),
            "--audit",
            str(audit),
            "--seed",
            str(seed),
        ] + argument_list(
            parameters,
            (
                "epoch_samples",
                "batch_size",
                "inference_batch_size",
                "max_epochs",
                "patience",
                "learning_rate",
            ),
        )
        self.run_command(
            task_key=key,
            step="training_and_evaluation",
            command=command,
            log_path=root / "training.log",
            postcondition_audit=audit,
            postcondition_artifacts=required_artifacts,
            expected_audit_fields=expected_fields,
        )
        if not self.dry_run and not audit_postcondition_passed(
            audit, required_artifacts, expected_fields
        ):
            raise AssertionError(f"{key}: audit did not pass")
        if not self.dry_run:
            self.task_completed(key, audit_sha256=sha256_file(audit))

    def temporal_main(self, seed: int) -> None:
        key = f"seed_{seed}/temporal_main"
        root = self.output_root / f"seed_{seed}" / "temporal_main"
        audit = root / "audit.json"
        parameters = self.config["strict_temporal_main"]
        required_artifacts = (
            root / "predictions.npz",
            root / "point_metrics.csv",
            root / "interval_metrics.csv",
            root / "model" / "temporal_history_quantile_tcn_best.pt",
            root / "model" / "input_normalization.json",
            root / "model" / "history_normalization.json",
            root / "model" / "conformal_thresholds.json",
        )
        expected_fields = {
            "seed": seed,
            "protocol": "strict within-user temporal forecasting",
            "epoch_samples": parameters["epoch_samples"],
            "batch_size": parameters["batch_size"],
            "inference_batch_size": parameters["inference_batch_size"],
            "max_epochs": parameters["max_epochs"],
            "patience": parameters["patience"],
            "learning_rate": parameters["learning_rate"],
            "history_dropout": parameters["history_dropout"],
        }
        if audit_postcondition_passed(audit, required_artifacts, expected_fields):
            self.task_completed(key, resumed_and_verified=True)
            return
        command = [
            sys.executable,
            str(self.project_root / "src" / "train_temporal_uncertainty_model.py"),
            "--array-dir",
            str(self.relative(self.config["data_boundaries"]["array_dir"])),
            "--temporal-partition",
            str(
                self.relative(
                    self.config["data_boundaries"]["strict_temporal_partition"]
                )
            ),
            "--temporal-audit",
            str(
                self.relative(
                    self.config["data_boundaries"]["strict_temporal_audit"]
                )
            ),
            "--output-dir",
            str(root / "model"),
            "--predictions",
            str(root / "predictions.npz"),
            "--point-metrics",
            str(root / "point_metrics.csv"),
            "--interval-metrics",
            str(root / "interval_metrics.csv"),
            "--audit",
            str(audit),
            "--seed",
            str(seed),
        ] + argument_list(
            parameters,
            (
                "epoch_samples",
                "batch_size",
                "inference_batch_size",
                "max_epochs",
                "patience",
                "learning_rate",
                "history_dropout",
            ),
        )
        self.run_command(
            task_key=key,
            step="training_and_evaluation",
            command=command,
            log_path=root / "training.log",
            postcondition_audit=audit,
            postcondition_artifacts=required_artifacts,
            expected_audit_fields=expected_fields,
        )
        if not self.dry_run and not audit_postcondition_passed(
            audit, required_artifacts, expected_fields
        ):
            raise AssertionError(f"{key}: audit did not pass")
        if not self.dry_run:
            self.task_completed(key, audit_sha256=sha256_file(audit))

    def temporal_baseline(self, seed: int, model_name: str) -> None:
        key = f"seed_{seed}/temporal_{model_name}"
        root = self.output_root / f"seed_{seed}" / f"temporal_{model_name}"
        audit = root / "audit.json"
        script = self.project_root / "src" / "train_temporal_neural_baselines.py"
        if not script.exists():
            raise FileNotFoundError(
                "strict-temporal learned-baseline script is not available yet"
            )
        parameters = self.config["learned_comparators"]
        required_artifacts = (
            root / "predictions.npz",
            root / "point_metrics.csv",
            root / "model" / f"{model_name}_best.pt",
            root / "model" / "resolved_config.json",
            root / "model" / "normalization_temporal_train.json",
        )
        expected_fields = {
            "seed": seed,
            "model": model_name,
            "protocol": "strict within-user temporal learned baseline",
            "run_purpose": "formal",
            "eligible_for_manuscript_results": True,
            "formal_budget_locked": True,
            "resolved_hyperparameters.epoch_samples": parameters["epoch_samples"],
            "resolved_hyperparameters.batch_size": parameters["batch_size"],
            "resolved_hyperparameters.inference_batch_size": parameters[
                "inference_batch_size"
            ],
            "resolved_hyperparameters.max_epochs": parameters["max_epochs"],
            "resolved_hyperparameters.patience": parameters["patience"],
            "resolved_hyperparameters.learning_rate": parameters["learning_rate"],
        }
        if audit_postcondition_passed(audit, required_artifacts, expected_fields):
            self.task_completed(key, resumed_and_verified=True)
            return
        command = [
            sys.executable,
            str(script),
            "--model",
            model_name,
            "--array-dir",
            str(self.relative(self.config["data_boundaries"]["array_dir"])),
            "--temporal-partition",
            str(
                self.relative(
                    self.config["data_boundaries"]["strict_temporal_partition"]
                )
            ),
            "--temporal-audit",
            str(
                self.relative(
                    self.config["data_boundaries"]["strict_temporal_audit"]
                )
            ),
            "--output-dir",
            str(root / "model"),
            "--predictions",
            str(root / "predictions.npz"),
            "--metrics",
            str(root / "point_metrics.csv"),
            "--audit",
            str(audit),
            "--seed",
            str(seed),
        ] + argument_list(
            parameters,
            (
                "epoch_samples",
                "batch_size",
                "inference_batch_size",
                "max_epochs",
                "patience",
                "learning_rate",
            ),
        )
        self.run_command(
            task_key=key,
            step="training_and_evaluation",
            command=command,
            log_path=root / "training.log",
            postcondition_audit=audit,
            postcondition_artifacts=required_artifacts,
            expected_audit_fields=expected_fields,
        )
        if not self.dry_run and not audit_postcondition_passed(
            audit, required_artifacts, expected_fields
        ):
            raise AssertionError(f"{key}: audit did not pass")
        if not self.dry_run:
            self.task_completed(key, audit_sha256=sha256_file(audit))

    def held_sport(self, seed: int, code: int, family: str) -> None:
        key = f"seed_{seed}/held_sport/{family}"
        root = self.output_root / f"seed_{seed}" / "held_sport" / family
        audit = root / "audit.json"
        parameters = self.config["held_sport_main"]
        required_artifacts = (
            root / "predictions.npz",
            root / "point_metrics.csv",
            root / "interval_metrics.csv",
            root / "model" / f"{family}_best.pt",
            root / "model" / "input_normalization.json",
            root / "model" / "history_normalization.json",
            root / "model" / "conformal_thresholds.json",
        )
        expected_fields = {
            "seed": seed,
            "held_sport_code": code,
            "held_sport_family": family,
            "epoch_samples": parameters["epoch_samples"],
            "batch_size": parameters["batch_size"],
            "inference_batch_size": parameters["inference_batch_size"],
            "max_epochs": parameters["max_epochs"],
            "patience": parameters["patience"],
            "learning_rate": parameters["learning_rate"],
            "history_dropout": parameters["history_dropout"],
            "sport_token_dropout": parameters["sport_dropout"],
        }
        if audit_postcondition_passed(audit, required_artifacts, expected_fields):
            self.task_completed(key, resumed_and_verified=True)
            return
        history_root = self.relative(
            self.config["data_boundaries"]["held_sport_history_root"]
        )
        command = [
            sys.executable,
            str(self.project_root / "src" / "train_sport_shift_model.py"),
            "--held-sport-code",
            str(code),
            "--array-dir",
            str(self.relative(self.config["data_boundaries"]["array_dir"])),
            "--history-dir",
            str(history_root / family),
            "--output-dir",
            str(root / "model"),
            "--predictions",
            str(root / "predictions.npz"),
            "--point-metrics",
            str(root / "point_metrics.csv"),
            "--interval-metrics",
            str(root / "interval_metrics.csv"),
            "--audit",
            str(audit),
            "--seed",
            str(seed),
        ] + argument_list(
            parameters,
            (
                "epoch_samples",
                "batch_size",
                "inference_batch_size",
                "max_epochs",
                "patience",
                "learning_rate",
                "history_dropout",
                "sport_dropout",
            ),
        )
        self.run_command(
            task_key=key,
            step="training_and_evaluation",
            command=command,
            log_path=root / "training.log",
            postcondition_audit=audit,
            postcondition_artifacts=required_artifacts,
            expected_audit_fields=expected_fields,
        )
        if not self.dry_run and not audit_postcondition_passed(
            audit, required_artifacts, expected_fields
        ):
            raise AssertionError(f"{key}: audit did not pass")
        if not self.dry_run:
            self.task_completed(key, audit_sha256=sha256_file(audit))

    def run(self, phases: list[str], selected_seeds: set[int] | None) -> None:
        self.manifest["last_requested_phases"] = list(phases)
        self.manifest["last_requested_seed_subset"] = (
            sorted(selected_seeds) if selected_seeds is not None else None
        )
        self.manifest["last_run_started_at_utc"] = utc_now()
        self.manifest["last_run_finished_at_utc"] = None
        self.save_manifest()
        primary = [
            int(seed)
            for seed in self.config["seeds"]["primary_models"]
            if selected_seeds is None or int(seed) in selected_seeds
        ]
        comparators = [
            int(seed)
            for seed in self.config["seeds"]["learned_comparators"]
            if selected_seeds is None or int(seed) in selected_seeds
        ]
        held = [
            int(seed)
            for seed in self.config["seeds"]["held_sport_models"]
            if selected_seeds is None or int(seed) in selected_seeds
        ]
        for phase in phases:
            if phase == "unseen_main":
                for seed in primary:
                    self.main_seed(seed)
            elif phase == "unseen_baselines":
                for seed in comparators:
                    for model_name in self.config["learned_comparators"]["models"]:
                        self.unseen_baseline(seed, model_name)
            elif phase == "temporal_main":
                for seed in primary:
                    self.temporal_main(seed)
            elif phase == "temporal_baselines":
                for seed in comparators:
                    for model_name in self.config["learned_comparators"]["models"]:
                        self.temporal_baseline(seed, model_name)
            elif phase == "held_sport":
                codes = self.config["held_sport_main"]["sport_codes"]
                families = self.config["held_sport_main"]["sport_families"]
                if len(codes) != len(families):
                    raise AssertionError("held-sport code/family length mismatch")
                for seed in held:
                    for code, family in zip(codes, families, strict=True):
                        self.held_sport(seed, int(code), str(family))
            else:
                raise ValueError(f"unknown phase: {phase}")
        self.manifest["last_run_finished_at_utc"] = utc_now()
        self.save_manifest()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the resumable Q1 multi-seed robustness queue."
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        default=project_root / "configs" / "q1_multiseed_v0_21_0.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "outputs" / "q1_multiseed_v0_21_0",
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=PHASES,
        help="Repeat to select phases; default runs all phases in registered order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Optional seed subset; repeat for more than one seed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configuration = args.configuration.resolve()
    output_root = args.output_root.resolve()
    lock_path = output_root / "queue.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        lock_handle = lock_path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"queue lock exists; another queue may be active: {lock_path}"
        ) from error
    try:
        lock_handle.write(
            json.dumps({"pid": os.getpid(), "started_at_utc": utc_now()})
        )
        lock_handle.close()
        runner = QueueRunner(
            project_root=project_root,
            configuration=configuration,
            output_root=output_root,
            dry_run=args.dry_run,
        )
        runner.run(args.phase or list(PHASES), set(args.seed) if args.seed else None)
    finally:
        if lock_path.exists():
            lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
