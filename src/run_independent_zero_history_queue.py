from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from run_q1_multiseed_queue import (
    atomic_json,
    audit_postcondition_passed,
    recoverable_native_shutdown,
    sha256_file,
    utc_now,
)


ANALYSIS_VERSION = "0.23.0"
PHASES = ("unseen_user", "strict_temporal")
EXPECTED_SEEDS = (20260722, 20260723, 20260724, 20260725, 20260726)
TRAINING_HISTORY_MODE = "always_zero"
SIGNAL_INPUT = "multimodal"
SELECTION_METRIC = "mean 1/3/5-min zero-history validation hierarchical MAE"
BUDGET_FIELDS = (
    "epoch_samples",
    "batch_size",
    "inference_batch_size",
    "max_epochs",
    "patience",
    "learning_rate",
)


def nested_value(payload: dict[str, object], dotted_key: str) -> object:
    observed: object = payload
    for component in dotted_key.split("."):
        if not isinstance(observed, dict) or component not in observed:
            raise KeyError(dotted_key)
        observed = observed[component]
    return observed


def positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def model_audit_postcondition_passed(
    audit_path: Path,
    required_artifacts: Iterable[Path],
    expected_fields: dict[str, object],
    positive_count_groups: Iterable[tuple[str, ...]],
) -> bool:
    """Validate a model audit, its declared mode, and nonempty result counts."""
    artifacts = tuple(Path(path) for path in required_artifacts)
    if not audit_postcondition_passed(audit_path, artifacts, expected_fields):
        return False
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        for alternatives in positive_count_groups:
            values = [payload.get(name) for name in alternatives]
            if not any(positive_integer(value) for value in values):
                return False
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return True


def resolved_config_postcondition_passed(
    path: Path, expected_fields: dict[str, object]
) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        return all(nested_value(payload, key) == value for key, value in expected_fields.items())
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False


def zero_history_npz_postcondition_passed(path: Path) -> bool:
    """Fail closed unless the prediction archive has the registered zero-mode schema."""
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"row_index", "zero_history_quantiles"}:
                return False
            row_index = archive["row_index"]
            quantiles = archive["zero_history_quantiles"]
        return (
            row_index.ndim == 1
            and len(row_index) > 0
            and np.issubdtype(row_index.dtype, np.integer)
            and quantiles.ndim == 3
            and quantiles.shape[0] == len(row_index)
            and quantiles.shape[1] == 3
            and quantiles.shape[2] > 1
            and np.isfinite(quantiles).all()
        )
    except (OSError, ValueError, KeyError):
        return False


def validate_frozen_configuration(config: dict[str, object]) -> None:
    if config.get("analysis_version") != ANALYSIS_VERSION:
        raise AssertionError("configuration version mismatch")
    if config.get("frozen_at_utc") != "2026-07-22T13:11:04Z":
        raise AssertionError("unexpected configuration freeze timestamp")
    if config.get("status") != "configuration-declared before execution":
        raise AssertionError("configuration is not declared frozen before execution")
    if tuple(config.get("seeds", ())) != EXPECTED_SEEDS:
        raise AssertionError("seed set differs from the frozen specification")
    if tuple(config.get("protocols", ())) != PHASES:
        raise AssertionError("protocol set or order differs from the frozen specification")
    training = config.get("training")
    if not isinstance(training, dict):
        raise AssertionError("missing training configuration")
    expected_training = {
        "training_history_mode": TRAINING_HISTORY_MODE,
        "epoch_samples": 500_000,
        "batch_size": 2_048,
        "inference_batch_size": 4_096,
        "max_epochs": 40,
        "patience": 4,
        "learning_rate": 0.001,
        "selection_metric": SELECTION_METRIC,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise AssertionError("training configuration differs from the frozen specification")
    external = config.get("external_inference")
    if not isinstance(external, dict):
        raise AssertionError("missing external-inference configuration")
    if (
        external.get("protocol") != "unseen_user only"
        or external.get("freeze_before_inference") is not True
        or external.get("adaptation_or_recalibration") is not False
    ):
        raise AssertionError("external-inference boundary differs from the specification")
    for key in (
        "array_dir",
        "strict_temporal_partition",
        "strict_temporal_audit",
        "output_root",
    ):
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise AssertionError(f"missing configured path: {key}")


def argument_list(parameters: dict[str, object]) -> list[str]:
    output: list[str] = []
    for name in BUDGET_FIELDS:
        output.extend([f"--{name.replace('_', '-')}", str(parameters[name])])
    return output


def freeze_record_postcondition_passed(
    path: Path,
    *,
    project_root: Path,
    configuration: Path,
    specification: Path,
    development_audit: Path,
    seed: int,
    artifacts: dict[str, Path],
) -> bool:
    """Re-hash every frozen input so a stale or mutated record cannot resume."""
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return False
        fixed = {
            "analysis_version": ANALYSIS_VERSION,
            "status": "frozen_before_external_inference",
            "seed": seed,
            "protocol": "unseen_user",
            "training_history_mode": TRAINING_HISTORY_MODE,
            "selection_metric": SELECTION_METRIC,
            "external_outcomes_used_for_selection": False,
            "external_adaptation_or_recalibration_allowed": False,
        }
        if any(record.get(key) != value for key, value in fixed.items()):
            return False
        hashed_inputs = {
            "configuration": configuration,
            "specification": specification,
            "development_audit": development_audit,
        }
        for name, input_path in hashed_inputs.items():
            entry = record.get(name)
            if not isinstance(entry, dict):
                return False
            if Path(str(entry.get("path"))).resolve() != input_path.resolve():
                return False
            if entry.get("sha256") != sha256_file(input_path):
                return False
        artifact_manifest = record.get("artifacts")
        if not isinstance(artifact_manifest, dict):
            return False
        for name, artifact_path in artifacts.items():
            entry = artifact_manifest.get(name)
            if not isinstance(entry, dict):
                return False
            if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
                return False
            if Path(str(entry.get("path"))).resolve() != artifact_path.resolve():
                return False
            if entry.get("sha256") != sha256_file(artifact_path):
                return False
        sources = {
            "training_script": project_root / "src" / "train_uncertainty_model.py",
            "external_inference_script": (
                project_root / "src" / "infer_frozen_external_uncertainty.py"
            ),
        }
        source_manifest = record.get("source_code")
        if not isinstance(source_manifest, dict):
            return False
        for name, source_path in sources.items():
            entry = source_manifest.get(name)
            if not isinstance(entry, dict):
                return False
            if Path(str(entry.get("path"))).resolve() != source_path.resolve():
                return False
            if entry.get("sha256") != sha256_file(source_path):
                return False
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return True


def build_freeze_record(
    *,
    project_root: Path,
    configuration: Path,
    specification: Path,
    development_audit: Path,
    seed: int,
    artifacts: dict[str, Path],
    training_command: str,
    training: dict[str, object],
) -> dict[str, object]:
    audit = json.loads(development_audit.read_text(encoding="utf-8"))
    resolved = {name: training[name] for name in BUDGET_FIELDS}
    resolved.update(
        {
            "seed": seed,
            "protocol": "unseen_user",
            "training_history_mode": TRAINING_HISTORY_MODE,
            "selection_metric": SELECTION_METRIC,
            "signal_input": SIGNAL_INPUT,
            "best_epoch": audit.get("best_epoch"),
            "best_validation_composite_mae": audit.get(
                "best_validation_composite_mae"
            ),
        }
    )
    sources = {
        "training_script": project_root / "src" / "train_uncertainty_model.py",
        "external_inference_script": (
            project_root / "src" / "infer_frozen_external_uncertainty.py"
        ),
    }
    return {
        "generated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "status": "frozen_before_external_inference",
        "seed": seed,
        "protocol": "unseen_user",
        "development_data_source": "Endomondo HR",
        "external_data_source": "GoldenCheetah OpenData",
        "training_history_mode": TRAINING_HISTORY_MODE,
        "selection_metric": SELECTION_METRIC,
        "external_outcomes_used_for_selection": False,
        "external_adaptation_or_recalibration_allowed": False,
        "resolved_training_parameters": resolved,
        "training_command": training_command,
        "configuration": {
            "path": str(configuration.resolve()),
            "sha256": sha256_file(configuration),
        },
        "specification": {
            "path": str(specification.resolve()),
            "sha256": sha256_file(specification),
        },
        "development_audit": {
            "path": str(development_audit.resolve()),
            "sha256": sha256_file(development_audit),
        },
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
        "source_code": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
    }


class QueueRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        configuration: Path,
        specification: Path,
        output_root: Path,
        dry_run: bool,
    ) -> None:
        self.project_root = project_root.resolve()
        self.configuration_path = configuration.resolve()
        self.specification_path = specification.resolve()
        self.config = json.loads(
            self.configuration_path.read_text(encoding="utf-8")
        )
        if not isinstance(self.config, dict):
            raise AssertionError("configuration must be a JSON object")
        validate_frozen_configuration(self.config)
        if not self.specification_path.is_file():
            raise FileNotFoundError(self.specification_path)
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_root / "queue_manifest.json"
        self.dry_run = dry_run
        configuration_hash = sha256_file(self.configuration_path)
        specification_hash = sha256_file(self.specification_path)
        if self.manifest_path.exists():
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(self.manifest, dict):
                raise AssertionError("queue manifest must be a JSON object")
            if self.manifest.get("configuration_sha256") != configuration_hash:
                raise AssertionError(
                    "configuration changed after queue initialization; use a new output root"
                )
            if self.manifest.get("specification_sha256") != specification_hash:
                raise AssertionError(
                    "specification changed after queue initialization; use a new output root"
                )
        else:
            self.manifest: dict[str, object] = {
                "analysis_version": ANALYSIS_VERSION,
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "configuration": str(self.configuration_path),
                "configuration_sha256": configuration_hash,
                "specification": str(self.specification_path),
                "specification_sha256": specification_hash,
                "python": sys.executable,
                "project_root": str(self.project_root),
                "tasks": {},
            }
            self.save_manifest()

    @property
    def training(self) -> dict[str, object]:
        training = self.config["training"]
        if not isinstance(training, dict):
            raise AssertionError("training configuration is not an object")
        return training

    def relative(self, configured_path: str) -> Path:
        return (self.project_root / configured_path).resolve()

    def save_manifest(self) -> None:
        self.manifest["updated_at_utc"] = utc_now()
        atomic_json(self.manifest_path, self.manifest)

    def task_update(self, key: str, **updates: object) -> None:
        tasks = self.manifest.setdefault("tasks", {})
        if not isinstance(tasks, dict):
            raise AssertionError("manifest tasks field is not an object")
        task = tasks.setdefault(key, {})
        if not isinstance(task, dict):
            raise AssertionError(f"manifest task is not an object: {key}")
        task.update(updates)
        self.save_manifest()

    def task_completed(self, key: str, **updates: object) -> None:
        tasks = self.manifest.setdefault("tasks", {})
        if not isinstance(tasks, dict):
            raise AssertionError("manifest tasks field is not an object")
        existing = tasks.setdefault(key, {})
        if not isinstance(existing, dict):
            raise AssertionError(f"manifest task is not an object: {key}")
        recovered = existing.get("recovered_native_shutdown_return_code")
        if recovered is not None and not recoverable_native_shutdown(int(recovered)):
            raise AssertionError(f"unrecognized recovered return code: {recovered}")
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
        postcondition_audit: Path,
        postcondition_artifacts: Iterable[Path],
        expected_audit_fields: dict[str, object],
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
            self.task_update(task_key, status="dry_run")
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
            and model_audit_postcondition_passed(
                postcondition_audit,
                postcondition_artifacts,
                expected_audit_fields,
                self._count_groups(step),
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

    @staticmethod
    def _count_groups(step: str) -> tuple[tuple[str, ...], ...]:
        if step == "frozen_external_inference":
            return (("external_rows",),)
        return (
            ("prediction_rows", "test_rows"),
            ("point_metric_rows",),
            ("interval_metric_rows", "uncertainty_metric_rows"),
        )

    def expected_model_fields(self, seed: int, protocol: str) -> dict[str, object]:
        fields = {
            "seed": seed,
            "protocol": protocol,
            "training_history_mode": TRAINING_HISTORY_MODE,
            "selection_metric": SELECTION_METRIC,
        }
        fields.update({name: self.training[name] for name in BUDGET_FIELDS})
        return fields

    def expected_resolved_config_fields(
        self, seed: int, protocol: str
    ) -> dict[str, object]:
        return self.expected_model_fields(seed, protocol)

    def unseen_seed(self, seed: int) -> None:
        key = f"seed_{seed}/unseen_user"
        root = self.output_root / f"seed_{seed}" / "unseen_user"
        # Keep formal artifact paths below the traditional Windows MAX_PATH
        # boundary.  The project title is intentionally descriptive and long;
        # using a compact leaf directory prevents the longest normalization
        # filename from exceeding 259 visible characters on Windows.
        model_dir = root / "m"
        audit = root / "audit.json"
        predictions = root / "test_predictions.npz"
        point_metrics = root / "point_metrics.csv"
        interval_metrics = root / "interval_metrics.csv"
        resolved_config = model_dir / "resolved_config.json"
        checkpoint = model_dir / "history_quantile_tcn_best_v0_11_0.pt"
        thresholds = model_dir / "conformal_thresholds_v0_11_0.json"
        input_normalization = model_dir / "normalization_unseen_user_train.json"
        history_normalization = (
            model_dir / "history_normalization_unseen_user_train.json"
        )
        model_artifacts = (
            predictions,
            point_metrics,
            interval_metrics,
            resolved_config,
            checkpoint,
            thresholds,
            input_normalization,
            history_normalization,
        )
        expected = self.expected_model_fields(seed, "unseen_user")
        expected.update(
            {
                "development_only": True,
                "external_inference_performed": False,
                "signal_input": SIGNAL_INPUT,
                "checkpoint": str(checkpoint),
                "thresholds_file": str(thresholds),
            }
        )
        config_expected = self.expected_resolved_config_fields(seed, "unseen_user")
        config_expected["signal_input"] = SIGNAL_INPUT
        training_ok = (
            model_audit_postcondition_passed(
                audit,
                model_artifacts,
                expected,
                self._count_groups("development_training"),
            )
            and resolved_config_postcondition_passed(
                resolved_config, config_expected
            )
            and zero_history_npz_postcondition_passed(predictions)
        )
        freeze_record = root / "freeze_record.json"
        if freeze_record.exists() and not training_ok:
            raise AssertionError(
                f"immutable freeze record exists without a valid matched development run: "
                f"{freeze_record}"
            )
        command = [
            sys.executable,
            str(self.project_root / "src" / "train_uncertainty_model.py"),
            "--array-dir",
            str(self.relative(str(self.config["array_dir"]))),
            "--output-dir",
            str(model_dir),
            "--predictions",
            str(predictions),
            "--point-metrics",
            str(point_metrics),
            "--uncertainty-metrics",
            str(interval_metrics),
            "--audit",
            str(audit),
            "--seed",
            str(seed),
            "--development-only",
            "--training-history-mode",
            TRAINING_HISTORY_MODE,
            "--signal-input",
            SIGNAL_INPUT,
        ] + argument_list(self.training)
        executed = False
        if not training_ok:
            executed = True
            self.run_command(
                task_key=key,
                step="development_training",
                command=command,
                log_path=root / "training.log",
                postcondition_audit=audit,
                postcondition_artifacts=model_artifacts,
                expected_audit_fields=expected,
            )
            if self.dry_run:
                return
            training_ok = (
                model_audit_postcondition_passed(
                    audit,
                    model_artifacts,
                    expected,
                    self._count_groups("development_training"),
                )
                and resolved_config_postcondition_passed(
                    resolved_config, config_expected
                )
                and zero_history_npz_postcondition_passed(predictions)
            )
            if not training_ok:
                self.task_update(
                    key,
                    status="failed",
                    failed_step="development_training_postcondition",
                    finished_at_utc=utc_now(),
                )
                raise AssertionError(f"{key}: development postcondition failed")

        frozen_artifacts = {
            "checkpoint": checkpoint,
            "thresholds": thresholds,
            "input_normalization": input_normalization,
            "history_normalization": history_normalization,
            "resolved_config": resolved_config,
        }
        freeze_ok = freeze_record_postcondition_passed(
            freeze_record,
            project_root=self.project_root,
            configuration=self.configuration_path,
            specification=self.specification_path,
            development_audit=audit,
            seed=seed,
            artifacts=frozen_artifacts,
        )
        if freeze_record.exists() and not freeze_ok:
            raise AssertionError(f"immutable freeze record is invalid: {freeze_record}")
        if not freeze_ok:
            executed = True
            self.task_update(
                key,
                status="running",
                current_step="freeze_record",
                step_started_at_utc=utc_now(),
                failed_step=None,
                return_code=None,
            )
            if self.dry_run:
                self.task_update(key, status="dry_run")
                return
            payload = build_freeze_record(
                project_root=self.project_root,
                configuration=self.configuration_path,
                specification=self.specification_path,
                development_audit=audit,
                seed=seed,
                artifacts=frozen_artifacts,
                training_command=subprocess.list2cmdline(command),
                training=self.training,
            )
            try:
                freeze_record.parent.mkdir(parents=True, exist_ok=True)
                with freeze_record.open("x", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                self.task_update(
                    key,
                    status="failed",
                    failed_step="freeze_record",
                    finished_at_utc=utc_now(),
                )
                raise
            freeze_ok = freeze_record_postcondition_passed(
                freeze_record,
                project_root=self.project_root,
                configuration=self.configuration_path,
                specification=self.specification_path,
                development_audit=audit,
                seed=seed,
                artifacts=frozen_artifacts,
            )
            if not freeze_ok:
                raise AssertionError(f"{key}: freeze-record postcondition failed")

        external_audit = root / "external_audit.json"
        external_predictions = root / "external_predictions.npz"
        external_point = root / "external_point_metrics.csv"
        external_interval = root / "external_interval_metrics.csv"
        external_artifacts = (
            external_predictions,
            external_point,
            external_interval,
        )
        external_expected = {
            "seed": seed,
            "protocol": (
                "frozen GoldenCheetah external inference after development freeze"
            ),
            "freeze_record_sha256": sha256_file(freeze_record),
            "checkpoint_sha256": sha256_file(checkpoint),
            "thresholds_sha256": sha256_file(thresholds),
            "external_adaptation_or_recalibration": False,
            "signal_input": SIGNAL_INPUT,
            "inference_batch_size": self.training["inference_batch_size"],
        }
        external_ok = (
            model_audit_postcondition_passed(
                external_audit,
                external_artifacts,
                external_expected,
                self._count_groups("frozen_external_inference"),
            )
            and zero_history_npz_postcondition_passed(external_predictions)
        )
        if not external_ok:
            executed = True
            external_command = [
                sys.executable,
                str(
                    self.project_root
                    / "src"
                    / "infer_frozen_external_uncertainty.py"
                ),
                "--array-dir",
                str(self.relative(str(self.config["array_dir"]))),
                "--checkpoint",
                str(checkpoint),
                "--thresholds",
                str(thresholds),
                "--input-normalization",
                str(input_normalization),
                "--history-normalization",
                str(history_normalization),
                "--freeze-record",
                str(freeze_record),
                "--predictions",
                str(external_predictions),
                "--point-metrics",
                str(external_point),
                "--interval-metrics",
                str(external_interval),
                "--audit",
                str(external_audit),
                "--inference-batch-size",
                str(self.training["inference_batch_size"]),
                "--signal-input",
                SIGNAL_INPUT,
            ]
            self.run_command(
                task_key=key,
                step="frozen_external_inference",
                command=external_command,
                log_path=root / "external_inference.log",
                postcondition_audit=external_audit,
                postcondition_artifacts=external_artifacts,
                expected_audit_fields=external_expected,
            )
            if self.dry_run:
                return
            external_ok = (
                model_audit_postcondition_passed(
                    external_audit,
                    external_artifacts,
                    external_expected,
                    self._count_groups("frozen_external_inference"),
                )
                and zero_history_npz_postcondition_passed(external_predictions)
            )
            if not external_ok:
                self.task_update(
                    key,
                    status="failed",
                    failed_step="frozen_external_inference_postcondition",
                    finished_at_utc=utc_now(),
                )
                raise AssertionError(f"{key}: external postcondition failed")
        self.task_completed(
            key,
            resumed_and_verified=not executed,
            development_audit_sha256=sha256_file(audit),
            resolved_config_sha256=sha256_file(resolved_config),
            freeze_record_sha256=sha256_file(freeze_record),
            external_audit_sha256=sha256_file(external_audit),
        )

    def temporal_seed(self, seed: int) -> None:
        key = f"seed_{seed}/strict_temporal"
        root = self.output_root / f"seed_{seed}" / "strict_temporal"
        model_dir = root / "m"
        audit = root / "audit.json"
        predictions = root / "predictions.npz"
        point_metrics = root / "point_metrics.csv"
        interval_metrics = root / "interval_metrics.csv"
        resolved_config = model_dir / "resolved_config.json"
        artifacts = (
            predictions,
            point_metrics,
            interval_metrics,
            resolved_config,
            model_dir / "temporal_history_quantile_tcn_best.pt",
            model_dir / "input_normalization.json",
            model_dir / "history_normalization.json",
            model_dir / "conformal_thresholds.json",
        )
        expected = self.expected_model_fields(seed, "strict_temporal")
        config_expected = self.expected_resolved_config_fields(
            seed, "strict_temporal"
        )
        complete = (
            model_audit_postcondition_passed(
                audit,
                artifacts,
                expected,
                self._count_groups("training_and_evaluation"),
            )
            and resolved_config_postcondition_passed(
                resolved_config, config_expected
            )
            and zero_history_npz_postcondition_passed(predictions)
        )
        if complete:
            self.task_completed(
                key,
                resumed_and_verified=True,
                audit_sha256=sha256_file(audit),
                resolved_config_sha256=sha256_file(resolved_config),
            )
            return
        command = [
            sys.executable,
            str(
                self.project_root / "src" / "train_temporal_uncertainty_model.py"
            ),
            "--array-dir",
            str(self.relative(str(self.config["array_dir"]))),
            "--temporal-partition",
            str(self.relative(str(self.config["strict_temporal_partition"]))),
            "--temporal-audit",
            str(self.relative(str(self.config["strict_temporal_audit"]))),
            "--output-dir",
            str(model_dir),
            "--predictions",
            str(predictions),
            "--point-metrics",
            str(point_metrics),
            "--interval-metrics",
            str(interval_metrics),
            "--audit",
            str(audit),
            "--seed",
            str(seed),
            "--training-history-mode",
            TRAINING_HISTORY_MODE,
        ] + argument_list(self.training)
        self.run_command(
            task_key=key,
            step="training_and_evaluation",
            command=command,
            log_path=root / "training.log",
            postcondition_audit=audit,
            postcondition_artifacts=artifacts,
            expected_audit_fields=expected,
        )
        if self.dry_run:
            return
        complete = (
            model_audit_postcondition_passed(
                audit,
                artifacts,
                expected,
                self._count_groups("training_and_evaluation"),
            )
            and resolved_config_postcondition_passed(
                resolved_config, config_expected
            )
            and zero_history_npz_postcondition_passed(predictions)
        )
        if not complete:
            self.task_update(
                key,
                status="failed",
                failed_step="training_and_evaluation_postcondition",
                finished_at_utc=utc_now(),
            )
            raise AssertionError(f"{key}: strict-temporal postcondition failed")
        self.task_completed(
            key,
            resumed_and_verified=False,
            audit_sha256=sha256_file(audit),
            resolved_config_sha256=sha256_file(resolved_config),
        )

    def run(self, phases: list[str], selected_seeds: set[int] | None) -> None:
        unknown = set(selected_seeds or ()) - set(EXPECTED_SEEDS)
        if unknown:
            raise ValueError(f"seed(s) outside frozen specification: {sorted(unknown)}")
        self.manifest["last_requested_phases"] = list(phases)
        self.manifest["last_requested_seed_subset"] = (
            sorted(selected_seeds) if selected_seeds is not None else None
        )
        self.manifest["last_run_started_at_utc"] = utc_now()
        self.manifest["last_run_finished_at_utc"] = None
        self.save_manifest()
        seeds = [
            seed
            for seed in EXPECTED_SEEDS
            if selected_seeds is None or seed in selected_seeds
        ]
        for phase in phases:
            if phase == "unseen_user":
                for seed in seeds:
                    self.unseen_seed(seed)
            elif phase == "strict_temporal":
                for seed in seeds:
                    self.temporal_seed(seed)
            else:
                raise ValueError(f"unknown phase: {phase}")
        self.manifest["last_run_finished_at_utc"] = utc_now()
        self.save_manifest()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the frozen independent always-zero-history ablation queue."
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        default=(
            project_root / "configs" / "independent_zero_history_v0_23_0.json"
        ),
    )
    parser.add_argument(
        "--specification",
        type=Path,
        default=(
            project_root
            / "protocol"
            / "INDEPENDENT_ZERO_HISTORY_ABLATION_SPECIFICATION.md"
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--phase",
        action="append",
        choices=PHASES,
        help="Repeat to select phases; default runs both registered phases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Optional frozen seed subset; repeat for more than one seed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configuration = args.configuration.resolve()
    config = json.loads(configuration.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise AssertionError("configuration must be a JSON object")
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (project_root / str(config["output_root"])).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "queue.lock"
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
            specification=args.specification,
            output_root=output_root,
            dry_run=args.dry_run,
        )
        runner.run(
            args.phase or list(PHASES), set(args.seed) if args.seed else None
        )
    finally:
        if lock_path.exists():
            lock_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
