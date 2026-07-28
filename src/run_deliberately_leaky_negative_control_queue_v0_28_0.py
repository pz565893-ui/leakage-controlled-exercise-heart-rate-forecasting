from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from build_deliberately_leaky_temporal_partition_v0_28_0 import (
    ANALYSIS_VERSION,
    index_sha256,
    require,
)
from run_q1_multiseed_queue import (
    atomic_json,
    recoverable_native_shutdown,
    sha256_file,
    utc_now,
)
from train_deliberately_leaky_temporal_negative_control_v0_28_0 import (
    PROTOCOL,
    SELECTION_METRIC,
    TRAINING_HISTORY_MODE,
    validate_configuration,
)


PHASES = ("partition", "training", "aggregation")
EXPECTED_SEEDS = (20260722, 20260723, 20260724)


def json_audit_passed(path: Path, expected: dict[str, object] | None = None) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("all_assertions_pass") is not True:
            return False
        if payload.get("valid_for_generalization") is not False:
            return False
        if expected and any(payload.get(key) != value for key, value in expected.items()):
            return False
        return True
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def prediction_postcondition(path: Path, expected_hash: str) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"row_index", "zero_history_quantiles"}:
                return False
            row_index = np.asarray(archive["row_index"], dtype=np.int64)
            prediction = archive["zero_history_quantiles"]
            return (
                index_sha256(row_index) == expected_hash
                and prediction.shape == (len(row_index), 3, 7)
                and np.isfinite(prediction).all()
            )
    except (OSError, ValueError, KeyError, AssertionError):
        return False


def artifact_set_nonempty(paths: list[Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


class QueueRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        configuration: Path,
        output_root: Path,
        acknowledge_invalid_generalization: bool,
        dry_run: bool,
    ) -> None:
        require(
            acknowledge_invalid_generalization,
            "queue requires --acknowledge-invalid-generalization",
        )
        self.project_root = project_root.resolve()
        self.configuration = configuration.resolve()
        self.config = json.loads(self.configuration.read_text(encoding="utf-8"))
        validate_configuration(self.config)
        require(tuple(self.config["seeds"]) == EXPECTED_SEEDS, "seed lock")
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.manifest_path = self.output_root / "queue_manifest.json"
        configuration_hash = sha256_file(self.configuration)
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            require(
                self.manifest.get("configuration_sha256") == configuration_hash,
                "configuration changed after queue initialization",
            )
        else:
            self.manifest: dict[str, object] = {
                "analysis_version": ANALYSIS_VERSION,
                "valid_for_generalization": False,
                "leaderboard_eligible": False,
                "acknowledge_invalid_generalization": True,
                "configuration": str(self.configuration),
                "configuration_sha256": configuration_hash,
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "python": sys.executable,
                "tasks": {},
            }
            self.save_manifest()

    def resolve(self, configured: str) -> Path:
        return (self.project_root / configured).resolve()

    def save_manifest(self) -> None:
        self.manifest["updated_at_utc"] = utc_now()
        atomic_json(self.manifest_path, self.manifest)

    def task_update(self, key: str, **updates: object) -> None:
        tasks = self.manifest.setdefault("tasks", {})
        require(isinstance(tasks, dict), "malformed queue tasks")
        task = tasks.setdefault(key, {})
        require(isinstance(task, dict), "malformed queue task")
        task.update(updates)
        self.save_manifest()

    def run_command(
        self,
        *,
        task_key: str,
        command: list[str],
        log_path: Path,
        postcondition,
    ) -> None:
        command_text = subprocess.list2cmdline(command)
        self.task_update(
            task_key,
            status="dry_run" if self.dry_run else "running",
            command=command_text,
            log=str(log_path),
            started_at_utc=utc_now(),
            finished_at_utc=None,
            return_code=None,
        )
        if self.dry_run:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now()}] {command_text}\n")
            handle.flush()
            result = subprocess.run(
                command,
                cwd=self.project_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            handle.write(f"[{utc_now()}] return_code={result.returncode}\n")
        recovered = recoverable_native_shutdown(result.returncode) and postcondition()
        if result.returncode != 0 and not recovered:
            self.task_update(
                task_key,
                status="failed",
                return_code=result.returncode,
                finished_at_utc=utc_now(),
            )
            raise RuntimeError(f"{task_key} failed; see {log_path}")
        require(postcondition(), f"{task_key}: postcondition failed")
        self.task_update(
            task_key,
            status="completed",
            return_code=0 if result.returncode == 0 else result.returncode,
            recovered_native_shutdown=bool(recovered),
            finished_at_utc=utc_now(),
        )

    def partition_paths(self) -> tuple[Path, Path]:
        return (
            self.resolve(str(self.config["leaky_partition"])),
            self.resolve(str(self.config["leaky_partition_audit"])),
        )

    def partition_postcondition(self) -> bool:
        partition, audit = self.partition_paths()
        if not json_audit_passed(audit, {"analysis_version": ANALYSIS_VERSION}):
            return False
        try:
            payload = json.loads(audit.read_text(encoding="utf-8"))
            return (
                partition.is_file()
                and payload.get("output_sha256") == sha256_file(partition)
                and payload.get("configuration_sha256")
                == sha256_file(self.configuration)
                and payload.get("expected_count_mismatches") == {}
                and payload.get("fixed_test", {}).get(
                    "row_index_sha256_int64_little_endian"
                )
                == self.config["fixed_test"][
                    "row_index_sha256_int64_little_endian"
                ]
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            return False

    def run_partition(self) -> None:
        key = "partition"
        if self.partition_postcondition():
            self.task_update(key, status="completed", resumed_and_verified=True)
            return
        partition, audit = self.partition_paths()
        command = [
            sys.executable,
            str(
                self.project_root
                / "src"
                / "build_deliberately_leaky_temporal_partition_v0_28_0.py"
            ),
            "--configuration",
            str(self.configuration),
            "--array-dir",
            str(self.resolve(str(self.config["array_dir"]))),
            "--strict-temporal-partition",
            str(self.resolve(str(self.config["strict_temporal_partition"]))),
            "--output",
            str(partition),
            "--audit",
            str(audit),
        ]
        self.run_command(
            task_key=key,
            command=command,
            log_path=self.output_root / "partition.log",
            postcondition=self.partition_postcondition,
        )

    def seed_paths(self, seed: int) -> dict[str, Path]:
        root = self.output_root / f"seed_{seed}"
        model = root / "m"
        return {
            "root": root,
            "model": model,
            "audit": root / "audit.json",
            "predictions": root / "predictions.npz",
            "point": root / "point_metrics.csv",
            "interval": root / "interval_metrics.csv",
            "checkpoint": model / "best.pt",
            "input_norm": model / "input_normalization.json",
            "history_norm": model / "history_normalization.json",
            "thresholds": model / "conformal_thresholds.json",
            "resolved": model / "resolved_config.json",
            "log": root / "training.log",
        }

    def seed_postcondition(self, seed: int) -> bool:
        paths = self.seed_paths(seed)
        fixed_hash = str(
            self.config["fixed_test"]["row_index_sha256_int64_little_endian"]
        )
        counts = self.config["expected_partition_counts"]
        artifacts = [
            paths[name]
            for name in (
                "predictions",
                "point",
                "interval",
                "checkpoint",
                "input_norm",
                "history_norm",
                "thresholds",
                "resolved",
            )
        ]
        return (
            json_audit_passed(
                paths["audit"],
                {
                    "analysis_version": ANALYSIS_VERSION,
                    "protocol": PROTOCOL,
                    "seed": seed,
                    "formal_budget_locked": True,
                    "training_history_mode": TRAINING_HISTORY_MODE,
                    "selection_metric": SELECTION_METRIC,
                    "clean_checkpoint_reused_or_warm_started": False,
                    "test_row_index_sha256_int64_little_endian": fixed_hash,
                    "clean_test_row_index_exact_order_match": True,
                    "training_rows": int(counts["leaky_train_rows"]),
                    "validation_rows": int(counts["leaky_validation_rows"]),
                    "calibration_rows": int(counts["leaky_calibration_rows"]),
                    "test_rows": int(counts["fixed_test_rows"]),
                    "cqr_coverage_guarantee_valid": False,
                },
            )
            and artifact_set_nonempty(artifacts)
            and prediction_postcondition(paths["predictions"], fixed_hash)
        )

    def run_seed(self, seed: int) -> None:
        key = f"seed_{seed}"
        if self.seed_postcondition(seed):
            self.task_update(key, status="completed", resumed_and_verified=True)
            return
        require(
            self.partition_postcondition() or self.dry_run,
            "leaky partition must pass before training",
        )
        paths = self.seed_paths(seed)
        clean_root = self.resolve(str(self.config["clean_control"]["root"]))
        clean_pattern = str(self.config["clean_control"]["prediction_pattern"])
        clean_predictions = clean_root / clean_pattern.format(seed=seed)
        training = self.config["training"]
        partition, partition_audit = self.partition_paths()
        command = [
            sys.executable,
            str(
                self.project_root
                / "src"
                / "train_deliberately_leaky_temporal_negative_control_v0_28_0.py"
            ),
            "--acknowledge-invalid-generalization",
            "--configuration",
            str(self.configuration),
            "--array-dir",
            str(self.resolve(str(self.config["array_dir"]))),
            "--leaky-partition",
            str(partition),
            "--leaky-partition-audit",
            str(partition_audit),
            "--clean-control-predictions",
            str(clean_predictions),
            "--output-dir",
            str(paths["model"]),
            "--predictions",
            str(paths["predictions"]),
            "--point-metrics",
            str(paths["point"]),
            "--interval-metrics",
            str(paths["interval"]),
            "--audit",
            str(paths["audit"]),
            "--seed",
            str(seed),
        ]
        for name in (
            "epoch_samples",
            "batch_size",
            "inference_batch_size",
            "max_epochs",
            "patience",
            "learning_rate",
            "history_dropout",
        ):
            command.extend([f"--{name.replace('_', '-')}", str(training[name])])
        self.run_command(
            task_key=key,
            command=command,
            log_path=paths["log"],
            postcondition=lambda: self.seed_postcondition(seed),
        )

    def aggregation_paths(self) -> tuple[Path, Path]:
        root = self.output_root / "aggregation"
        return root, root / "audit.json"

    def aggregation_postcondition(self) -> bool:
        root, audit = self.aggregation_paths()
        required = [
            root / "paired_metrics_per_seed_v0_28_0.csv",
            root / "paired_metrics_seed_summary_v0_28_0.csv",
            root / "paired_user_seed_mean_v0_28_0.csv",
            root / "paired_user_bootstrap_v0_28_0.csv",
            root / "interval_diagnostics_per_seed_v0_28_0.csv",
        ]
        return json_audit_passed(
            audit,
            {
                "analysis_version": ANALYSIS_VERSION,
                "matched_seed_count": 3,
                "clean_leaky_exact_row_order_match_all_seeds": True,
                "coverage_guarantee_valid": False,
            },
        ) and artifact_set_nonempty(required)

    def run_aggregation(self) -> None:
        key = "aggregation"
        if self.aggregation_postcondition():
            self.task_update(key, status="completed", resumed_and_verified=True)
            return
        require(
            self.dry_run or all(self.seed_postcondition(seed) for seed in EXPECTED_SEEDS),
            "all three seed jobs must pass before aggregation",
        )
        root, _ = self.aggregation_paths()
        command = [
            sys.executable,
            str(
                self.project_root
                / "src"
                / "aggregate_deliberately_leaky_negative_control_v0_28_0.py"
            ),
            "--acknowledge-invalid-generalization",
            "--configuration",
            str(self.configuration),
            "--array-dir",
            str(self.resolve(str(self.config["array_dir"]))),
            "--leaky-root",
            str(self.output_root),
            "--clean-root",
            str(self.resolve(str(self.config["clean_control"]["root"]))),
            "--partition-audit",
            str(self.partition_paths()[1]),
            "--output-dir",
            str(root),
        ]
        self.run_command(
            task_key=key,
            command=command,
            log_path=root / "aggregation.log",
            postcondition=self.aggregation_postcondition,
        )

    def run(self, phases: list[str], selected_seeds: set[int] | None) -> None:
        unknown = set(selected_seeds or ()) - set(EXPECTED_SEEDS)
        require(not unknown, f"seed(s) outside lock: {sorted(unknown)}")
        seeds = [
            seed
            for seed in EXPECTED_SEEDS
            if selected_seeds is None or seed in selected_seeds
        ]
        self.manifest["last_requested_phases"] = phases
        self.manifest["last_requested_seeds"] = seeds
        self.manifest["last_run_started_at_utc"] = utc_now()
        self.manifest["last_run_finished_at_utc"] = None
        self.save_manifest()
        for phase in phases:
            if phase == "partition":
                self.run_partition()
            elif phase == "training":
                for seed in seeds:
                    self.run_seed(seed)
            elif phase == "aggregation":
                require(
                    selected_seeds is None or set(seeds) == set(EXPECTED_SEEDS),
                    "formal aggregation requires all three seeds",
                )
                self.run_aggregation()
            else:
                raise ValueError(f"unknown phase: {phase}")
        self.manifest["last_run_finished_at_utc"] = utc_now()
        self.save_manifest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the guarded v0.28 deliberately leaky three-seed queue."
    )
    parser.add_argument(
        "--acknowledge-invalid-generalization", action="store_true", required=True
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        default=root / "configs" / "leaky_negative_control_v0_28_0.json",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--phase", action="append", choices=PHASES)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.configuration.read_text(encoding="utf-8"))
    output_root = args.output_root or (root / str(config["output_root"]))
    runner = QueueRunner(
        project_root=root,
        configuration=args.configuration,
        output_root=output_root,
        acknowledge_invalid_generalization=args.acknowledge_invalid_generalization,
        dry_run=args.dry_run,
    )
    runner.run(args.phase or list(PHASES), set(args.seed) if args.seed else None)
    print(json.dumps(runner.manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
