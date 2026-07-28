from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ANALYSIS_VERSION = "0.21.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(project_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "working_tree_dirty": None}


def build_record(args: argparse.Namespace) -> dict[str, object]:
    audit = json.loads(args.development_audit.read_text(encoding="utf-8"))
    if audit.get("all_assertions_pass") is not True:
        raise AssertionError("development audit did not pass")
    if audit.get("development_only") is not True:
        raise AssertionError("development run was not isolated from external inference")
    if audit.get("external_inference_performed") is not False:
        raise AssertionError("development audit indicates external inference")
    if int(audit.get("seed")) != args.seed:
        raise AssertionError("seed mismatch between audit and freeze request")
    artifacts = {
        "checkpoint": args.checkpoint,
        "thresholds": args.thresholds,
        "input_normalization": args.input_normalization,
        "history_normalization": args.history_normalization,
    }
    for name, path in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    parameters = {
        name: audit[name]
        for name in (
            "seed",
            "epoch_samples",
            "batch_size",
            "inference_batch_size",
            "max_epochs",
            "patience",
            "learning_rate",
            "history_dropout",
            "signal_input",
            "best_epoch",
            "best_validation_composite_mae",
        )
    }
    project_root = Path(__file__).resolve().parents[1]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": ANALYSIS_VERSION,
        "status": "frozen_before_external_inference",
        "seed": args.seed,
        "development_data_source": "Endomondo HR",
        "external_data_source": "GoldenCheetah OpenData",
        "selection_metric": (
            "mean history-informed and zero-history validation hierarchical MAE"
        ),
        "external_outcomes_used_for_selection": False,
        "external_adaptation_or_recalibration_allowed": False,
        "resolved_training_parameters": parameters,
        "training_command": args.training_command,
        "configuration": {
            "path": str(args.configuration.resolve()),
            "sha256": sha256_file(args.configuration),
        },
        "development_audit": {
            "path": str(args.development_audit.resolve()),
            "sha256": sha256_file(args.development_audit),
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "source_code": {
            "training_script": {
                "path": str((project_root / "src" / "train_uncertainty_model.py").resolve()),
                "sha256": sha256_file(
                    project_root / "src" / "train_uncertainty_model.py"
                ),
            },
            "external_inference_script": {
                "path": str(
                    (project_root / "src" / "infer_frozen_external_uncertainty.py").resolve()
                ),
                "sha256": sha256_file(
                    project_root / "src" / "infer_frozen_external_uncertainty.py"
                ),
            },
        },
        "git": git_state(project_root),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write an immutable checkpoint freeze record before external inference."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--input-normalization", type=Path, required=True)
    parser.add_argument("--history-normalization", type=Path, required=True)
    parser.add_argument("--training-command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"freeze record already exists: {args.output}")
    record = build_record(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
