from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from train_strict_temporal_learned_baseline import (
    FORMAL_EPOCH_SAMPLES,
    FORMAL_MAX_EPOCHS,
    FORMAL_PATIENCE,
    validate_budget,
)


def build_command(
    *,
    trainer: Path,
    model: str,
    seed: int,
    array_dir: Path,
    temporal_partition: Path,
    temporal_audit: Path,
    output_root: Path,
    batch_size: int,
    inference_batch_size: int,
    epoch_samples: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
) -> list[str]:
    validate_budget("formal", epoch_samples, max_epochs, patience)
    run_dir = output_root / f"seed_{seed}" / f"temporal_{model}"
    return [
        sys.executable,
        str(trainer),
        "--model",
        model,
        "--seed",
        str(seed),
        "--run-purpose",
        "formal",
        "--array-dir",
        str(array_dir),
        "--temporal-partition",
        str(temporal_partition),
        "--temporal-audit",
        str(temporal_audit),
        "--output-dir",
        str(run_dir),
        "--predictions",
        str(run_dir / "strict_temporal_test_predictions.npz"),
        "--metrics",
        str(run_dir / "strict_temporal_test_metrics.csv"),
        "--audit",
        str(run_dir / "audit.json"),
        "--batch-size",
        str(batch_size),
        "--inference-batch-size",
        str(inference_batch_size),
        "--epoch-samples",
        str(epoch_samples),
        "--max-epochs",
        str(max_epochs),
        "--patience",
        str(patience),
        "--learning-rate",
        str(learning_rate),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or sequentially execute formal strict-temporal GRU/TCN runs. "
            "Dry-run is the default; --execute is required to train."
        )
    )
    parser.add_argument(
        "--models", nargs="+", choices=("gru", "tcn"), default=("gru", "tcn")
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    parser.add_argument(
        "--temporal-partition",
        type=Path,
        default=Path(
            "outputs/features/model_arrays_v0_6_0/temporal_partition_strict.npy"
        ),
    )
    parser.add_argument(
        "--temporal-audit",
        type=Path,
        default=Path("outputs/audit/strict_temporal_partition_v0_13_0.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/strict_temporal_learned_baselines_v0_22_0"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument(
        "--epoch-samples", type=int, default=FORMAL_EPOCH_SAMPLES
    )
    parser.add_argument("--max-epochs", type=int, default=FORMAL_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=FORMAL_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    validate_budget(
        "formal", args.epoch_samples, args.max_epochs, args.patience
    )
    trainer = Path(__file__).with_name(
        "train_temporal_neural_baselines.py"
    )
    plan: list[dict[str, object]] = []
    for model in args.models:
        for seed in args.seeds:
            command = build_command(
                trainer=trainer,
                model=model,
                seed=seed,
                array_dir=args.array_dir,
                temporal_partition=args.temporal_partition,
                temporal_audit=args.temporal_audit,
                output_root=args.output_root,
                batch_size=args.batch_size,
                inference_batch_size=args.inference_batch_size,
                epoch_samples=args.epoch_samples,
                max_epochs=args.max_epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
            )
            run_dir = (
                args.output_root / f"seed_{seed}" / f"temporal_{model}"
            )
            plan.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "command": command,
                }
            )
    print(
        json.dumps(
            {
                "execute": args.execute,
                "sequential_gpu_runs": True,
                "formal_budget": {
                    "epoch_samples": args.epoch_samples,
                    "max_epochs": args.max_epochs,
                    "patience": args.patience,
                },
                "runs": plan,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.execute:
        for item in plan:
            subprocess.run(item["command"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
