from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_strict_temporal_learned_baseline import (
    FORMAL_EPOCH_SAMPLES,
    FORMAL_MAX_EPOCHS,
    FORMAL_PATIENCE,
    train,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Compatibility entry point for GRU/point-TCN training under the "
            "strict within-user temporal protocol."
        )
    )
    result.add_argument("--model", choices=("gru", "tcn"), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument(
        "--run-purpose", choices=("formal", "smoke"), default="formal"
    )
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--temporal-partition", type=Path, required=True)
    result.add_argument("--temporal-audit", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True)
    result.add_argument("--metrics", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=2048)
    result.add_argument("--inference-batch-size", type=int, default=8192)
    result.add_argument(
        "--epoch-samples", type=int, default=FORMAL_EPOCH_SAMPLES
    )
    result.add_argument("--max-epochs", type=int, default=FORMAL_MAX_EPOCHS)
    result.add_argument("--patience", type=int, default=FORMAL_PATIENCE)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--smooth-l1-beta", type=float, default=5.0)
    result.add_argument("--gradient-clip-norm", type=float, default=1.0)
    result.add_argument("--allow-overwrite", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(train(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
