from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bootstrap_model_comparisons import (
    BOOTSTRAP_REPLICATES,
    SEED,
    holm_adjust,
    paired_bootstrap,
    user_mae,
)
from train_xgboost_baseline import EXTERNAL_FROZEN, HORIZONS, PARTITION_TEST


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    users = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(args.array_dir / "session_index.npy", mmap_mode="r")
    dataset = np.load(args.array_dir / "dataset_code.npy", mmap_mode="r")
    unseen = np.load(args.array_dir / "unseen_user_partition.npy", mmap_mode="r")
    external = np.load(
        args.array_dir / "primary_external_partition.npy", mmap_mode="r"
    )
    with np.load(args.multimodal_predictions) as multimodal, np.load(
        args.hr_only_predictions
    ) as hr_only:
        if not np.array_equal(multimodal["row_index"], hr_only["row_index"]):
            raise AssertionError("signal-ablation predictions are not row aligned")
        row_index = multimodal["row_index"]
        full_history = multimodal["history_quantiles"][:, :, 3]
        full_zero = multimodal["zero_history_quantiles"][:, :, 3]
        hr_history = hr_only["history_quantiles"][:, :, 3]
        hr_zero = hr_only["zero_history_quantiles"][:, :, 3]
    internal = (dataset[row_index] == 0) & (unseen[row_index] == PARTITION_TEST)
    frozen_external = (dataset[row_index] == 1) & (
        external[row_index] == EXTERNAL_FROZEN
    )
    comparisons = (
        (
            "unseen_history_multimodal_vs_hr_only",
            internal,
            "multimodal_history",
            full_history,
            "hr_only_history",
            hr_history,
        ),
        (
            "unseen_zero_multimodal_vs_hr_only",
            internal,
            "multimodal_zero_history",
            full_zero,
            "hr_only_zero_history",
            hr_zero,
        ),
        (
            "external_zero_multimodal_vs_hr_only",
            frozen_external,
            "multimodal_zero_history",
            full_zero,
            "hr_only_zero_history",
            hr_zero,
        ),
    )
    generator = np.random.default_rng(SEED + 14)
    rows: list[dict[str, object]] = []
    for family, subset, model_a, prediction_a, model_b, prediction_b in comparisons:
        selected_rows = row_index[subset]
        for position, horizon in enumerate(HORIZONS):
            a = user_mae(
                prediction_a[subset, position],
                np.asarray(targets[selected_rows, position]),
                np.asarray(users[selected_rows]),
                np.asarray(sessions[selected_rows]),
            )
            b = user_mae(
                prediction_b[subset, position],
                np.asarray(targets[selected_rows, position]),
                np.asarray(users[selected_rows]),
                np.asarray(sessions[selected_rows]),
            )
            paired = a.to_frame("a").join(b.to_frame("b"), how="inner")
            difference = (paired["a"] - paired["b"]).to_numpy(dtype=np.float64)
            rows.append(
                {
                    "comparison_family": family,
                    "horizon_seconds": horizon,
                    "model_a": model_a,
                    "model_b": model_b,
                    "delta_definition": "MAE(model_a) - MAE(model_b); negative favors multimodal",
                    **paired_bootstrap(difference, generator),
                    "users": len(difference),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "inference": "paired user bootstrap CI; paired two-sided Wilcoxon; Holm within three horizons",
                    "seed": SEED + 14,
                }
            )
    holm_adjust(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload: dict[str, object] = {
        "seed": SEED + 14,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "rows": len(rows),
        "families": sorted({str(row["comparison_family"]) for row in rows}),
        "all_assertions_pass": len(rows) == 9
        and all(int(row["users"]) > 0 for row in rows),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--multimodal-predictions", type=Path, required=True)
    result.add_argument("--hr-only-predictions", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parser().parse_args())))
