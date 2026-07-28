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
from train_xgboost_baseline import HORIZONS


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    users = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(args.array_dir / "session_index.npy", mmap_mode="r")
    with np.load(args.main_predictions) as main, np.load(
        args.baseline_predictions
    ) as baseline:
        if not np.array_equal(main["row_index"], baseline["row_index"]):
            raise AssertionError("temporal model and baseline rows are not aligned")
        row_index = main["row_index"]
        history = main["history_quantiles"][:, :, 3]
        zero = main["zero_history_quantiles"][:, :, 3]
        persistence = baseline["persistence"]
        ewma = baseline["ewma_alpha_0_1"]
    comparisons = (
        ("temporal_history_vs_zero", "history_informed", history, "zero_history", zero),
        (
            "temporal_history_vs_persistence",
            "history_informed",
            history,
            "persistence",
            persistence,
        ),
        (
            "temporal_history_vs_ewma",
            "history_informed",
            history,
            "ewma_alpha_0_1",
            ewma,
        ),
    )
    generator = np.random.default_rng(SEED + 13)
    rows: list[dict[str, object]] = []
    for family, model_a, prediction_a, model_b, prediction_b in comparisons:
        for position, horizon in enumerate(HORIZONS):
            a = user_mae(
                prediction_a[:, position],
                np.asarray(targets[row_index, position]),
                np.asarray(users[row_index]),
                np.asarray(sessions[row_index]),
            )
            b = user_mae(
                prediction_b[:, position],
                np.asarray(targets[row_index, position]),
                np.asarray(users[row_index]),
                np.asarray(sessions[row_index]),
            )
            paired = a.to_frame("a").join(b.to_frame("b"), how="inner")
            difference = (paired["a"] - paired["b"]).to_numpy(dtype=np.float64)
            rows.append(
                {
                    "comparison_family": family,
                    "regime": "within_user_temporal_test",
                    "horizon_seconds": horizon,
                    "model_a": model_a,
                    "model_b": model_b,
                    "delta_definition": "MAE(model_a) - MAE(model_b); negative favors model_a",
                    **paired_bootstrap(difference, generator),
                    "users": len(difference),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "inference": "paired user bootstrap CI; paired two-sided Wilcoxon; Holm within three horizons",
                    "seed": SEED + 13,
                }
            )
    holm_adjust(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload: dict[str, object] = {
        "seed": SEED + 13,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "rows": len(rows),
        "users_per_comparison": sorted({int(row["users"]) for row in rows}),
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
    result.add_argument("--main-predictions", type=Path, required=True)
    result.add_argument("--baseline-predictions", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parser().parse_args())))
