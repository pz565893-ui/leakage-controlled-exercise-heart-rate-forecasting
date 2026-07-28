from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from train_xgboost_baseline import EXTERNAL_FROZEN, HORIZONS, PARTITION_TEST


SEED = 20260722
BOOTSTRAP_REPLICATES = 10_000


def user_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "user": users,
            "session": sessions,
            "absolute_error": np.abs(prediction - target),
        }
    )
    session = frame.groupby(["user", "session"], sort=False)["absolute_error"].mean()
    return session.groupby(level="user", sort=False).mean()


def paired_bootstrap(
    difference: np.ndarray, generator: np.random.Generator
) -> dict[str, float]:
    observed = float(difference.mean())
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 1000):
        end = min(BOOTSTRAP_REPLICATES, start + 1000)
        selections = generator.integers(
            0, len(difference), size=(end - start, len(difference))
        )
        replicates[start:end] = difference[selections].mean(axis=1)
    return {
        "delta_mae_bpm": observed,
        "ci_low_bpm": float(np.quantile(replicates, 0.025)),
        "ci_high_bpm": float(np.quantile(replicates, 0.975)),
        "paired_wilcoxon_p_value": float(
            wilcoxon(difference, alternative="two-sided", method="auto").pvalue
        ),
    }


def holm_adjust(rows: list[dict[str, object]]) -> None:
    families: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        families.setdefault(str(row["comparison_family"]), []).append(index)
    for indices in families.values():
        ordered = sorted(
            indices, key=lambda index: float(rows[index]["paired_wilcoxon_p_value"])
        )
        running = 0.0
        m = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(
                1.0, (m - rank) * float(rows[index]["paired_wilcoxon_p_value"])
            )
            running = max(running, adjusted)
            rows[index]["holm_adjusted_wilcoxon_p_value"] = running


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    arrays = args.array_dir
    targets = np.load(arrays / "targets.npy", mmap_mode="r")
    dataset = np.load(arrays / "dataset_code.npy", mmap_mode="r")
    unseen = np.load(arrays / "unseen_user_partition.npy", mmap_mode="r")
    external = np.load(arrays / "primary_external_partition.npy", mmap_mode="r")
    users = np.load(arrays / "user_index.npy", mmap_mode="r")
    sessions = np.load(arrays / "session_index.npy", mmap_mode="r")
    main = np.load(args.main_predictions)
    gru = np.load(args.gru_predictions)
    if not np.array_equal(main["row_index"], gru["row_index"]):
        raise AssertionError("main and GRU prediction rows are not aligned")
    row_index = main["row_index"]
    history = main["history_quantiles"][:, :, 3]
    zero = main["zero_history_quantiles"][:, :, 3]
    gru_prediction = gru["predictions"]
    subsets = {
        "unseen_user_test": (dataset[row_index] == 0)
        & (unseen[row_index] == PARTITION_TEST),
        "goldencheetah_frozen_external": (dataset[row_index] == 1)
        & (external[row_index] == EXTERNAL_FROZEN),
    }
    comparisons = [
        (
            "unseen_history_vs_gru",
            "unseen_user_test",
            "history_quantile_tcn_history_informed",
            history,
            "gru",
            gru_prediction,
        ),
        (
            "unseen_history_vs_zero",
            "unseen_user_test",
            "history_quantile_tcn_history_informed",
            history,
            "history_quantile_tcn_zero_history",
            zero,
        ),
        (
            "external_zero_vs_gru",
            "goldencheetah_frozen_external",
            "history_quantile_tcn_zero_history",
            zero,
            "gru",
            gru_prediction,
        ),
    ]
    generator = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for family, regime, model_a, prediction_a, model_b, prediction_b in comparisons:
        subset = subsets[regime]
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
            delta = (paired["a"] - paired["b"]).to_numpy(dtype=np.float64)
            rows.append(
                {
                    "comparison_family": family,
                    "regime": regime,
                    "horizon_seconds": horizon,
                    "model_a": model_a,
                    "model_b": model_b,
                    "delta_definition": "MAE(model_a) - MAE(model_b); negative favors model_a",
                    **paired_bootstrap(delta, generator),
                    "users": len(delta),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "inference": "paired user bootstrap CI; paired two-sided Wilcoxon; Holm within three horizons",
                    "seed": SEED,
                }
            )
    holm_adjust(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "rows": len(rows),
        "families": sorted({row["comparison_family"] for row in rows}),
        "all_user_counts_positive": all(int(row["users"]) > 0 for row in rows),
        "output": str(args.output),
    }
    payload["all_assertions_pass"] = payload["all_user_counts_positive"] and len(rows) == 9
    temporary = args.audit.with_suffix(args.audit.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.audit)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired user-clustered model bootstrap.")
    parser.add_argument("--array-dir", type=Path, required=True)
    parser.add_argument("--main-predictions", type=Path, required=True)
    parser.add_argument("--gru-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
