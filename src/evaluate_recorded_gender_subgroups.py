from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_model_comparisons import user_mae
from train_xgboost_baseline import HORIZONS


SEED = 20260737
REPLICATES = 10_000


def bootstrap_mean(values: np.ndarray, generator: np.random.Generator) -> tuple[float, float]:
    replicates = np.empty(REPLICATES, dtype=np.float64)
    for start in range(0, REPLICATES, 1000):
        end = min(REPLICATES, start + 1000)
        selected = generator.integers(0, len(values), size=(end - start, len(values)))
        replicates[start:end] = values[selected].mean(axis=1)
    return float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))


def bootstrap_difference(
    female: np.ndarray, male: np.ndarray, generator: np.random.Generator
) -> tuple[float, float, float]:
    replicates = np.empty(REPLICATES, dtype=np.float64)
    for start in range(0, REPLICATES, 1000):
        end = min(REPLICATES, start + 1000)
        female_selected = generator.integers(
            0, len(female), size=(end - start, len(female))
        )
        male_selected = generator.integers(
            0, len(male), size=(end - start, len(male))
        )
        replicates[start:end] = female[female_selected].mean(axis=1) - male[
            male_selected
        ].mean(axis=1)
    return (
        float(female.mean() - male.mean()),
        float(np.quantile(replicates, 0.025)),
        float(np.quantile(replicates, 0.975)),
    )


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    user_manifest = pd.read_csv(args.array_dir / "users.csv", dtype={"user_id": str})
    quality = pd.read_csv(
        args.endomondo_quality,
        usecols=["user_id", "gender"],
        dtype={"user_id": str},
        low_memory=False,
    )
    gender_counts_per_user = quality.groupby("user_id")["gender"].nunique()
    conflicts = int((gender_counts_per_user > 1).sum())
    if conflicts:
        raise AssertionError(f"inconsistent recorded gender for {conflicts} users")
    gender_lookup = quality.groupby("user_id", sort=False)["gender"].first()
    user_manifest["recorded_gender"] = user_manifest["user_id"].map(gender_lookup)
    user_gender = user_manifest.set_index("user_index")["recorded_gender"]

    targets = np.load(args.array_dir / "targets.npy", mmap_mode="r")
    users = np.load(args.array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(args.array_dir / "session_index.npy", mmap_mode="r")
    prediction_sources = {
        "unseen_user_test": args.unseen_predictions,
        "within_user_temporal_test": args.temporal_predictions,
    }
    generator = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    mapped_users: dict[str, int] = {}
    for regime, prediction_path in prediction_sources.items():
        with np.load(prediction_path) as prediction_file:
            row_index = prediction_file["row_index"]
            mode_predictions = {
                "history_informed": prediction_file["history_quantiles"][:, :, 3],
                "zero_history": prediction_file["zero_history_quantiles"][:, :, 3],
            }
        if regime == "unseen_user_test":
            dataset = np.load(args.array_dir / "dataset_code.npy", mmap_mode="r")
            unseen = np.load(
                args.array_dir / "unseen_user_partition.npy", mmap_mode="r"
            )
            selected = (dataset[row_index] == 0) & (unseen[row_index] == 4)
            row_index = row_index[selected]
            mode_predictions = {
                mode: prediction[selected]
                for mode, prediction in mode_predictions.items()
            }
        row_user_gender = (
            pd.Series(np.asarray(users[row_index]))
            .map(user_gender)
            .fillna("missing")
            .to_numpy()
        )
        mapped_users[regime] = int(
            user_gender.reindex(np.unique(users[row_index])).notna().sum()
        )
        for mode, prediction in mode_predictions.items():
            for position, horizon in enumerate(HORIZONS):
                user_values: dict[str, np.ndarray] = {}
                for gender in ("female", "male", "unknown", "missing"):
                    subset = row_user_gender == gender
                    if not subset.any():
                        continue
                    series = user_mae(
                        prediction[subset, position],
                        np.asarray(targets[row_index[subset], position]),
                        np.asarray(users[row_index[subset]]),
                        np.asarray(sessions[row_index[subset]]),
                    )
                    values = series.to_numpy(dtype=np.float64)
                    user_values[gender] = values
                    low, high = bootstrap_mean(values, generator)
                    rows.append(
                        {
                            "regime": regime,
                            "mode": mode,
                            "horizon_seconds": horizon,
                            "recorded_gender": gender,
                            "mae_bpm": float(values.mean()),
                            "ci_low_bpm": low,
                            "ci_high_bpm": high,
                            "users": len(values),
                            "sessions": int(
                                len(np.unique(sessions[row_index[subset]]))
                            ),
                            "origins": int(subset.sum()),
                            "support_status": (
                                "supported_descriptive"
                                if len(values) >= 50
                                else "exploratory_small_subgroup"
                            ),
                            "interpretation": "descriptive recorded-gender subgroup; not a biological or causal contrast",
                        }
                    )
                if "female" in user_values and "male" in user_values:
                    delta, low, high = bootstrap_difference(
                        user_values["female"], user_values["male"], generator
                    )
                    difference_rows.append(
                        {
                            "regime": regime,
                            "mode": mode,
                            "horizon_seconds": horizon,
                            "contrast": "female_minus_male_mae",
                            "delta_mae_bpm": delta,
                            "ci_low_bpm": low,
                            "ci_high_bpm": high,
                            "female_users": len(user_values["female"]),
                            "male_users": len(user_values["male"]),
                            "support_status": (
                                "supported_descriptive"
                                if len(user_values["female"]) >= 50
                                else "exploratory_small_subgroup"
                            ),
                            "interpretation": "unadjusted descriptive contrast; activity mix and other confounding remain",
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.differences.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(difference_rows[0]))
        writer.writeheader()
        writer.writerows(difference_rows)
    payload: dict[str, object] = {
        "seed": SEED,
        "bootstrap_replicates": REPLICATES,
        "recorded_gender_conflicts": conflicts,
        "mapped_users": mapped_users,
        "subgroup_rows": len(rows),
        "difference_rows": len(difference_rows),
        "goldencheetah_gender_limitation": "144 male and 4 female among 148 valid metadata records; no external gender-stratified inference",
        "all_assertions_pass": conflicts == 0 and len(rows) > 0,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--endomondo-quality", type=Path, required=True)
    result.add_argument("--unseen-predictions", type=Path, required=True)
    result.add_argument("--temporal-predictions", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--differences", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parser().parse_args())))
