from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.25.0"
SOURCE_EXPERIMENT_VERSION = "0.21.0"
HORIZONS = (60, 180, 300)
QUANTILE_MEDIAN_POSITION = 3
DEFAULT_SEEDS = (20260722, 20260723, 20260724)
DEFAULT_BOOTSTRAP_SEED = 20260725
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1
EWMA_ALPHA = 0.1
SPORT_FAMILIES = (
    "outdoor_cycling",
    "indoor_virtual_cycling",
    "running",
    "walking_hiking",
    "strength_cross_training",
)
LEARNED_REFERENCE_REGIMES = {
    "strict_temporal": "within_user_temporal_test",
    "unseen_user": "unseen_user_test",
    "goldencheetah_cross_source": "goldencheetah_frozen_external",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def hierarchical_paired_mae(
    main_prediction: np.ndarray,
    comparator_prediction: np.ndarray,
    target: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.DataFrame:
    require(
        len(main_prediction)
        == len(comparator_prediction)
        == len(target)
        == len(users)
        == len(sessions),
        "paired arrays differ in length",
    )
    frame = pd.DataFrame(
        {
            "user": users,
            "session": sessions,
            "main_absolute_error": np.abs(
                np.asarray(main_prediction, dtype=np.float64)
                - np.asarray(target, dtype=np.float64)
            ),
            "comparator_absolute_error": np.abs(
                np.asarray(comparator_prediction, dtype=np.float64)
                - np.asarray(target, dtype=np.float64)
            ),
        }
    )
    by_session = frame.groupby(["user", "session"], sort=True)[
        ["main_absolute_error", "comparator_absolute_error"]
    ].mean()
    by_user = by_session.groupby(level="user", sort=True).mean()
    by_user.columns = ["main_mae_bpm", "comparator_mae_bpm"]
    by_user["delta_mae_bpm"] = (
        by_user["main_mae_bpm"] - by_user["comparator_mae_bpm"]
    )
    require(len(by_user) >= 2, "paired user bootstrap requires at least two users")
    require(np.isfinite(by_user.to_numpy(dtype=np.float64)).all(), "non-finite user metric")
    return by_user


def mean_across_seeds(
    frames: list[pd.DataFrame], seeds: tuple[int, ...]
) -> pd.DataFrame:
    require(len(frames) == len(seeds), "seed-frame count mismatch")
    reference_index = frames[0].index
    for seed, frame in zip(seeds[1:], frames[1:], strict=True):
        require(
            frame.index.equals(reference_index),
            f"user support differs for seed {seed}",
        )
    stacked = pd.concat(frames, keys=seeds, names=["seed", "user"])
    averaged = stacked.groupby(level="user", sort=True).mean()
    require(
        np.allclose(
            averaged["delta_mae_bpm"],
            averaged["main_mae_bpm"] - averaged["comparator_mae_bpm"],
            rtol=0.0,
            atol=1e-12,
        ),
        "seed-averaged difference identity failed",
    )
    return averaged


def bootstrap_summary(
    user_frame: pd.DataFrame,
    replicates: int,
    seed: int,
    batch_size: int = 2_000,
) -> dict[str, float | int | bool]:
    values = user_frame["delta_mae_bpm"].to_numpy(dtype=np.float64)
    require(values.ndim == 1 and len(values) >= 2, "invalid bootstrap input")
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        end = min(replicates, start + batch_size)
        selections = generator.integers(
            0, len(values), size=(end - start, len(values))
        )
        bootstrap[start:end] = values[selections].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    delta = float(values.mean())
    return {
        "main_mae_bpm": float(user_frame["main_mae_bpm"].mean()),
        "comparator_mae_bpm": float(
            user_frame["comparator_mae_bpm"].mean()
        ),
        "delta_mae_bpm": delta,
        "ci_low_bpm": float(ci_low),
        "ci_high_bpm": float(ci_high),
        "ci_excludes_zero": bool(ci_high < 0.0 or ci_low > 0.0),
        "users": int(len(values)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def load_core_arrays(array_dir: Path) -> dict[str, np.ndarray]:
    filenames = {
        "targets": "targets.npy",
        "dataset": "dataset_code.npy",
        "unseen": "unseen_user_partition.npy",
        "external": "primary_external_partition.npy",
        "temporal": "temporal_partition_strict.npy",
        "users": "user_index.npy",
        "sessions": "session_index.npy",
        "values": "sequence_values.npy",
        "masks": "sequence_masks.npy",
    }
    arrays = {
        name: np.load(array_dir / filename, mmap_mode="r")
        for name, filename in filenames.items()
    }
    row_counts = {name: int(len(value)) for name, value in arrays.items()}
    require(len(set(row_counts.values())) == 1, f"core array mismatch: {row_counts}")
    require(arrays["targets"].shape[1] == len(HORIZONS), "target horizon mismatch")
    return arrays


def selected_learned_prediction(
    path: Path,
    regime: str,
    arrays: dict[str, np.ndarray],
    main: bool,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as source:
        require("row_index" in source.files, f"{path}: missing row_index")
        row_index = np.asarray(source["row_index"], dtype=np.int64)
        require(len(row_index) == len(np.unique(row_index)), f"{path}: duplicate rows")
        require(
            len(row_index) > 0
            and int(row_index.min()) >= 0
            and int(row_index.max()) < len(arrays["targets"]),
            f"{path}: invalid row index",
        )
        if regime == "strict_temporal":
            mask = (arrays["dataset"][row_index] == 0) & (
                arrays["temporal"][row_index] == PARTITION_TEST
            )
            key = "zero_history_quantiles" if main else "predictions"
        elif regime == "unseen_user":
            mask = (arrays["dataset"][row_index] == 0) & (
                arrays["unseen"][row_index] == PARTITION_TEST
            )
            key = "zero_history_quantiles" if main else "predictions"
        elif regime == "goldencheetah_cross_source":
            mask = (arrays["dataset"][row_index] == 1) & (
                arrays["external"][row_index] == EXTERNAL_FROZEN
            )
            key = "zero_history_quantiles" if main else "predictions"
        else:
            raise ValueError(f"unknown regime: {regime}")
        require(key in source.files, f"{path}: missing {key}")
        raw = np.asarray(source[key])
        prediction = raw[:, :, QUANTILE_MEDIAN_POSITION] if main else raw
        require(prediction.shape == (len(row_index), len(HORIZONS)), f"{path}: shape")
        selected_rows = row_index[mask]
        selected_prediction = np.asarray(prediction[mask], dtype=np.float64)
    require(len(selected_rows) > 0, f"{path}: empty {regime}")
    require(np.isfinite(selected_prediction).all(), f"{path}: non-finite prediction")
    return selected_rows, selected_prediction


def learned_paths(
    prediction_root: Path, seed: int, regime: str, comparator: str
) -> tuple[Path, Path]:
    seed_root = prediction_root / f"seed_{seed}"
    if regime == "strict_temporal":
        return (
            seed_root / "temporal_main" / "predictions.npz",
            seed_root / f"temporal_{comparator}" / "predictions.npz",
        )
    if regime == "unseen_user":
        return (
            seed_root / "unseen_main" / "development_predictions.npz",
            seed_root / f"unseen_{comparator}" / "predictions.npz",
        )
    if regime == "goldencheetah_cross_source":
        return (
            seed_root / "unseen_main" / "external_predictions.npz",
            seed_root / f"unseen_{comparator}" / "predictions.npz",
        )
    raise ValueError(regime)


def learned_comparator_rows(
    arrays: dict[str, np.ndarray],
    prediction_root: Path,
    seeds: tuple[int, ...],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    input_paths: set[Path],
) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    all_alignment_exact = True
    for regime in ("strict_temporal", "unseen_user", "goldencheetah_cross_source"):
        main_mode = "history_masked"
        for comparator in ("gru", "tcn"):
            per_horizon: dict[int, list[pd.DataFrame]] = {h: [] for h in HORIZONS}
            reference_rows: np.ndarray | None = None
            for seed in seeds:
                main_path, comparator_path = learned_paths(
                    prediction_root, seed, regime, comparator
                )
                require(main_path.exists(), f"missing {main_path}")
                require(comparator_path.exists(), f"missing {comparator_path}")
                input_paths.update((main_path, comparator_path))
                main_rows, main_prediction = selected_learned_prediction(
                    main_path, regime, arrays, main=True
                )
                comparator_rows, comparator_prediction = selected_learned_prediction(
                    comparator_path, regime, arrays, main=False
                )
                aligned = np.array_equal(main_rows, comparator_rows)
                all_alignment_exact &= aligned
                require(aligned, f"{regime}/{comparator}/seed {seed}: rows not aligned")
                if reference_rows is None:
                    reference_rows = main_rows
                else:
                    require(
                        np.array_equal(reference_rows, main_rows),
                        f"{regime}/{comparator}: support differs across seeds",
                    )
                for position, horizon in enumerate(HORIZONS):
                    per_horizon[horizon].append(
                        hierarchical_paired_mae(
                            main_prediction[:, position],
                            comparator_prediction[:, position],
                            np.asarray(arrays["targets"][main_rows, position]),
                            np.asarray(arrays["users"][main_rows]),
                            np.asarray(arrays["sessions"][main_rows]),
                        )
                    )
            for horizon in HORIZONS:
                averaged = mean_across_seeds(per_horizon[horizon], seeds)
                label = f"learned|{regime}|{comparator}|{horizon}"
                summary = bootstrap_summary(
                    averaged,
                    bootstrap_replicates,
                    stable_seed(bootstrap_seed, label),
                )
                rows.append(
                    {
                        "analysis_version": ANALYSIS_VERSION,
                        "source_experiment_version": SOURCE_EXPERIMENT_VERSION,
                        "regime": regime,
                        "horizon_seconds": horizon,
                        "main_model": "history_quantile_tcn",
                        "main_mode": main_mode,
                        "comparator_model": comparator,
                        "delta_definition": (
                            "MAE(main) - MAE(comparator); negative favors main"
                        ),
                        **summary,
                        "matched_seeds": ";".join(str(seed) for seed in seeds),
                        "n_matched_seeds": len(seeds),
                        "aggregation": (
                            "origin within session; session within user; paired user "
                            "effect averaged across matched seeds; equal-user mean"
                        ),
                        "inference": (
                            "95% percentile user bootstrap CI over seed-averaged "
                            "paired user MAE differences"
                        ),
                        "analysis_role": "post_hoc_sensitivity",
                    }
                )
    require(len(rows) == 18, f"expected 18 learned-comparator rows, got {len(rows)}")
    return rows, all_alignment_exact


def validate_learned_reference(
    rows: list[dict[str, object]],
    reference_path: Path,
    seeds: tuple[int, ...],
) -> float:
    require(reference_path.exists(), f"missing {reference_path}")
    reference = pd.read_csv(reference_path)
    required = {
        "seed",
        "comparator_model",
        "regime",
        "horizon_seconds",
        "main_mae_bpm",
        "comparator_mae_bpm",
        "main_minus_comparator_mae_bpm",
        "users",
    }
    require(required.issubset(reference.columns), "learned reference schema mismatch")
    maximum_delta = 0.0
    for row in rows:
        selected = reference[
            reference["seed"].isin(seeds)
            & (reference["comparator_model"] == row["comparator_model"])
            & (
                reference["regime"]
                == LEARNED_REFERENCE_REGIMES[str(row["regime"])]
            )
            & (reference["horizon_seconds"] == row["horizon_seconds"])
        ]
        require(
            len(selected) == len(seeds),
            "learned reference does not contain one row per matched seed",
        )
        require(
            set(selected["seed"].astype(int)) == set(seeds),
            "learned reference seed mismatch",
        )
        require(
            selected["users"].nunique() == 1
            and int(selected["users"].iloc[0]) == int(row["users"]),
            "learned reference user support mismatch",
        )
        comparisons = (
            ("main_mae_bpm", "main_mae_bpm"),
            ("comparator_mae_bpm", "comparator_mae_bpm"),
            ("delta_mae_bpm", "main_minus_comparator_mae_bpm"),
        )
        for output_column, reference_column in comparisons:
            maximum_delta = max(
                maximum_delta,
                abs(
                    float(row[output_column])
                    - float(selected[reference_column].mean())
                ),
            )
    require(
        maximum_delta <= 5e-6,
        f"learned comparator reference mismatch: {maximum_delta}",
    )
    return maximum_delta


def ewma_from_context(
    values: np.ndarray, masks: np.ndarray, alpha: float = EWMA_ALPHA
) -> np.ndarray:
    heart_rate = np.asarray(values[:, :, 0], dtype=np.float32)
    observed = np.asarray(masks[:, :, 0], dtype=bool)
    require(
        bool(np.all(observed.sum(axis=1) > 0)),
        "an evaluation context contains no observed heart rate",
    )
    ewma = np.zeros(len(heart_rate), dtype=np.float32)
    initialized = np.zeros(len(heart_rate), dtype=bool)
    for position in range(heart_rate.shape[1]):
        present = observed[:, position]
        first = present & ~initialized
        ewma[first] = heart_rate[first, position]
        update = present & initialized
        ewma[update] = (
            alpha * heart_rate[update, position]
            + (1.0 - alpha) * ewma[update]
        )
        initialized |= present
    return np.clip(ewma, 30.0, 240.0).astype(np.float64)


def load_sport_prediction(path: Path) -> tuple[np.ndarray, int, np.ndarray]:
    with np.load(path) as source:
        required = {"row_index", "same_user_rows", "history_quantiles"}
        require(required.issubset(source.files), f"{path}: missing fields")
        row_index = np.asarray(source["row_index"], dtype=np.int64)
        boundary = int(np.asarray(source["same_user_rows"])[0])
        quantiles = np.asarray(source["history_quantiles"])
    require(0 < boundary < len(row_index), f"{path}: invalid boundary")
    require(
        quantiles.shape == (len(row_index), len(HORIZONS), 7),
        f"{path}: prediction shape",
    )
    require(len(row_index) == len(np.unique(row_index)), f"{path}: duplicate rows")
    require(np.isfinite(quantiles).all(), f"{path}: non-finite prediction")
    return row_index, boundary, np.asarray(
        quantiles[:, :, QUANTILE_MEDIAN_POSITION], dtype=np.float64
    )


def sport_shift_rows(
    arrays: dict[str, np.ndarray],
    prediction_root: Path,
    aligned_reference_path: Path,
    seeds: tuple[int, ...],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    input_paths: set[Path],
) -> tuple[list[dict[str, object]], bool, float]:
    require(aligned_reference_path.exists(), f"missing {aligned_reference_path}")
    input_paths.add(aligned_reference_path)
    reference = pd.read_csv(aligned_reference_path)
    reference = reference[reference["model"] == "ewma_alpha_0_1"].copy()
    output_rows: list[dict[str, object]] = []
    all_alignment_exact = True
    maximum_reference_delta = 0.0
    for family in SPORT_FAMILIES:
        per_seed_payload: list[tuple[np.ndarray, int, np.ndarray]] = []
        for seed in seeds:
            path = (
                prediction_root
                / f"seed_{seed}"
                / "held_sport"
                / family
                / "predictions.npz"
            )
            require(path.exists(), f"missing {path}")
            input_paths.add(path)
            per_seed_payload.append(load_sport_prediction(path))
        reference_rows, reference_boundary, _ = per_seed_payload[0]
        for seed, (row_index, boundary, _) in zip(
            seeds[1:], per_seed_payload[1:], strict=True
        ):
            aligned = np.array_equal(reference_rows, row_index) and (
                reference_boundary == boundary
            )
            all_alignment_exact &= aligned
            require(aligned, f"{family}: support differs for seed {seed}")
        ewma = ewma_from_context(
            arrays["values"][reference_rows], arrays["masks"][reference_rows]
        )
        regimes = {
            "same_user_unseen_sport": slice(0, reference_boundary),
            "joint_user_sport": slice(reference_boundary, len(reference_rows)),
        }
        for regime, selected in regimes.items():
            selected_rows = reference_rows[selected]
            selected_ewma = ewma[selected]
            for position, horizon in enumerate(HORIZONS):
                frames: list[pd.DataFrame] = []
                for _, _, main_prediction in per_seed_payload:
                    frames.append(
                        hierarchical_paired_mae(
                            main_prediction[selected, position],
                            selected_ewma,
                            np.asarray(arrays["targets"][selected_rows, position]),
                            np.asarray(arrays["users"][selected_rows]),
                            np.asarray(arrays["sessions"][selected_rows]),
                        )
                    )
                averaged = mean_across_seeds(frames, seeds)
                label = f"sport|{family}|{regime}|{horizon}"
                summary = bootstrap_summary(
                    averaged,
                    bootstrap_replicates,
                    stable_seed(bootstrap_seed, label),
                )
                reference_regime = (
                    f"unseen_sport__{family}"
                    if regime == "same_user_unseen_sport"
                    else f"joint_user_sport__{family}"
                )
                reference_row = reference[
                    (reference["held_sport_family"] == family)
                    & (reference["regime"] == reference_regime)
                    & (reference["horizon_seconds"] == horizon)
                ]
                require(
                    len(reference_row) == 1,
                    f"missing EWMA reference {family}/{regime}/{horizon}",
                )
                expected = reference_row.iloc[0]
                maximum_reference_delta = max(
                    maximum_reference_delta,
                    abs(
                        float(summary["comparator_mae_bpm"])
                        - float(expected["mae_bpm"])
                    ),
                )
                require(
                    int(summary["users"]) == int(expected["users"]),
                    f"EWMA user support mismatch {family}/{regime}/{horizon}",
                )
                users = int(summary["users"])
                support_caution = regime == "joint_user_sport" and users < 25
                analysis_role = (
                    "exploratory_joint_shift_lt25_users"
                    if support_caution
                    else (
                        "exploratory_joint_shift"
                        if regime == "joint_user_sport"
                        else "post_hoc_sensitivity"
                    )
                )
                output_rows.append(
                    {
                        "analysis_version": ANALYSIS_VERSION,
                        "source_experiment_version": SOURCE_EXPERIMENT_VERSION,
                        "held_sport_family": family,
                        "regime": regime,
                        "horizon_seconds": horizon,
                        "main_model": "history_quantile_tcn",
                        "main_mode": "history_informed",
                        "comparator_model": "aligned_ewma_alpha_0_1",
                        "delta_definition": (
                            "MAE(main) - MAE(comparator); negative favors main"
                        ),
                        **summary,
                        "matched_seeds": ";".join(str(seed) for seed in seeds),
                        "n_matched_seeds": len(seeds),
                        "joint_support_caution_lt25_users": support_caution,
                        "aggregation": (
                            "origin within session; session within user; paired user "
                            "effect averaged across matched seeds; equal-user mean"
                        ),
                        "inference": (
                            "95% percentile user bootstrap CI over seed-averaged "
                            "paired user MAE differences"
                        ),
                        "analysis_role": analysis_role,
                    }
                )
    require(len(output_rows) == 30, f"expected 30 sport rows, got {len(output_rows)}")
    require(
        maximum_reference_delta <= 5e-6,
        f"reconstructed EWMA mismatch: {maximum_reference_delta}",
    )
    return output_rows, all_alignment_exact, maximum_reference_delta


def validate_summary_rows(
    rows: Iterable[dict[str, object]], expected_count: int
) -> dict[str, int]:
    frame = pd.DataFrame(rows)
    require(len(frame) == expected_count, "unexpected output count")
    numeric = frame[
        [
            "main_mae_bpm",
            "comparator_mae_bpm",
            "delta_mae_bpm",
            "ci_low_bpm",
            "ci_high_bpm",
        ]
    ].to_numpy(dtype=np.float64)
    checks = {
        "nonfinite_numeric_values": int((~np.isfinite(numeric)).sum()),
        "nonpositive_user_counts": int((frame["users"] < 2).sum()),
        "invalid_ci_order": int((frame["ci_low_bpm"] > frame["ci_high_bpm"]).sum()),
        "delta_identity_failures": int(
            (
                np.abs(
                    frame["delta_mae_bpm"]
                    - (frame["main_mae_bpm"] - frame["comparator_mae_bpm"])
                )
                > 1e-10
            ).sum()
        ),
    }
    require(not any(checks.values()), f"summary validation failed: {checks}")
    forbidden = {
        "user",
        "user_id",
        "user_index",
        "session",
        "session_id",
        "session_index",
        "row_index",
    }
    require(
        not forbidden.intersection(frame.columns),
        "row-level or identifier columns leaked to output",
    )
    return checks


def analyze(args: argparse.Namespace) -> dict[str, object]:
    require(args.bootstrap_replicates >= 1_000, "too few bootstrap replicates")
    seeds = tuple(int(seed) for seed in args.seeds)
    require(len(seeds) == 3 and len(set(seeds)) == 3, "exactly three unique seeds required")
    arrays = load_core_arrays(args.array_dir)
    input_paths: set[Path] = {
        args.array_dir / name
        for name in (
            "targets.npy",
            "dataset_code.npy",
            "unseen_user_partition.npy",
            "primary_external_partition.npy",
            "temporal_partition_strict.npy",
            "user_index.npy",
            "session_index.npy",
            "sequence_values.npy",
            "sequence_masks.npy",
        )
    }
    learned, learned_alignment = learned_comparator_rows(
        arrays,
        args.prediction_root,
        seeds,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        input_paths,
    )
    learned_reference_delta = validate_learned_reference(
        learned, args.learned_comparison_reference, seeds
    )
    input_paths.add(args.learned_comparison_reference)
    sport, sport_alignment, ewma_reference_delta = sport_shift_rows(
        arrays,
        args.prediction_root,
        args.aligned_sport_reference,
        seeds,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        input_paths,
    )
    learned_checks = validate_summary_rows(learned, 18)
    sport_checks = validate_summary_rows(sport, 30)
    atomic_csv(args.learned_output, learned)
    atomic_csv(args.sport_output, sport)
    output_hashes = {
        str(args.learned_output): sha256_file(args.learned_output),
        str(args.sport_output): sha256_file(args.sport_output),
    }
    input_hashes = {
        str(path): sha256_file(path) for path in sorted(input_paths, key=str)
    }
    small_joint_rows = sum(
        int(row["joint_support_caution_lt25_users"]) for row in sport
    )
    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "source_experiment_version": SOURCE_EXPERIMENT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "scope": (
            "Frozen-prediction post-processing only; no model fitting, retraining, "
            "checkpoint selection, adaptation, or recalibration. The aligned EWMA "
            "is deterministically reconstructed from the frozen five-minute context. "
            "Learned-comparator contrasts use the history-capable main model with its "
            "prior-workout input masked at inference, matching the existing architecture "
            "comparison estimand."
        ),
        "matched_seeds": list(seeds),
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "base_seed": args.bootstrap_seed,
            "unit": "user",
            "interval": "two-sided 95% percentile interval",
            "seed_combination": (
                "compute paired session-then-user MAE differences within seed, "
                "average each user's effect across three matched seeds, then bootstrap users"
            ),
        },
        "ewma": {
            "alpha": EWMA_ALPHA,
            "reconstruction": "deterministic causal update over frozen context",
            "reference_maximum_absolute_mae_delta_bpm": ewma_reference_delta,
        },
        "learned_comparator_reference": {
            "reference_maximum_absolute_metric_delta_bpm": learned_reference_delta,
            "reference": str(args.learned_comparison_reference),
        },
        "outputs": {
            "learned_comparator_rows": len(learned),
            "sport_shift_rows": len(sport),
            "joint_shift_rows_lt25_users": small_joint_rows,
            "contains_row_level_or_identifier_output": False,
            "sha256": output_hashes,
        },
        "input_sha256": input_hashes,
        "validation_checks": {
            "learned": learned_checks,
            "sport": sport_checks,
        },
        "assertions": {
            "exactly_three_matched_seeds": True,
            "learned_prediction_rows_exactly_aligned": learned_alignment,
            "held_sport_rows_and_regime_boundaries_exactly_aligned": sport_alignment,
            "user_support_exactly_aligned_across_seeds": True,
            "hierarchical_session_then_user_aggregation": True,
            "learned_comparisons_match_history_masked_architecture_estimand": all(
                row["main_mode"] == "history_masked" for row in learned
            ),
            "learned_metrics_match_existing_seed_summary_within_5e_minus_6_bpm": (
                learned_reference_delta <= 5e-6
            ),
            "ewma_reconstruction_matches_reference_within_5e_minus_6_bpm": (
                ewma_reference_delta <= 5e-6
            ),
            "bootstrap_randomness_fixed_and_label_derived": True,
            "no_row_level_or_identifier_outputs": True,
            "no_training_or_model_selection": True,
            "joint_cells_below_25_users_explicitly_exploratory": all(
                row["analysis_role"] == "exploratory_joint_shift_lt25_users"
                for row in sport
                if row["joint_support_caution_lt25_users"]
            ),
        },
    }
    audit["all_assertions_pass"] = bool(all(audit["assertions"].values()))
    require(bool(audit["all_assertions_pass"]), "audit assertion failed")
    atomic_json(args.audit, audit)
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "v0.25.0 paired-user bootstrap over three saved prediction seeds; "
            "no training or recalibration"
        )
    )
    result.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    result.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("outputs/q1_multiseed_v0_21_0"),
    )
    result.add_argument(
        "--aligned-sport-reference",
        type=Path,
        default=Path("outputs/results/sport_shift_aligned_baselines_v0_12_0.csv"),
    )
    result.add_argument(
        "--learned-comparison-reference",
        type=Path,
        default=Path(
            "outputs/q1_multiseed_v0_21_0/aggregation/"
            "main_vs_comparator_seed_paired_v0_22_0.csv"
        ),
    )
    result.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    result.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    result.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    result.add_argument(
        "--learned-output",
        type=Path,
        default=Path(
            "outputs/results/multiseed_paired_model_comparisons_v0_25_0.csv"
        ),
    )
    result.add_argument(
        "--sport-output",
        type=Path,
        default=Path(
            "outputs/results/multiseed_paired_sport_shift_v0_25_0.csv"
        ),
    )
    result.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "outputs/audit/multiseed_paired_user_bootstrap_v0_25_0.audit.json"
        ),
    )
    return result


def main() -> int:
    audit = analyze(parser().parse_args())
    print(
        json.dumps(
            {
                "analysis_version": audit["analysis_version"],
                "matched_seeds": audit["matched_seeds"],
                "outputs": audit["outputs"],
                "all_assertions_pass": audit["all_assertions_pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
