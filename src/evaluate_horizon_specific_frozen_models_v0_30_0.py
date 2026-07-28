"""Evaluate five frozen main-model seeds on horizon-specific extra origins.

No fitting, checkpoint selection, normalization, calibration, or target-source
adaptation is performed.  Authoritative saved predictions are retained for the
common-three-target rows; only additional fixed-session origins are newly
inferred.  Before inference is accepted, each checkpoint must reproduce a
deterministic sample of its saved common-cohort quantiles and the complete
common-cohort hierarchical MAE recorded by the v0.22 aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_uncertainty_model import HistoryQuantileTCN, predict_quantiles
from train_xgboost_baseline import hierarchical_metrics


VERSION = "0.30.0"
SEEDS = (20260722, 20260723, 20260724, 20260725, 20260726)
HORIZONS = (60, 180, 300)
QUANTILE_MEDIAN_INDEX = 3
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260728
REINFERENCE_BATCH_SIZE = 4096
REINFERENCE_BATCH_COUNT = 3
REINFERENCE_TOLERANCE_BPM = 1e-6
COMMON_MAE_TOLERANCE_BPM = 1e-6

REGIMES = {
    "within_user_temporal_test": {
        "bit": 1,
        "experiment": "temporal_main",
        "mode": "history_informed",
        "force_zero_history": False,
        "protocol": "temporal",
    },
    "unseen_user_test": {
        "bit": 2,
        "experiment": "unseen_main",
        "mode": "history_informed",
        "force_zero_history": False,
        "protocol": "unseen",
    },
    "goldencheetah_frozen_external": {
        "bit": 4,
        "experiment": "unseen_main",
        "mode": "zero_history",
        "force_zero_history": True,
        "protocol": "unseen",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_arrays(directory: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in names}


def model_paths(root: Path, seed: int, protocol: str) -> dict[str, Path]:
    seed_root = root / f"seed_{seed}"
    if protocol == "temporal":
        model_dir = seed_root / "temporal_main" / "model"
        return {
            "checkpoint": model_dir / "temporal_history_quantile_tcn_best.pt",
            "input_normalization": model_dir / "input_normalization.json",
            "history_normalization": model_dir / "history_normalization.json",
            "predictions": seed_root / "temporal_main" / "predictions.npz",
        }
    if protocol == "unseen":
        model_dir = seed_root / "unseen_main" / "model"
        return {
            "checkpoint": model_dir / "history_quantile_tcn_best_v0_11_0.pt",
            "input_normalization": model_dir / "normalization_unseen_user_train.json",
            "history_normalization": model_dir
            / "history_normalization_unseen_user_train.json",
            "development_predictions": seed_root
            / "unseen_main"
            / "development_predictions.npz",
            "external_predictions": seed_root
            / "unseen_main"
            / "external_predictions.npz",
        }
    raise ValueError(f"unknown protocol: {protocol}")


def load_model_and_normalization(
    paths: dict[str, Path], device: torch.device
) -> tuple[HistoryQuantileTCN, dict[str, object], dict[str, object]]:
    for key in ("checkpoint", "input_normalization", "history_normalization"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    model = HistoryQuantileTCN().to(device)
    with paths["checkpoint"].open("rb") as handle:
        saved = torch.load(handle, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    input_normalization = json.loads(
        paths["input_normalization"].read_text(encoding="utf-8")
    )
    history_normalization = json.loads(
        paths["history_normalization"].read_text(encoding="utf-8")
    )
    return model, input_normalization, history_normalization


def common_saved_predictions(
    regime: str,
    paths: dict[str, Path],
    global_arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if regime == "within_user_temporal_test":
        with np.load(paths["predictions"]) as archive:
            rows = np.asarray(archive["row_index"], dtype=np.int64)
            prediction = np.asarray(archive["history_quantiles"], dtype=np.float32)
        expected = np.flatnonzero(
            (global_arrays["dataset"] == 0)
            & (global_arrays["evaluation"] == 1)
            & (global_arrays["strict"] == 4)
        )
    elif regime == "unseen_user_test":
        with np.load(paths["development_predictions"]) as archive:
            all_rows = np.asarray(archive["row_index"], dtype=np.int64)
            selected = (
                (global_arrays["dataset"][all_rows] == 0)
                & (global_arrays["evaluation"][all_rows] == 1)
                & (global_arrays["unseen"][all_rows] == 4)
            )
            rows = all_rows[selected]
            prediction = np.asarray(
                archive["history_quantiles"][selected], dtype=np.float32
            )
        expected = np.flatnonzero(
            (global_arrays["dataset"] == 0)
            & (global_arrays["evaluation"] == 1)
            & (global_arrays["unseen"] == 4)
        )
    elif regime == "goldencheetah_frozen_external":
        with np.load(paths["external_predictions"]) as archive:
            rows = np.asarray(archive["row_index"], dtype=np.int64)
            prediction = np.asarray(
                archive["zero_history_quantiles"], dtype=np.float32
            )
        expected = np.flatnonzero(
            (global_arrays["dataset"] == 1)
            & (global_arrays["evaluation"] == 1)
            & (global_arrays["external"] == 1)
            & np.isin(global_arrays["sport"], [1, 2, 3])
        )
    else:
        raise ValueError(regime)
    if not np.array_equal(rows, expected):
        raise AssertionError(f"saved common prediction rows do not match {regime}")
    if prediction.shape != (len(rows), 3, 7):
        raise AssertionError(f"unexpected saved prediction shape for {regime}")
    return rows, prediction


def q1_expected_mae(
    q1: pd.DataFrame, seed: int, regime: str, horizon: int
) -> float:
    spec = REGIMES[regime]
    selected = q1[
        (q1.seed == seed)
        & (q1.experiment == spec["experiment"])
        & (q1.regime == regime)
        & (q1["mode"] == spec["mode"])
        & (q1.source_kind == "point")
        & (q1.metric == "mae_bpm")
        & (q1.horizon_seconds == horizon)
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"expected one v0.22 common metric row, found {len(selected)}"
        )
    return float(selected.iloc[0].value)


def per_user_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "user": np.asarray(users, dtype=np.int32),
            "session": np.asarray(sessions, dtype=np.int32),
            "absolute_error": np.abs(
                np.asarray(prediction, dtype=np.float32)
                - np.asarray(target, dtype=np.float32)
            ),
        }
    )
    by_session = frame.groupby(["user", "session"], sort=False)[
        "absolute_error"
    ].mean()
    return by_session.groupby(level="user", sort=False).mean().sort_index()


def bootstrap_interval(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    n = len(values)
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        stop = min(start + 250, BOOTSTRAP_REPLICATES)
        draws = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = values[draws].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deterministic frozen-model reproduction")
    if args.batch_size != REINFERENCE_BATCH_SIZE:
        raise ValueError(
            f"batch size must reproduce the original inference contract: "
            f"{REINFERENCE_BATCH_SIZE}"
        )
    device = torch.device("cuda")
    global_arrays = load_arrays(
        args.array_dir,
        (
            "sequence_values",
            "sequence_masks",
            "targets",
            "origin_offset_seconds",
            "sport_code",
            "dataset_code",
            "evaluation_origin",
            "unseen_user_partition",
            "primary_external_partition",
            "user_index",
            "session_index",
            "session_history_values",
            "session_history_mask",
            "temporal_partition_strict",
        ),
    )
    global_arrays = {
        "values": global_arrays["sequence_values"],
        "masks": global_arrays["sequence_masks"],
        "targets": global_arrays["targets"],
        "elapsed": global_arrays["origin_offset_seconds"],
        "sport": global_arrays["sport_code"],
        "dataset": global_arrays["dataset_code"],
        "evaluation": global_arrays["evaluation_origin"],
        "unseen": global_arrays["unseen_user_partition"],
        "external": global_arrays["primary_external_partition"],
        "users": global_arrays["user_index"],
        "sessions": global_arrays["session_index"],
        "history_values": global_arrays["session_history_values"],
        "history_mask": global_arrays["session_history_mask"],
        "strict": global_arrays["temporal_partition_strict"],
    }
    extra = load_arrays(
        args.extra_dir,
        (
            "sequence_values",
            "sequence_masks",
            "targets",
            "target_mask",
            "origin_offset_seconds",
            "sport_code",
            "user_index",
            "session_index",
            "regime_flags",
        ),
    )
    extra = {
        "values": extra["sequence_values"],
        "masks": extra["sequence_masks"],
        "targets": extra["targets"],
        "target_mask": extra["target_mask"],
        "elapsed": extra["origin_offset_seconds"],
        "sport": extra["sport_code"],
        "users": extra["user_index"],
        "sessions": extra["session_index"],
        "regime_flags": extra["regime_flags"],
    }
    array_audit = json.loads(args.extra_audit.read_text(encoding="utf-8"))
    if not array_audit.get("all_assertions_pass"):
        raise AssertionError("extra-array audit did not pass")
    q1 = pd.read_csv(args.q1_metrics)

    detail_rows: list[dict[str, object]] = []
    common_checks: list[dict[str, object]] = []
    reinference_checks: list[dict[str, object]] = []
    user_deltas: defaultdict[tuple[str, int], list[pd.Series]] = defaultdict(list)
    checkpoint_hashes: dict[str, str] = {}

    for seed in SEEDS:
        loaded_models: dict[str, tuple[HistoryQuantileTCN, dict[str, object], dict[str, object], dict[str, Path]]] = {}
        for protocol in ("temporal", "unseen"):
            paths = model_paths(args.model_root, seed, protocol)
            model, input_norm, history_norm = load_model_and_normalization(paths, device)
            loaded_models[protocol] = (model, input_norm, history_norm, paths)
            checkpoint_hashes[f"{seed}_{protocol}"] = sha256_file(paths["checkpoint"])

        for regime, spec in REGIMES.items():
            protocol = str(spec["protocol"])
            model, input_norm, history_norm, paths = loaded_models[protocol]
            common_rows, common_quantiles = common_saved_predictions(
                regime, paths, global_arrays
            )
            common_point = common_quantiles[:, :, QUANTILE_MEDIAN_INDEX]

            available_starts = [
                0,
                (len(common_rows) // 2 // REINFERENCE_BATCH_SIZE)
                * REINFERENCE_BATCH_SIZE,
                (len(common_rows) // REINFERENCE_BATCH_SIZE - 1)
                * REINFERENCE_BATCH_SIZE,
            ]
            batch_starts = sorted(set(available_starts))
            if len(batch_starts) != REINFERENCE_BATCH_COUNT:
                raise AssertionError("could not select three distinct full replay batches")
            sample_positions = np.concatenate(
                [
                    np.arange(start, start + REINFERENCE_BATCH_SIZE, dtype=np.int64)
                    for start in batch_starts
                ]
            )
            sample_rows = common_rows[sample_positions]
            repeated = predict_quantiles(
                model,
                sample_rows,
                global_arrays["values"],
                global_arrays["masks"],
                global_arrays["elapsed"],
                global_arrays["sport"],
                global_arrays["sessions"],
                global_arrays["history_values"],
                global_arrays["history_mask"],
                input_norm,
                history_norm,
                device,
                args.batch_size,
                bool(spec["force_zero_history"]),
            )
            max_difference = float(
                np.max(np.abs(repeated - common_quantiles[sample_positions]))
            )
            reinference_check = {
                "seed": seed,
                "regime": regime,
                "sample_rows": int(len(sample_rows)),
                "max_absolute_quantile_difference_bpm": max_difference,
                "tolerance_bpm": REINFERENCE_TOLERANCE_BPM,
                "pass": max_difference <= REINFERENCE_TOLERANCE_BPM,
            }
            reinference_checks.append(reinference_check)
            if not reinference_check["pass"]:
                raise AssertionError(f"saved-prediction reproduction failed: {reinference_check}")

            extra_index = np.flatnonzero(
                (extra["regime_flags"] & int(spec["bit"])) != 0
            )
            extra_quantiles = predict_quantiles(
                model,
                extra_index,
                extra["values"],
                extra["masks"],
                extra["elapsed"],
                extra["sport"],
                extra["sessions"],
                global_arrays["history_values"],
                global_arrays["history_mask"],
                input_norm,
                history_norm,
                device,
                args.batch_size,
                bool(spec["force_zero_history"]),
            )
            extra_point = extra_quantiles[:, :, QUANTILE_MEDIAN_INDEX]

            for position, horizon in enumerate(HORIZONS):
                common_prediction = common_point[:, position]
                common_target = np.asarray(
                    global_arrays["targets"][common_rows, position], dtype=np.float32
                )
                common_users = np.asarray(
                    global_arrays["users"][common_rows], dtype=np.int32
                )
                common_sessions = np.asarray(
                    global_arrays["sessions"][common_rows], dtype=np.int32
                )
                common_metric = hierarchical_metrics(
                    common_prediction,
                    common_target,
                    common_users,
                    common_sessions,
                )
                expected_mae = q1_expected_mae(q1, seed, regime, horizon)
                mae_difference = abs(float(common_metric["mae_bpm"]) - expected_mae)
                common_check = {
                    "seed": seed,
                    "regime": regime,
                    "horizon_seconds": horizon,
                    "observed_mae_bpm": float(common_metric["mae_bpm"]),
                    "expected_v0_22_mae_bpm": expected_mae,
                    "absolute_difference_bpm": mae_difference,
                    "pass": mae_difference <= COMMON_MAE_TOLERANCE_BPM,
                }
                common_checks.append(common_check)
                if not common_check["pass"]:
                    raise AssertionError(f"common MAE reproduction failed: {common_check}")

                available_local = np.flatnonzero(
                    extra["target_mask"][extra_index, position] == 1
                )
                selected_extra = extra_index[available_local]
                selected_prediction = extra_point[available_local, position]
                expanded_prediction = np.concatenate(
                    [common_prediction, selected_prediction]
                )
                expanded_target = np.concatenate(
                    [
                        common_target,
                        np.asarray(
                            extra["targets"][selected_extra, position],
                            dtype=np.float32,
                        ),
                    ]
                )
                expanded_users = np.concatenate(
                    [common_users, np.asarray(extra["users"][selected_extra])]
                )
                expanded_sessions = np.concatenate(
                    [common_sessions, np.asarray(extra["sessions"][selected_extra])]
                )
                expanded_metric = hierarchical_metrics(
                    expanded_prediction,
                    expanded_target,
                    expanded_users,
                    expanded_sessions,
                )
                if int(expanded_metric["users"]) != int(common_metric["users"]):
                    raise AssertionError("expanded cohort changed user support")
                if int(expanded_metric["sessions"]) != int(common_metric["sessions"]):
                    raise AssertionError("expanded cohort changed session support")
                if int(expanded_metric["origins"]) != int(common_metric["origins"]) + len(
                    selected_extra
                ):
                    raise AssertionError("expanded origin count is inconsistent")

                common_user = per_user_mae(
                    common_prediction,
                    common_target,
                    common_users,
                    common_sessions,
                )
                expanded_user = per_user_mae(
                    expanded_prediction,
                    expanded_target,
                    expanded_users,
                    expanded_sessions,
                )
                if not common_user.index.equals(expanded_user.index):
                    raise AssertionError("expanded per-user support changed")
                user_deltas[(regime, horizon)].append(expanded_user - common_user)

                for cohort, metric in (
                    ("common_three_target", common_metric),
                    ("horizon_specific", expanded_metric),
                ):
                    detail_rows.append(
                        {
                            "analysis_version": VERSION,
                            "seed": seed,
                            "regime": regime,
                            "mode": spec["mode"],
                            "horizon_seconds": horizon,
                            "cohort": cohort,
                            "mae_bpm": float(metric["mae_bpm"]),
                            "rmse_bpm": float(metric["rmse_bpm"]),
                            "bias_bpm": float(metric["bias_bpm"]),
                            "users": int(metric["users"]),
                            "sessions": int(metric["sessions"]),
                            "origins": int(metric["origins"]),
                        }
                    )
            del extra_quantiles, extra_point, repeated
            torch.cuda.empty_cache()
        del loaded_models
        torch.cuda.empty_cache()
        print(f"completed frozen inference for seed {seed}", flush=True)

    detail = pd.DataFrame(detail_rows).sort_values(
        ["regime", "horizon_seconds", "seed", "cohort"]
    )
    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for regime in REGIMES:
        for horizon in HORIZONS:
            selected = detail[
                (detail.regime == regime) & (detail.horizon_seconds == horizon)
            ]
            common = selected[selected.cohort == "common_three_target"].set_index(
                "seed"
            )
            expanded = selected[selected.cohort == "horizon_specific"].set_index(
                "seed"
            )
            if not common.index.equals(expanded.index):
                raise AssertionError("seed pairing failed")
            seed_delta = expanded.mae_bpm - common.mae_bpm
            deltas = user_deltas[(regime, horizon)]
            if len(deltas) != len(SEEDS):
                raise AssertionError("missing per-seed user deltas")
            reference_index = deltas[0].index
            if any(not item.index.equals(reference_index) for item in deltas[1:]):
                raise AssertionError("per-user seed alignment failed")
            mean_user_delta = np.stack(
                [item.to_numpy(dtype=np.float64) for item in deltas], axis=1
            ).mean(axis=1)
            estimate = float(mean_user_delta.mean())
            lower, upper = bootstrap_interval(mean_user_delta, rng)
            summary_rows.append(
                {
                    "analysis_version": VERSION,
                    "regime": regime,
                    "mode": REGIMES[regime]["mode"],
                    "horizon_seconds": horizon,
                    "seeds": ";".join(str(item) for item in SEEDS),
                    "n_seeds": len(SEEDS),
                    "common_mae_median_bpm": float(common.mae_bpm.median()),
                    "common_mae_min_bpm": float(common.mae_bpm.min()),
                    "common_mae_max_bpm": float(common.mae_bpm.max()),
                    "horizon_specific_mae_median_bpm": float(
                        expanded.mae_bpm.median()
                    ),
                    "horizon_specific_mae_min_bpm": float(expanded.mae_bpm.min()),
                    "horizon_specific_mae_max_bpm": float(expanded.mae_bpm.max()),
                    "expanded_minus_common_seed_median_bpm": float(
                        seed_delta.median()
                    ),
                    "expanded_minus_common_seed_min_bpm": float(seed_delta.min()),
                    "expanded_minus_common_seed_max_bpm": float(seed_delta.max()),
                    "paired_user_estimate_bpm": estimate,
                    "paired_user_ci_lower_bpm": lower,
                    "paired_user_ci_upper_bpm": upper,
                    "users": int(expanded.users.iloc[0]),
                    "sessions": int(expanded.sessions.iloc[0]),
                    "common_origins": int(common.origins.iloc[0]),
                    "horizon_specific_origins": int(expanded.origins.iloc[0]),
                    "added_origins": int(expanded.origins.iloc[0] - common.origins.iloc[0]),
                }
            )
    summary = pd.DataFrame(summary_rows)
    atomic_csv(args.detail_output, detail)
    atomic_csv(args.summary_output, summary)

    audit: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "analysis_version": VERSION,
        "design": (
            "authoritative saved predictions on common rows plus new inference only "
            "for fixed-session horizon-specific extra rows"
        ),
        "frozen_contract": {
            "training": False,
            "checkpoint_selection": False,
            "normalization_refit": False,
            "calibration": False,
            "external_adaptation": False,
            "seeds": list(SEEDS),
        },
        "reinference_design": {
            "full_contiguous_batches_per_seed_regime": REINFERENCE_BATCH_COUNT,
            "original_inference_batch_size": REINFERENCE_BATCH_SIZE,
        },
        "reinference_checks": reinference_checks,
        "common_mae_checks": common_checks,
        "extra_array_audit": str(args.extra_audit.resolve()),
        "extra_array_audit_sha256": sha256_file(args.extra_audit),
        "checkpoint_sha256": checkpoint_hashes,
        "aggregation": "origin within session, session within user, equal-user mean",
        "paired_uncertainty": {
            "method": "per-user differences averaged across five matched seeds, then user bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "outputs": {
            "detail": str(args.detail_output.resolve()),
            "summary": str(args.summary_output.resolve()),
        },
        "output_sha256": {
            "detail": sha256_file(args.detail_output),
            "summary": sha256_file(args.summary_output),
        },
        "privacy": "aggregate outputs only; per-user bootstrap inputs were not saved",
        "all_assertions_pass": True,
    }
    atomic_json(args.audit, audit)
    print(json.dumps(audit, indent=2))
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    result.add_argument(
        "--extra-dir",
        type=Path,
        default=Path("outputs/features/horizon_specific_extra_v0_30_0"),
    )
    result.add_argument(
        "--extra-audit",
        type=Path,
        default=Path("outputs/audit/horizon_specific_extra_arrays_v0_30_0.json"),
    )
    result.add_argument(
        "--model-root",
        type=Path,
        default=Path("outputs/q1_multiseed_v0_21_0"),
    )
    result.add_argument(
        "--q1-metrics",
        type=Path,
        default=Path(
            "outputs/q1_multiseed_v0_21_0/aggregation/per_seed_metrics_long_v0_22_0.csv"
        ),
    )
    result.add_argument(
        "--detail-output",
        type=Path,
        default=Path("outputs/results/horizon_specific_frozen_model_per_seed_v0_30_0.csv"),
    )
    result.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/results/horizon_specific_frozen_model_summary_v0_30_0.csv"),
    )
    result.add_argument(
        "--audit",
        type=Path,
        default=Path("outputs/audit/horizon_specific_frozen_models_v0_30_0.json"),
    )
    result.add_argument("--batch-size", type=int, default=REINFERENCE_BATCH_SIZE)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
