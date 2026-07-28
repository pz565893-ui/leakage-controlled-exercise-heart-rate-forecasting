from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.19.0"
SOURCE_MODEL_VERSION = "0.12.0"
HORIZONS = (60, 180, 300)
PARTITION_TRAIN = 1
PARTITION_TEST = 4
SPORTS = {
    1: "outdoor_cycling",
    2: "indoor_virtual_cycling",
    3: "running",
    4: "walking_hiking",
    7: "strength_cross_training",
}


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("no output rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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


def bootstrap_user_mean(
    values: np.ndarray,
    replicates: int,
    seed: int,
    batch_size: int = 2_000,
) -> np.ndarray:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("expected at least two user-level values")
    generator = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        end = min(replicates, start + batch_size)
        indices = generator.integers(
            0, len(values), size=(end - start, len(values))
        )
        result[start:end] = values[indices].mean(axis=1)
    return result


def hierarchical_user_mae(
    absolute_error: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "absolute_error": absolute_error,
            "user": users,
            "session": sessions,
        }
    )
    session_level = frame.groupby(["user", "session"], sort=False)[
        "absolute_error"
    ].mean()
    return session_level.groupby(level="user", sort=False).mean()


def analyze(args: argparse.Namespace) -> dict[str, object]:
    arrays = {
        "targets": np.load(args.array_dir / "targets.npy", mmap_mode="r"),
        "dataset": np.load(args.array_dir / "dataset_code.npy", mmap_mode="r"),
        "evaluation": np.load(
            args.array_dir / "evaluation_origin.npy", mmap_mode="r"
        ),
        "unseen": np.load(
            args.array_dir / "unseen_user_partition.npy", mmap_mode="r"
        ),
        "sport": np.load(args.array_dir / "sport_code.npy", mmap_mode="r"),
        "users": np.load(args.array_dir / "user_index.npy", mmap_mode="r"),
        "sessions": np.load(args.array_dir / "session_index.npy", mmap_mode="r"),
    }
    row_counts = {key: int(len(value)) for key, value in arrays.items()}
    require(len(set(row_counts.values())) == 1, f"array mismatch: {row_counts}")
    total_rows = next(iter(row_counts.values()))

    reference = pd.read_csv(args.reference_csv)
    output_rows: list[dict[str, object]] = []
    family_audits: dict[str, object] = {}
    maximum_reference_delta = 0.0

    for sport_code, family in SPORTS.items():
        prediction_path = (
            args.prediction_dir / f"sport_shift_{family}_v0_12_0.npz"
        )
        require(prediction_path.exists(), f"missing {prediction_path}")
        with np.load(prediction_path) as source:
            require(
                {"row_index", "same_user_rows", "history_quantiles"}.issubset(
                    source.files
                ),
                f"{family}: missing prediction fields",
            )
            row_index = np.asarray(source["row_index"], dtype=np.int64)
            same_user_marker = np.asarray(source["same_user_rows"])
            history_quantiles = np.asarray(source["history_quantiles"])

        require(
            same_user_marker.shape == (1,), f"{family}: invalid regime marker"
        )
        same_user_rows = int(same_user_marker[0])
        require(
            0 < same_user_rows < len(row_index),
            f"{family}: invalid regime boundary",
        )
        require(
            len(np.unique(row_index)) == len(row_index)
            and int(row_index.min()) >= 0
            and int(row_index.max()) < total_rows,
            f"{family}: invalid row index",
        )
        require(
            history_quantiles.shape == (len(row_index), 3, 7),
            f"{family}: prediction shape mismatch",
        )
        require(
            np.isfinite(history_quantiles).all(),
            f"{family}: non-finite predictions",
        )
        require(
            int((np.diff(history_quantiles, axis=2) < -1e-6).sum()) == 0,
            f"{family}: quantile crossings",
        )

        train_index = np.flatnonzero(
            (arrays["dataset"] == 0)
            & (arrays["unseen"] == PARTITION_TRAIN)
            & (arrays["sport"] != sport_code)
        )
        train_users = np.unique(arrays["users"][train_index])
        expected_same = np.flatnonzero(
            (arrays["dataset"] == 0)
            & (arrays["unseen"] == PARTITION_TRAIN)
            & (arrays["sport"] == sport_code)
            & (arrays["evaluation"] == 1)
            & np.isin(arrays["users"], train_users)
        )
        expected_joint = np.flatnonzero(
            (arrays["dataset"] == 0)
            & (arrays["unseen"] == PARTITION_TEST)
            & (arrays["sport"] == sport_code)
            & (arrays["evaluation"] == 1)
        )
        same_exact = np.array_equal(
            row_index[:same_user_rows], expected_same
        )
        joint_exact = np.array_equal(row_index[same_user_rows:], expected_joint)
        require(same_exact, f"{family}: same-user mapping mismatch")
        require(joint_exact, f"{family}: joint mapping mismatch")

        regime_slices = {
            f"unseen_sport__{family}": slice(0, same_user_rows),
            f"joint_user_sport__{family}": slice(
                same_user_rows, len(row_index)
            ),
        }
        users_by_regime: dict[str, int] = {}
        for regime, selected_slice in regime_slices.items():
            selected_rows = row_index[selected_slice]
            predictions = history_quantiles[selected_slice]
            selected_users = np.asarray(arrays["users"][selected_rows])
            selected_sessions = np.asarray(arrays["sessions"][selected_rows])
            users_by_regime[regime] = int(len(np.unique(selected_users)))

            for horizon_position, horizon in enumerate(HORIZONS):
                target = np.asarray(
                    arrays["targets"][selected_rows, horizon_position]
                )
                median = predictions[:, horizon_position, 3]
                absolute_error = np.abs(median - target)
                user_mae = hierarchical_user_mae(
                    absolute_error, selected_users, selected_sessions
                )
                mae = float(user_mae.mean())
                n_users = int(len(user_mae))
                n_sessions = int(len(np.unique(selected_sessions)))
                n_origins = int(len(selected_rows))

                reference_row = reference[
                    (reference["held_sport_family"] == family)
                    & (reference["regime"] == regime)
                    & (reference["mode"] == "history_informed")
                    & (reference["horizon_seconds"] == horizon)
                ]
                require(
                    len(reference_row) == 1,
                    f"{family}/{regime}/{horizon}: reference row count",
                )
                reference_record = reference_row.iloc[0]
                maximum_reference_delta = max(
                    maximum_reference_delta,
                    abs(mae - float(reference_record["mae_bpm"])),
                )
                require(
                    int(reference_record["users"]) == n_users
                    and int(reference_record["sessions"]) == n_sessions
                    and int(reference_record["origins"]) == n_origins,
                    f"{family}/{regime}/{horizon}: support mismatch",
                )

                bootstrap = bootstrap_user_mean(
                    user_mae.to_numpy(dtype=np.float64),
                    args.bootstrap_replicates,
                    stable_seed(
                        args.seed, f"{family}|{regime}|{horizon}|mae"
                    ),
                )
                ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
                output_rows.append(
                    {
                        "analysis_version": ANALYSIS_VERSION,
                        "source_model_version": SOURCE_MODEL_VERSION,
                        "held_sport_code": sport_code,
                        "held_sport_family": family,
                        "regime": regime,
                        "mode": "history_informed",
                        "horizon_seconds": horizon,
                        "mae_bpm": mae,
                        "mae_ci_low_bpm": float(ci_low),
                        "mae_ci_high_bpm": float(ci_high),
                        "users": n_users,
                        "sessions": n_sessions,
                        "origins": n_origins,
                        "support_caution_lt25_users": n_users < 25,
                        "bootstrap_replicates": args.bootstrap_replicates,
                        "bootstrap_unit": "user",
                        "aggregation": (
                            "origin-within-session, session-within-user, "
                            "equal-user mean"
                        ),
                    }
                )

        family_audits[family] = {
            "held_sport_code": sport_code,
            "prediction_file": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_rows": int(len(row_index)),
            "same_user_rows_saved": same_user_rows,
            "same_user_rows_reconstructed": int(len(expected_same)),
            "joint_rows_saved": int(len(row_index) - same_user_rows),
            "joint_rows_reconstructed": int(len(expected_joint)),
            "same_user_mapping_exact": same_exact,
            "joint_mapping_exact": joint_exact,
            "users_by_regime": users_by_regime,
            "all_assertions_pass": True,
        }

    require(len(output_rows) == 30, f"expected 30 rows, got {len(output_rows)}")
    tolerance = 5e-6
    require(
        maximum_reference_delta <= tolerance,
        f"reference MAE mismatch: {maximum_reference_delta}",
    )
    output_frame = pd.DataFrame(output_rows)
    output_checks = {
        "duplicate_analysis_keys": int(
            output_frame.duplicated(
                ["held_sport_family", "regime", "horizon_seconds"]
            ).sum()
        ),
        "nonfinite_mae_values": int(
            (
                ~np.isfinite(
                    output_frame[
                        ["mae_bpm", "mae_ci_low_bpm", "mae_ci_high_bpm"]
                    ].to_numpy()
                )
            ).sum()
        ),
        "nonpositive_mae_failures": int((output_frame["mae_bpm"] <= 0).sum()),
        "point_outside_ci": int(
            (
                (output_frame["mae_bpm"] < output_frame["mae_ci_low_bpm"])
                | (output_frame["mae_bpm"] > output_frame["mae_ci_high_bpm"])
            ).sum()
        ),
    }
    require(
        not any(output_checks.values()), f"output validation failed: {output_checks}"
    )

    atomic_csv(args.output_csv, output_rows)
    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "source_model_version": SOURCE_MODEL_VERSION,
        "intended_use": (
            "Supplementary Table S5a history-informed held-sport and joint-shift "
            "hierarchical MAE confidence intervals without retraining"
        ),
        "array_rows": total_rows,
        "array_row_counts": row_counts,
        "families": family_audits,
        "aggregation": (
            "absolute errors averaged within session, sessions averaged within "
            "user, users equally weighted"
        ),
        "bootstrap": {
            "unit": "user",
            "replicates": args.bootstrap_replicates,
            "base_seed": args.seed,
            "interval": "two-sided 95% percentile",
        },
        "reference_csv": str(args.reference_csv),
        "maximum_absolute_reference_mae_delta": maximum_reference_delta,
        "reference_tolerance": tolerance,
        "output_csv": str(args.output_csv),
        "output_csv_sha256": sha256_file(args.output_csv),
        "output_rows": len(output_rows),
        "output_checks": output_checks,
        "limitations": [
            (
                "Intervals quantify between-user sampling variation conditional "
                "on each existing v0.12.0 checkpoint and do not include training "
                "or seed variability."
            ),
            (
                "Indoor/virtual cycling, walking/hiking, and strength/cross-"
                "training joint intersections contain fewer than 25 users and "
                "remain cautionary."
            ),
            (
                "Separate leave-one-family-out models were fitted for each sport; "
                "between-family differences are descriptive, not causal sport "
                "effects."
            ),
        ],
        "all_assertions_pass": True,
    }
    atomic_json(args.audit_json, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap history-informed held-sport and joint-shift hierarchical "
            "MAE at the user level without retraining."
        )
    )
    parser.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("outputs/predictions"),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("outputs/results/sport_shift_point_v0_12_0.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/results/sport_shift_mae_bootstrap_v0_19_0.csv"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/audit/sport_shift_mae_bootstrap_v0_19_0.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("at least 1,000 bootstrap replicates are required")
    audit = analyze(args)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
