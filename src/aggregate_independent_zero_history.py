from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "0.23.0"
HORIZONS = (60, 180, 300)
PARTITION_TEST = 4
EXTERNAL_FROZEN = 1
CANONICAL_AGGREGATION = (
    "origin-within-session, session-within-user, equal-user mean"
)
CONTRASTS = (
    (
        "mixed_history_minus_mixed_zero",
        "mixed_history",
        "mixed_zero",
        "mixed history-informed minus mixed forced-zero-history",
    ),
    (
        "mixed_zero_minus_independent_zero",
        "mixed_zero",
        "independent_zero",
        "mixed forced-zero-history minus independently trained zero-history",
    ),
    (
        "mixed_history_minus_independent_zero",
        "mixed_history",
        "independent_zero",
        "mixed history-informed minus independently trained zero-history",
    ),
)
MODEL_POSITION = {
    "mixed_history": 0,
    "mixed_zero": 1,
    "independent_zero": 2,
}


class AggregationBlockedError(RuntimeError):
    """Raised when strict final aggregation is not permitted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"refusing to emit an empty final table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    require(array.size == 1, "metadata must be scalar")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def nested_training_value(payload: dict[str, Any], key: str) -> Any:
    values: list[Any] = []
    if key in payload:
        values.append(payload[key])
    training = payload.get("training")
    if isinstance(training, dict) and key in training:
        values.append(training[key])
    require(bool(values), f"missing machine-readable field {key}")
    first = values[0]
    require(all(value == first for value in values[1:]), f"conflicting {key}")
    return first


@dataclass(frozen=True)
class ExpectedJob:
    seed: int
    protocol: str
    directory: Path

    @property
    def job_id(self) -> str:
        return f"seed_{self.seed}/{self.protocol}"


def validate_frozen_config(config: dict[str, Any]) -> tuple[list[int], list[str]]:
    require(str(config.get("analysis_version")) == ANALYSIS_VERSION, "analysis version")
    seeds = [int(seed) for seed in config.get("seeds", [])]
    protocols = [str(value) for value in config.get("protocols", [])]
    require(seeds and len(seeds) == len(set(seeds)), "invalid seed declaration")
    require(
        set(protocols) == {"unseen_user", "strict_temporal"}
        and len(protocols) == 2,
        "protocols must be unseen_user and strict_temporal",
    )
    training = config.get("training")
    require(isinstance(training, dict), "missing training configuration")
    require(training.get("training_history_mode") == "always_zero", "training mode")
    expected_budget = {
        "epoch_samples": 500000,
        "batch_size": 2048,
        "inference_batch_size": 4096,
        "max_epochs": 40,
        "patience": 4,
        "learning_rate": 0.001,
    }
    for key, expected in expected_budget.items():
        require(training.get(key) == expected, f"frozen training budget mismatch: {key}")
    reporting = config.get("reporting")
    require(isinstance(reporting, dict), "missing reporting configuration")
    require(reporting.get("primary_horizon_seconds") == 300, "primary horizon")
    require(
        reporting.get("secondary_horizons_seconds") == [60, 180],
        "secondary horizons",
    )
    require(reporting.get("user_bootstrap_replicates") == 10000, "bootstrap count")
    require(
        reporting.get("seeds_are_independent_participants") is False,
        "seeds may not be declared independent participants",
    )
    return seeds, protocols


def build_expected_jobs(
    config: dict[str, Any], root: Path
) -> list[ExpectedJob]:
    seeds, protocols = validate_frozen_config(config)
    return [
        ExpectedJob(seed, protocol, root / f"seed_{seed}" / protocol)
        for seed in seeds
        for protocol in protocols
    ]


def required_exact_paths(job: ExpectedJob) -> list[Path]:
    common = [
        job.directory / "point_metrics.csv",
        job.directory / "interval_metrics.csv",
        job.directory / "audit.json",
        job.directory / "m" / "resolved_config.json",
    ]
    if job.protocol == "unseen_user":
        return common + [
            job.directory / "test_predictions.npz",
            job.directory / "freeze_record.json",
            job.directory / "external_predictions.npz",
            job.directory / "external_point_metrics.csv",
            job.directory / "external_interval_metrics.csv",
            job.directory / "external_audit.json",
        ]
    return common + [job.directory / "predictions.npz"]


def inspect_job_files(job: ExpectedJob, root: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "job_id": job.job_id,
        "seed": job.seed,
        "protocol": job.protocol,
        "status": "pending",
        "files": [],
        "errors": [],
    }
    if not job.directory.exists():
        return record
    errors: list[str] = []
    files: list[dict[str, object]] = []
    for path in required_exact_paths(job):
        if not path.is_file():
            errors.append(f"missing {path.relative_to(job.directory).as_posix()}")
            continue
        if path.stat().st_size <= 0:
            errors.append(f"empty {path.relative_to(job.directory).as_posix()}")
            continue
        files.append(
            {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    glob_rules = {
        "checkpoint": ("m/*.pt", 1, 1),
        "normalization": ("m/*normalization*.json", 1, None),
        "conformal": ("m/*conformal*.json", 1, None),
    }
    for label, (pattern, minimum, maximum) in glob_rules.items():
        matches = sorted(path for path in job.directory.glob(pattern) if path.is_file())
        if len(matches) < minimum or (maximum is not None and len(matches) > maximum):
            errors.append(f"{label} glob {pattern} matched {len(matches)} files")
        for path in matches:
            if path.stat().st_size <= 0:
                errors.append(f"empty {path.relative_to(job.directory).as_posix()}")
            else:
                files.append(
                    {
                        "path": path.resolve().relative_to(root.resolve()).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    record["files"] = sorted(files, key=lambda value: str(value["path"]))
    record["errors"] = errors
    record["status"] = "incomplete" if errors else "present"
    return record


def validate_machine_metadata(
    payload: dict[str, Any],
    *,
    seed: int,
    protocol: str,
    selection_metric: str,
    training: dict[str, Any],
    label: str,
) -> None:
    require(int(payload.get("seed")) == seed, f"{label}: seed mismatch")
    require(payload.get("protocol") == protocol, f"{label}: protocol mismatch")
    require(
        nested_training_value(payload, "training_history_mode") == "always_zero",
        f"{label}: mode mismatch",
    )
    require(
        nested_training_value(payload, "selection_metric") == selection_metric,
        f"{label}: selection metric mismatch",
    )
    for key in (
        "epoch_samples",
        "batch_size",
        "inference_batch_size",
        "max_epochs",
        "patience",
        "learning_rate",
    ):
        require(
            nested_training_value(payload, key) == training[key],
            f"{label}: frozen budget mismatch: {key}",
        )


def validate_point_csv(
    path: Path,
    *,
    seed: int,
    protocol: str,
    regime: str,
) -> dict[int, dict[str, object]]:
    frame = pd.read_csv(path)
    required = {
        "regime",
        "mode",
        "horizon_seconds",
        "mae_bpm",
        "users",
        "sessions",
        "origins",
    }
    require(required.issubset(frame.columns), f"{path}: point columns")
    selected = frame.loc[frame["regime"].astype(str) == regime].copy()
    require(not selected.empty, f"{path}: missing regime {regime}")
    require(set(selected["mode"].astype(str)) == {"zero_history"}, f"{path}: mode")
    if "seed" in selected:
        require(set(selected["seed"].astype(int)) == {seed}, f"{path}: seed")
    if "protocol" in selected:
        require(set(selected["protocol"].astype(str)) == {protocol}, f"{path}: protocol")
    rows: dict[int, dict[str, object]] = {}
    for horizon in HORIZONS:
        subset = selected.loc[selected["horizon_seconds"].astype(int) == horizon]
        require(len(subset) == 1, f"{path}: horizon {horizon} must occur once")
        row = subset.iloc[0].to_dict()
        for column in ("mae_bpm", "users", "sessions", "origins"):
            require(np.isfinite(float(row[column])), f"{path}: non-finite {column}")
        require(float(row["mae_bpm"]) >= 0, f"{path}: negative MAE")
        require(
            int(row["users"]) > 0
            and int(row["sessions"]) > 0
            and int(row["origins"]) > 0,
            f"{path}: empty support",
        )
        rows[horizon] = row
    return rows


def validate_interval_csv(
    path: Path,
    *,
    seed: int,
    protocol: str,
    regime: str,
) -> None:
    frame = pd.read_csv(path)
    required = {
        "regime",
        "mode",
        "horizon_seconds",
        "picp",
        "mean_interval_width_bpm",
    }
    require(required.issubset(frame.columns), f"{path}: interval columns")
    selected = frame.loc[frame["regime"].astype(str) == regime].copy()
    require(not selected.empty, f"{path}: missing interval regime {regime}")
    require(set(selected["mode"].astype(str)) == {"zero_history"}, f"{path}: mode")
    if "seed" in selected:
        require(set(selected["seed"].astype(int)) == {seed}, f"{path}: seed")
    if "protocol" in selected:
        require(set(selected["protocol"].astype(str)) == {protocol}, f"{path}: protocol")
    require(
        set(selected["horizon_seconds"].astype(int)) == set(HORIZONS),
        f"{path}: horizon support",
    )
    for column in ("picp", "mean_interval_width_bpm"):
        values = pd.to_numeric(selected[column], errors="coerce").to_numpy(float)
        require(np.isfinite(values).all(), f"{path}: non-finite {column}")
    require(
        selected["picp"].astype(float).between(0.0, 1.0).all(),
        f"{path}: PICP outside [0,1]",
    )
    require(
        (selected["mean_interval_width_bpm"].astype(float) >= 0).all(),
        f"{path}: negative interval width",
    )
    key = ["horizon_seconds"]
    for optional in ("nominal_coverage", "calibrated"):
        if optional in selected:
            key.append(optional)
    require(not selected.duplicated(key).any(), f"{path}: duplicate interval key")


def validate_job_semantics(job: ExpectedJob, config: dict[str, Any]) -> None:
    training = dict(config["training"])
    selection_metric = str(training["selection_metric"])
    resolved = load_json(job.directory / "m" / "resolved_config.json", "resolved config")
    audit = load_json(job.directory / "audit.json", "audit")
    validate_machine_metadata(
        resolved,
        seed=job.seed,
        protocol=job.protocol,
        selection_metric=selection_metric,
        training=training,
        label=f"{job.job_id} resolved config",
    )
    validate_machine_metadata(
        audit,
        seed=job.seed,
        protocol=job.protocol,
        selection_metric=selection_metric,
        training=training,
        label=f"{job.job_id} audit",
    )
    require(audit.get("all_assertions_pass") is True, f"{job.job_id}: audit failed")
    if job.protocol == "unseen_user":
        validate_point_csv(
            job.directory / "point_metrics.csv",
            seed=job.seed,
            protocol=job.protocol,
            regime="unseen_user_test",
        )
        validate_interval_csv(
            job.directory / "interval_metrics.csv",
            seed=job.seed,
            protocol=job.protocol,
            regime="unseen_user_test",
        )
        freeze = load_json(job.directory / "freeze_record.json", "freeze record")
        require(int(freeze.get("seed")) == job.seed, f"{job.job_id}: freeze seed")
        require(freeze.get("protocol") == job.protocol, f"{job.job_id}: freeze protocol")
        require(
            freeze.get("status") == "frozen_before_external_inference",
            f"{job.job_id}: not frozen before external inference",
        )
        require(
            freeze.get("external_outcomes_used_for_selection") is False,
            f"{job.job_id}: external outcomes used for selection",
        )
        require(
            freeze.get("external_adaptation_or_recalibration_allowed") is False,
            f"{job.job_id}: external adaptation allowed",
        )
        checkpoints = sorted(job.directory.glob("m/*.pt"))
        require(len(checkpoints) == 1, f"{job.job_id}: checkpoint cardinality")
        frozen_paths = {
            "checkpoint": checkpoints[0],
            "thresholds": job.directory
            / "m"
            / "conformal_thresholds_v0_11_0.json",
            "input_normalization": job.directory
            / "m"
            / "normalization_unseen_user_train.json",
            "history_normalization": job.directory
            / "m"
            / "history_normalization_unseen_user_train.json",
            "resolved_config": job.directory / "m" / "resolved_config.json",
        }
        artifact_manifest = freeze.get("artifacts")
        require(
            isinstance(artifact_manifest, dict),
            f"{job.job_id}: freeze artifact manifest",
        )
        for artifact_name, artifact_path in frozen_paths.items():
            entry = artifact_manifest.get(artifact_name)
            require(
                isinstance(entry, dict),
                f"{job.job_id}: freeze missing {artifact_name}",
            )
            require(
                Path(str(entry.get("path"))).resolve() == artifact_path.resolve(),
                f"{job.job_id}: freeze path mismatch {artifact_name}",
            )
            require(
                entry.get("sha256") == sha256_file(artifact_path),
                f"{job.job_id}: freeze hash mismatch {artifact_name}",
            )
        external_audit = load_json(job.directory / "external_audit.json", "external audit")
        require(int(external_audit.get("seed")) == job.seed, "external audit seed")
        require(
            external_audit.get("protocol")
            == "frozen GoldenCheetah external inference after development freeze",
            "external audit protocol",
        )
        require(external_audit.get("all_assertions_pass") is True, "external audit failed")
        require(
            external_audit.get("freeze_record_sha256")
            == sha256_file(job.directory / "freeze_record.json"),
            "external audit freeze-record hash",
        )
        require(
            external_audit.get("checkpoint_sha256")
            == sha256_file(frozen_paths["checkpoint"]),
            "external audit checkpoint hash",
        )
        require(
            external_audit.get("thresholds_sha256")
            == sha256_file(frozen_paths["thresholds"]),
            "external audit thresholds hash",
        )
        require(
            external_audit.get("external_adaptation_or_recalibration") is False,
            "external adaptation or recalibration occurred",
        )
        validate_point_csv(
            job.directory / "external_point_metrics.csv",
            seed=job.seed,
            protocol=job.protocol,
            regime="goldencheetah_frozen_external",
        )
        validate_interval_csv(
            job.directory / "external_interval_metrics.csv",
            seed=job.seed,
            protocol=job.protocol,
            regime="goldencheetah_frozen_external",
        )
    else:
        validate_point_csv(
            job.directory / "point_metrics.csv",
            seed=job.seed,
            protocol=job.protocol,
            regime="within_user_temporal_test",
        )
        validate_interval_csv(
            job.directory / "interval_metrics.csv",
            seed=job.seed,
            protocol=job.protocol,
            regime="within_user_temporal_test",
        )


def validate_q1_completion(mixed_root: Path, expected_seeds: list[int]) -> list[Path]:
    audit_path = mixed_root / "aggregation" / "aggregation_audit_v0_22_0.json"
    progress_path = mixed_root / "aggregation" / "progress_manifest.json"
    require(audit_path.is_file(), "Q1 final aggregation audit is missing")
    require(progress_path.is_file(), "Q1 final aggregation progress is missing")
    audit = load_json(audit_path, "Q1 aggregation audit")
    progress = load_json(progress_path, "Q1 progress")
    require(audit.get("status") == "complete", "Q1 aggregation is not complete")
    require(audit.get("all_assertions_pass") is True, "Q1 aggregation audit failed")
    require(progress.get("status") == "complete", "Q1 progress is not complete")
    require(progress.get("final_artifacts_emitted") is True, "Q1 final artifacts absent")
    seed_groups = audit.get("expected_seed_groups")
    require(isinstance(seed_groups, dict), "Q1 audit has no seed groups")
    require(
        [int(value) for value in seed_groups.get("primary_models", [])]
        == expected_seeds,
        "Q1 primary seeds do not match independent ablation seeds",
    )
    return [audit_path, progress_path]


def load_arrays(array_dir: Path) -> dict[str, np.ndarray]:
    names = {
        "targets": "targets.npy",
        "users": "user_index.npy",
        "sessions": "session_index.npy",
        "dataset": "dataset_code.npy",
        "evaluation": "evaluation_origin.npy",
        "unseen": "unseen_user_partition.npy",
        "temporal": "temporal_partition_strict.npy",
        "external": "primary_external_partition.npy",
    }
    arrays = {
        key: np.load(array_dir / filename, mmap_mode="r")
        for key, filename in names.items()
    }
    row_lengths = {key: len(value) for key, value in arrays.items()}
    require(len(set(row_lengths.values())) == 1, f"source array lengths: {row_lengths}")
    require(arrays["targets"].shape == (len(arrays["dataset"]), 3), "target shape")
    return arrays


def expected_rows(arrays: dict[str, np.ndarray], evaluation: str) -> np.ndarray:
    if evaluation == "unseen_user_test":
        mask = (
            (arrays["dataset"] == 0)
            & (arrays["unseen"] == PARTITION_TEST)
            & (arrays["evaluation"] == 1)
        )
    elif evaluation == "strict_temporal_test":
        mask = (
            (arrays["dataset"] == 0)
            & (arrays["temporal"] == PARTITION_TEST)
            & (arrays["evaluation"] == 1)
        )
    elif evaluation == "frozen_external":
        mask = (arrays["dataset"] == 1) & (arrays["external"] == EXTERNAL_FROZEN)
    else:
        raise KeyError(evaluation)
    rows = np.flatnonzero(mask).astype(np.int64)
    require(len(rows) > 0, f"empty expected support: {evaluation}")
    return rows


def load_prediction_npz(
    path: Path,
    keys: Iterable[str],
    *,
    seed: int | None = None,
    protocol: str | None = None,
    mode: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as source:
        require("row_index" in source.files, f"{path}: missing row_index")
        for key in keys:
            require(key in source.files, f"{path}: missing {key}")
        rows = np.asarray(source["row_index"], dtype=np.int64)
        predictions = {key: np.asarray(source[key]) for key in keys}
        if "seed" in source.files and seed is not None:
            require(int(np.asarray(source["seed"]).reshape(-1)[0]) == seed, f"{path}: seed")
        if "protocol" in source.files and protocol is not None:
            require(scalar_text(source["protocol"]) == protocol, f"{path}: protocol")
        if "mode" in source.files and mode is not None:
            require(scalar_text(source["mode"]) == mode, f"{path}: mode")
        if "horizon_seconds" in source.files:
            require(
                np.array_equal(np.asarray(source["horizon_seconds"]).astype(int), HORIZONS),
                f"{path}: horizons",
            )
    require(rows.ndim == 1 and len(rows) > 0, f"{path}: invalid row_index")
    require(len(np.unique(rows)) == len(rows), f"{path}: duplicate row_index")
    for key, prediction in predictions.items():
        require(prediction.shape == (len(rows), 3, 7), f"{path}: {key} shape")
        require(np.isfinite(prediction).all(), f"{path}: {key} non-finite")
        require(
            ((prediction >= 30.0) & (prediction <= 240.0)).all(),
            f"{path}: {key} outside physiological bounds",
        )
        require(
            (np.diff(prediction, axis=2) >= -1e-6).all(),
            f"{path}: {key} quantile crossing",
        )
    return rows, predictions


def subset_mixed_development(
    rows: np.ndarray,
    predictions: dict[str, np.ndarray],
    expected: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    require(rows.min() >= 0 and rows.max() < len(arrays["dataset"]), "mixed row bounds")
    mask = (
        (arrays["dataset"][rows] == 0)
        & (arrays["unseen"][rows] == PARTITION_TEST)
        & (arrays["evaluation"][rows] == 1)
    )
    selected_rows = rows[mask]
    require(np.array_equal(selected_rows, expected), "mixed unseen row_index mismatch")
    return selected_rows, {key: value[mask] for key, value in predictions.items()}


def hierarchy_user_means(
    losses: np.ndarray, users: np.ndarray, sessions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    losses = np.asarray(losses, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    sessions = np.asarray(sessions, dtype=np.int64)
    require(losses.ndim == 2 and len(losses) == len(users), "loss shape")
    require(len(users) == len(sessions) and len(users) > 0, "support shape")
    require(np.isfinite(losses).all(), "non-finite loss")
    order = np.lexsort((sessions, users))
    sorted_users = users[order]
    sorted_sessions = sessions[order]
    sorted_losses = losses[order]
    session_start = np.r_[
        0,
        np.flatnonzero(
            (np.diff(sorted_users) != 0) | (np.diff(sorted_sessions) != 0)
        )
        + 1,
    ]
    session_count = np.diff(np.r_[session_start, len(sorted_users)])
    session_means = np.add.reduceat(sorted_losses, session_start, axis=0)
    session_means = session_means / session_count[:, None]
    session_users = sorted_users[session_start]
    user_start = np.r_[0, np.flatnonzero(np.diff(session_users) != 0) + 1]
    user_count = np.diff(np.r_[user_start, len(session_users)])
    user_means = np.add.reduceat(session_means, user_start, axis=0)
    user_means = user_means / user_count[:, None]
    user_ids = session_users[user_start]
    require(len(np.unique(user_ids)) == len(user_ids), "duplicate user aggregate")
    return user_ids, user_means


def assert_reported_point_matches(
    rows: dict[int, dict[str, object]],
    expected_mae: np.ndarray,
    *,
    users: int,
    sessions: int,
    origins: int,
    label: str,
) -> None:
    for position, horizon in enumerate(HORIZONS):
        row = rows[horizon]
        require(
            np.isclose(
                float(row["mae_bpm"]),
                float(expected_mae[position]),
                rtol=1e-5,
                atol=5e-5,
            ),
            f"{label}: reported MAE mismatch at {horizon}",
        )
        require(int(row["users"]) == users, f"{label}: user support")
        require(int(row["sessions"]) == sessions, f"{label}: session support")
        require(int(row["origins"]) == origins, f"{label}: origin support")


def mixed_point_rows(
    path: Path, regime: str, mode: str
) -> dict[int, dict[str, object]]:
    frame = pd.read_csv(path)
    required = {
        "regime",
        "mode",
        "horizon_seconds",
        "mae_bpm",
        "users",
        "sessions",
        "origins",
    }
    require(required.issubset(frame.columns), f"{path}: mixed point columns")
    selected = frame.loc[
        (frame["regime"].astype(str) == regime)
        & (frame["mode"].astype(str) == mode)
    ]
    rows: dict[int, dict[str, object]] = {}
    for horizon in HORIZONS:
        subset = selected.loc[selected["horizon_seconds"].astype(int) == horizon]
        require(len(subset) == 1, f"{path}: mixed {mode} horizon {horizon}")
        rows[horizon] = subset.iloc[0].to_dict()
    return rows


def validate_mixed_audits(seed_root: Path, seed: int) -> list[Path]:
    paths = [
        seed_root / "unseen_main" / "development_audit.json",
        seed_root / "unseen_main" / "external_audit.json",
        seed_root / "temporal_main" / "audit.json",
    ]
    for path in paths:
        require(path.is_file(), f"missing mixed audit {path}")
        audit = load_json(path, "mixed audit")
        require(int(audit.get("seed")) == seed, f"{path}: seed mismatch")
        require(audit.get("all_assertions_pass") is True, f"{path}: audit failed")
    external = load_json(paths[1], "mixed external audit")
    require(
        external.get("external_adaptation_or_recalibration") is False,
        "mixed external adaptation/recalibration",
    )
    return paths


def percentile_user_bootstrap(
    values: np.ndarray, *, replicates: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0, "empty user bootstrap input")
    require(np.isfinite(values).all(), "non-finite user bootstrap input")
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        positions = generator.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[positions].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "per_seed": output_dir / "strategy_contrasts_per_seed_v0_23_0.csv",
        "seed_summary": output_dir / "strategy_contrast_seed_summary_v0_23_0.csv",
        "user_seed_mean": output_dir / "strategy_contrast_user_seed_mean_v0_23_0.csv",
        "user_bootstrap": output_dir / "strategy_contrast_user_bootstrap_v0_23_0.csv",
        "audit": output_dir / "aggregation_audit_v0_23_0.json",
        "progress": output_dir / "progress_manifest.json",
    }


def status_payload(
    *,
    config: dict[str, Any],
    jobs: list[dict[str, object]],
    status: str,
    errors: list[str],
    paths: dict[str, Path],
) -> dict[str, object]:
    counts = {
        name: sum(record["status"] == name for record in jobs)
        for name in ("pending", "incomplete", "invalid", "complete")
    }
    stale = [
        path.name
        for key, path in paths.items()
        if key not in {"audit", "progress"} and path.exists()
    ]
    return {
        "generated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "expected_jobs": len(jobs),
        "status_counts": counts,
        "jobs": jobs,
        "errors": errors,
        "final_artifacts_emitted": False,
        "stale_final_artifacts_present": stale,
        "fail_closed": True,
        "note": (
            "No final CSV is current unless status is complete and "
            "final_artifacts_emitted is true."
        ),
        "declared_seeds": config.get("seeds", []),
        "seeds_treated_as_independent_participants": False,
    }


def fail_or_return(
    payload: dict[str, object],
    paths: dict[str, Path],
    allow_incomplete: bool,
) -> dict[str, object]:
    atomic_json(paths["audit"], payload)
    atomic_json(paths["progress"], payload)
    if allow_incomplete:
        return payload
    raise AggregationBlockedError(
        f"independent zero-history aggregation blocked: {payload['status']}"
    )


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    independent_root = Path(args.independent_root)
    mixed_root = Path(args.mixed_root)
    array_dir = Path(args.array_dir)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    allow_incomplete = bool(getattr(args, "allow_incomplete", False))
    bootstrap_seed = int(getattr(args, "bootstrap_seed", 20260722))
    paths = output_paths(output_dir)
    config = load_json(config_path, "independent ablation configuration")
    seeds, _ = validate_frozen_config(config)
    expected_jobs = build_expected_jobs(config, independent_root)
    job_records = [inspect_job_files(job, independent_root) for job in expected_jobs]
    for job, record in zip(expected_jobs, job_records, strict=True):
        if record["status"] != "present":
            continue
        try:
            validate_job_semantics(job, config)
        except Exception as error:  # audit state must survive malformed inputs
            record["status"] = "invalid"
            record["errors"] = [f"{type(error).__name__}: {error}"]
        else:
            record["status"] = "complete"
    if any(record["status"] != "complete" for record in job_records):
        if any(record["status"] == "invalid" for record in job_records):
            status = "invalid"
        elif any(record["status"] == "incomplete" for record in job_records):
            status = "incomplete"
        else:
            status = "pending"
        payload = status_payload(
            config=config,
            jobs=job_records,
            status=status,
            errors=[],
            paths=paths,
        )
        return fail_or_return(payload, paths, allow_incomplete)

    input_hashes: dict[str, str] = {str(config_path): sha256_file(config_path)}
    for record in job_records:
        for file_record in record["files"]:  # type: ignore[index]
            input_hashes[str(file_record["path"])] = str(file_record["sha256"])
    per_seed_rows: list[dict[str, object]] = []
    user_seed_values: dict[
        tuple[str, str, int, str, int], list[tuple[int, float]]
    ] = {}
    support_reference: dict[tuple[str, str], tuple[str, str]] = {}
    calculation_errors: list[str] = []
    try:
        q1_paths = validate_q1_completion(mixed_root, seeds)
        for path in q1_paths:
            input_hashes[str(path)] = sha256_file(path)
        arrays = load_arrays(array_dir)
        for filename in (
            "targets.npy",
            "user_index.npy",
            "session_index.npy",
            "dataset_code.npy",
            "evaluation_origin.npy",
            "unseen_user_partition.npy",
            "temporal_partition_strict.npy",
            "primary_external_partition.npy",
        ):
            input_hashes[str(array_dir / filename)] = sha256_file(array_dir / filename)
        evaluation_specs = (
            ("unseen_user", "internal_test", "unseen_user_test"),
            ("unseen_user", "frozen_external", "frozen_external"),
            ("strict_temporal", "internal_test", "strict_temporal_test"),
        )
        expected_by_evaluation = {
            key: expected_rows(arrays, key)
            for key in ("unseen_user_test", "strict_temporal_test", "frozen_external")
        }
        for seed in seeds:
            seed_root = mixed_root / f"seed_{seed}"
            for path in validate_mixed_audits(seed_root, seed):
                input_hashes[str(path)] = sha256_file(path)
            for protocol, evaluation_label, expected_key in evaluation_specs:
                expected = expected_by_evaluation[expected_key]
                independent_dir = independent_root / f"seed_{seed}" / protocol
                if protocol == "unseen_user" and evaluation_label == "internal_test":
                    mixed_prediction_path = seed_root / "unseen_main" / "development_predictions.npz"
                    mixed_point_path = seed_root / "unseen_main" / "development_point_metrics.csv"
                    independent_prediction_path = independent_dir / "test_predictions.npz"
                    independent_point_path = independent_dir / "point_metrics.csv"
                    regime = "unseen_user_test"
                elif protocol == "unseen_user":
                    mixed_prediction_path = seed_root / "unseen_main" / "external_predictions.npz"
                    mixed_point_path = seed_root / "unseen_main" / "external_point_metrics.csv"
                    independent_prediction_path = independent_dir / "external_predictions.npz"
                    independent_point_path = independent_dir / "external_point_metrics.csv"
                    regime = "goldencheetah_frozen_external"
                else:
                    mixed_prediction_path = seed_root / "temporal_main" / "predictions.npz"
                    mixed_point_path = seed_root / "temporal_main" / "point_metrics.csv"
                    independent_prediction_path = independent_dir / "predictions.npz"
                    independent_point_path = independent_dir / "point_metrics.csv"
                    regime = "within_user_temporal_test"
                for path in (
                    mixed_prediction_path,
                    mixed_point_path,
                    independent_prediction_path,
                    independent_point_path,
                ):
                    require(path.is_file(), f"missing comparison input {path}")
                    input_hashes[str(path)] = sha256_file(path)
                mixed_keys = (
                    ("zero_history_quantiles",)
                    if evaluation_label == "frozen_external"
                    else ("history_quantiles", "zero_history_quantiles")
                )
                mixed_rows, mixed_predictions = load_prediction_npz(
                    mixed_prediction_path, mixed_keys
                )
                if expected_key == "unseen_user_test":
                    mixed_rows, mixed_predictions = subset_mixed_development(
                        mixed_rows, mixed_predictions, expected, arrays
                    )
                else:
                    require(
                        np.array_equal(mixed_rows, expected),
                        f"{seed} {protocol} {evaluation_label}: mixed row_index mismatch",
                    )
                independent_rows, independent_predictions = load_prediction_npz(
                    independent_prediction_path,
                    ("zero_history_quantiles",),
                    seed=seed,
                    protocol=protocol,
                    mode="zero_history",
                )
                require(
                    np.array_equal(independent_rows, expected),
                    f"{seed} {protocol} {evaluation_label}: independent row_index mismatch",
                )
                require(
                    np.array_equal(mixed_rows, independent_rows),
                    f"{seed} {protocol} {evaluation_label}: unmatched row_index",
                )
                targets = np.asarray(arrays["targets"][expected], dtype=np.float64)
                users = np.asarray(arrays["users"][expected], dtype=np.int64)
                sessions = np.asarray(arrays["sessions"][expected], dtype=np.int64)
                require(np.isfinite(targets).all(), "non-finite target")
                mixed_zero = mixed_predictions["zero_history_quantiles"][:, :, 3]
                if evaluation_label == "frozen_external":
                    mixed_history = mixed_zero
                else:
                    mixed_history = mixed_predictions["history_quantiles"][:, :, 3]
                independent_zero = independent_predictions["zero_history_quantiles"][:, :, 3]
                prediction_stack = np.stack(
                    [mixed_history, mixed_zero, independent_zero], axis=1
                )
                losses = np.abs(prediction_stack - targets[:, None, :]).reshape(
                    len(expected), 9
                )
                user_ids, user_model_mae_flat = hierarchy_user_means(
                    losses, users, sessions
                )
                user_model_mae = user_model_mae_flat.reshape(len(user_ids), 3, 3)
                model_mae = user_model_mae.mean(axis=0)
                n_users = len(user_ids)
                n_sessions = len(np.unique(sessions))
                n_origins = len(expected)
                support_key = (protocol, evaluation_label)
                support_value = (sha256_array(expected), sha256_array(user_ids))
                if support_key in support_reference:
                    require(
                        support_reference[support_key] == support_value,
                        f"{support_key}: support changed across seeds",
                    )
                else:
                    support_reference[support_key] = support_value
                independent_reported = validate_point_csv(
                    independent_point_path,
                    seed=seed,
                    protocol=protocol,
                    regime=regime,
                )
                assert_reported_point_matches(
                    independent_reported,
                    model_mae[MODEL_POSITION["independent_zero"]],
                    users=n_users,
                    sessions=n_sessions,
                    origins=n_origins,
                    label=f"{seed} {protocol} independent zero",
                )
                mixed_zero_reported = mixed_point_rows(
                    mixed_point_path, regime, "zero_history"
                )
                assert_reported_point_matches(
                    mixed_zero_reported,
                    model_mae[MODEL_POSITION["mixed_zero"]],
                    users=n_users,
                    sessions=n_sessions,
                    origins=n_origins,
                    label=f"{seed} {protocol} mixed zero",
                )
                if evaluation_label != "frozen_external":
                    mixed_history_reported = mixed_point_rows(
                        mixed_point_path, regime, "history_informed"
                    )
                    assert_reported_point_matches(
                        mixed_history_reported,
                        model_mae[MODEL_POSITION["mixed_history"]],
                        users=n_users,
                        sessions=n_sessions,
                        origins=n_origins,
                        label=f"{seed} {protocol} mixed history",
                    )
                available_contrasts = (
                    (CONTRASTS[1],)
                    if evaluation_label == "frozen_external"
                    else CONTRASTS
                )
                for contrast_id, left, right, definition in available_contrasts:
                    left_position = MODEL_POSITION[left]
                    right_position = MODEL_POSITION[right]
                    for horizon_position, horizon in enumerate(HORIZONS):
                        user_difference = (
                            user_model_mae[:, left_position, horizon_position]
                            - user_model_mae[:, right_position, horizon_position]
                        )
                        difference = float(user_difference.mean())
                        per_seed_rows.append(
                            {
                                "seed": seed,
                                "protocol": protocol,
                                "evaluation": evaluation_label,
                                "horizon_seconds": horizon,
                                "contrast": contrast_id,
                                "definition": definition,
                                "mixed_history_mae_bpm": (
                                    None
                                    if evaluation_label == "frozen_external"
                                    else float(model_mae[0, horizon_position])
                                ),
                                "mixed_zero_mae_bpm": float(model_mae[1, horizon_position]),
                                "independent_zero_mae_bpm": float(model_mae[2, horizon_position]),
                                "difference_bpm": difference,
                                "negative_favors_left": True,
                                "users": n_users,
                                "sessions": n_sessions,
                                "origins": n_origins,
                                "aggregation": CANONICAL_AGGREGATION,
                                "row_index_sha256": support_value[0],
                                "user_support_sha256": support_value[1],
                            }
                        )
                        for user_id, value in zip(
                            user_ids, user_difference, strict=True
                        ):
                            key = (
                                protocol,
                                evaluation_label,
                                horizon,
                                contrast_id,
                                int(user_id),
                            )
                            user_seed_values.setdefault(key, []).append(
                                (seed, float(value))
                            )
        require(bool(per_seed_rows), "no paired contrasts computed")
    except Exception as error:
        calculation_errors.append(f"{type(error).__name__}: {error}")

    if calculation_errors:
        payload = status_payload(
            config=config,
            jobs=job_records,
            status="invalid",
            errors=calculation_errors,
            paths=paths,
        )
        payload["input_sha256"] = dict(sorted(input_hashes.items()))
        return fail_or_return(payload, paths, allow_incomplete)

    expected_seed_set = set(seeds)
    for key, values in user_seed_values.items():
        observed = [seed for seed, _ in values]
        require(len(observed) == len(set(observed)), f"duplicate user-seed value: {key}")
        require(set(observed) == expected_seed_set, f"incomplete user seed support: {key}")
    per_seed_rows.sort(
        key=lambda row: (
            str(row["protocol"]),
            str(row["evaluation"]),
            int(row["horizon_seconds"]),
            str(row["contrast"]),
            int(row["seed"]),
        )
    )
    per_seed_frame = pd.DataFrame(per_seed_rows)
    grouping = ["protocol", "evaluation", "horizon_seconds", "contrast", "definition"]
    seed_summary_rows: list[dict[str, object]] = []
    for keys, frame in per_seed_frame.groupby(grouping, sort=True):
        values = frame["difference_bpm"].to_numpy(float)
        observed_seeds = frame["seed"].astype(int).tolist()
        require(set(observed_seeds) == expected_seed_set, f"incomplete seed contrast: {keys}")
        seed_summary_rows.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                "n_matched_seeds": len(values),
                "difference_median_bpm": float(np.median(values)),
                "difference_minimum_bpm": float(np.min(values)),
                "difference_maximum_bpm": float(np.max(values)),
                "seed_inferential_test": False,
                "seed_confidence_interval": False,
            }
        )
    user_seed_mean_rows: list[dict[str, object]] = []
    bootstrap_groups: dict[tuple[str, str, int, str], list[float]] = {}
    for key, seed_values in sorted(user_seed_values.items()):
        protocol, evaluation, horizon, contrast_id, user_id = key
        ordered = sorted(seed_values)
        mean_difference = float(np.mean([value for _, value in ordered]))
        user_seed_mean_rows.append(
            {
                "protocol": protocol,
                "evaluation": evaluation,
                "horizon_seconds": horizon,
                "contrast": contrast_id,
                "user_index": user_id,
                "n_matched_seeds": len(ordered),
                "mean_paired_loss_difference_bpm": mean_difference,
            }
        )
        bootstrap_groups.setdefault(
            (protocol, evaluation, horizon, contrast_id), []
        ).append(mean_difference)
    bootstrap_rows: list[dict[str, object]] = []
    replicates = int(config["reporting"]["user_bootstrap_replicates"])
    for key, values in sorted(bootstrap_groups.items()):
        protocol, evaluation, horizon, contrast_id = key
        stable_offset = int.from_bytes(
            hashlib.sha256("|".join(map(str, key)).encode("utf-8")).digest()[:4],
            "little",
        )
        estimate, low, high = percentile_user_bootstrap(
            np.asarray(values),
            replicates=replicates,
            seed=(bootstrap_seed + stable_offset) % (2**32),
        )
        bootstrap_rows.append(
            {
                "protocol": protocol,
                "evaluation": evaluation,
                "horizon_seconds": horizon,
                "contrast": contrast_id,
                "estimate_bpm": estimate,
                "percentile_95_ci_low_bpm": low,
                "percentile_95_ci_high_bpm": high,
                "users": len(values),
                "matched_seeds_per_user": len(seeds),
                "bootstrap_replicates": replicates,
                "bootstrap_unit": "user",
                "seed_resampling": False,
                "uncertainty_interpretation": (
                    "sampling uncertainty conditional on the declared seed set"
                ),
            }
        )
    final_tables = {
        "per_seed": per_seed_rows,
        "seed_summary": seed_summary_rows,
        "user_seed_mean": user_seed_mean_rows,
        "user_bootstrap": bootstrap_rows,
    }
    for key, rows in final_tables.items():
        atomic_csv(paths[key], rows)
    audit: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "configuration_sha256": sha256_file(config_path),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "expected_jobs": len(expected_jobs),
        "completed_jobs": len(expected_jobs),
        "jobs": job_records,
        "declared_seeds": seeds,
        "input_sha256": dict(sorted(input_hashes.items())),
        "output_sha256": {
            paths[key].name: sha256_file(paths[key]) for key in final_tables
        },
        "table_rows": {key: len(rows) for key, rows in final_tables.items()},
        "contrast_definitions": {
            contrast_id: definition
            for contrast_id, _, _, definition in CONTRASTS
        },
        "contrast_availability": {
            "internal_unseen_user_and_strict_temporal": [
                contrast_id for contrast_id, _, _, _ in CONTRASTS
            ],
            "frozen_external": ["mixed_zero_minus_independent_zero"],
            "reason": (
                "GoldenCheetah has no cross-source identity linkage, so the "
                "protocol exposes only the no-history state."
            ),
        },
        "statistical_policy": {
            "seeds_treated_as_independent_participants": False,
            "hypothesis_tests_over_seeds": False,
            "confidence_intervals_over_seeds": False,
            "across_seed_statistics": ["median", "minimum", "maximum"],
            "user_bootstrap_replicates": replicates,
            "user_bootstrap_order": (
                "paired user loss difference within each matched seed; mean across "
                "matched seeds for each user; percentile bootstrap over users"
            ),
            "uncertainty_interpretation": (
                "sampling uncertainty conditional on the declared seed set"
            ),
        },
        "assertions": {
            "all_ten_declared_training_jobs_complete": len(expected_jobs) == 10,
            "all_job_audits_and_resolved_configs_match_seed_protocol_mode": True,
            "all_frozen_budgets_and_selection_metrics_match": True,
            "all_prediction_row_indices_match_exactly": True,
            "all_horizons_present": True,
            "all_predictions_targets_and_metrics_finite": True,
            "all_user_support_consistent_across_seeds": True,
            "reported_hierarchical_mae_independently_recomputed": True,
            "external_inference_frozen_without_adaptation_or_recalibration": True,
            "user_bootstrap_uses_seed_averaged_user_differences": True,
            "seeds_not_used_as_independent_samples": True,
        },
        "all_assertions_pass": True,
        "final_artifacts_emitted": True,
    }
    require(
        len(expected_jobs) == 10,
        "frozen production configuration must resolve to ten training jobs",
    )
    atomic_json(paths["audit"], audit)
    progress = {
        "generated_at_utc": utc_now(),
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "expected_jobs": len(expected_jobs),
        "status_counts": {
            "pending": 0,
            "incomplete": 0,
            "invalid": 0,
            "complete": len(expected_jobs),
        },
        "final_artifacts_emitted": True,
        "final_artifacts": {
            path.name: sha256_file(path)
            for key, path in paths.items()
            if key != "progress" and path.exists()
        },
    }
    atomic_json(paths["progress"], progress)
    return progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly aggregate the independently trained zero-history ablation. "
            "Optimization seeds are summarized descriptively and are never treated "
            "as independent study participants."
        )
    )
    parser.add_argument(
        "--independent-root",
        type=Path,
        default=Path("outputs/independent_zero_history_v0_23_0"),
    )
    parser.add_argument(
        "--mixed-root",
        type=Path,
        default=Path("outputs/q1_multiseed_v0_21_0"),
    )
    parser.add_argument(
        "--array-dir",
        type=Path,
        default=Path("outputs/features/model_arrays_v0_6_0"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/independent_zero_history_v0_23_0.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/independent_zero_history_v0_23_0/aggregation"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write audit/progress state and exit successfully without final CSVs.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    result = aggregate(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "expected_jobs": result["expected_jobs"],
                "status_counts": result["status_counts"],
                "final_artifacts_emitted": result["final_artifacts_emitted"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
