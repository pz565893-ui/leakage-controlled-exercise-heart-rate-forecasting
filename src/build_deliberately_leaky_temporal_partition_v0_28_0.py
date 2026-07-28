from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from run_q1_multiseed_queue import atomic_json, sha256_file, utc_now


ANALYSIS_VERSION = "0.28.0"
PARTITION_EXCLUDED = 0
PARTITION_TRAIN = 1
PARTITION_VALIDATION = 2
PARTITION_CALIBRATION = 3
PARTITION_TEST = 4
HORIZONS_SECONDS = np.asarray([60, 180, 300], dtype=np.int64)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def index_sha256(index: np.ndarray) -> str:
    canonical = np.asarray(index, dtype="<i8")
    require(canonical.ndim == 1, "row index must be one-dimensional")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def stable_window_assignments(
    sessions: np.ndarray,
    origin_times: np.ndarray,
    *,
    namespace: str,
    train_threshold: float,
    validation_threshold: float,
) -> np.ndarray:
    """Assign rows by a platform-stable SHA-256 hash, independently of model seed."""
    sessions = np.asarray(sessions, dtype=np.int64)
    origin_times = np.asarray(origin_times, dtype=np.float64)
    require(sessions.shape == origin_times.shape, "session/time shape mismatch")
    require(
        0.0 < train_threshold < validation_threshold < 1.0,
        "invalid split thresholds",
    )
    rounded = np.rint(origin_times)
    require(
        bool(np.all(np.isfinite(origin_times)))
        and bool(np.all(np.abs(origin_times - rounded) <= 1e-6)),
        "origin times must be finite integer seconds",
    )
    output = np.empty(len(sessions), dtype=np.uint8)
    denominator = float(2**64)
    for position, (session, origin) in enumerate(
        zip(sessions.tolist(), rounded.astype(np.int64).tolist())
    ):
        key = f"{namespace}|{session}|{origin}".encode("utf-8")
        unit = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / denominator
        if unit < train_threshold:
            output[position] = PARTITION_TRAIN
        elif unit < validation_threshold:
            output[position] = PARTITION_VALIDATION
        else:
            output[position] = PARTITION_CALIBRATION
    return output


def pairwise_overlap_counts(
    values: np.ndarray, partition: np.ndarray
) -> dict[str, int]:
    labels = {
        "train": PARTITION_TRAIN,
        "validation": PARTITION_VALIDATION,
        "calibration": PARTITION_CALIBRATION,
        "test": PARTITION_TEST,
    }
    sets = {
        name: np.unique(values[partition == code]) for name, code in labels.items()
    }
    output: dict[str, int] = {}
    names = list(labels)
    for left_position, left in enumerate(names):
        for right in names[left_position + 1 :]:
            output[f"{left}_{right}"] = int(
                np.intersect1d(sets[left], sets[right], assume_unique=True).size
            )
    return output


def _group_times(
    row_index: np.ndarray, sessions: np.ndarray, origin_times: np.ndarray
) -> dict[int, np.ndarray]:
    if len(row_index) == 0:
        return {}
    selected_sessions = np.asarray(sessions[row_index], dtype=np.int64)
    selected_times = np.rint(
        np.asarray(origin_times[row_index], dtype=np.float64)
    ).astype(np.int64)
    order = np.lexsort((selected_times, selected_sessions))
    selected_sessions = selected_sessions[order]
    selected_times = selected_times[order]
    boundaries = np.flatnonzero(np.r_[True, selected_sessions[1:] != selected_sessions[:-1], True])
    output: dict[int, np.ndarray] = {}
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        output[int(selected_sessions[start])] = selected_times[start:end]
    return output


def proximity_and_collision_audit(
    test_index: np.ndarray,
    contaminated_train_index: np.ndarray,
    sessions: np.ndarray,
    origin_times: np.ndarray,
) -> dict[str, object]:
    """Audit the overlap mechanism without inspecting HR targets or predictions."""
    test_groups = _group_times(test_index, sessions, origin_times)
    train_groups = _group_times(contaminated_train_index, sessions, origin_times)
    nearest_distances: list[np.ndarray] = []
    context_overlaps: list[np.ndarray] = []
    target_collision_by_horizon = np.zeros(3, dtype=np.int64)
    origins_with_any_target_collision = 0
    origins_with_train_session = 0

    for session, test_times in test_groups.items():
        train_times = train_groups.get(session)
        if train_times is None or len(train_times) == 0:
            continue
        origins_with_train_session += len(test_times)
        insertion = np.searchsorted(train_times, test_times)
        left_position = np.clip(insertion - 1, 0, len(train_times) - 1)
        right_position = np.clip(insertion, 0, len(train_times) - 1)
        left_distance = np.abs(test_times - train_times[left_position])
        right_distance = np.abs(test_times - train_times[right_position])
        nearest = np.minimum(left_distance, right_distance)
        nearest_distances.append(nearest)
        context_overlaps.append(np.maximum(0, 300 - nearest))

        train_target_times = np.unique(
            (train_times[:, None] + HORIZONS_SECONDS[None, :]).reshape(-1)
        )
        session_any = np.zeros(len(test_times), dtype=bool)
        for position, horizon in enumerate(HORIZONS_SECONDS):
            collided = np.isin(test_times + int(horizon), train_target_times)
            target_collision_by_horizon[position] += int(collided.sum())
            session_any |= collided
        origins_with_any_target_collision += int(session_any.sum())

    if nearest_distances:
        distance = np.concatenate(nearest_distances).astype(np.float64)
        overlap = np.concatenate(context_overlaps).astype(np.float64)
    else:
        distance = np.asarray([], dtype=np.float64)
        overlap = np.asarray([], dtype=np.float64)
    total = int(len(test_index))

    def count_rate(mask: np.ndarray) -> dict[str, float | int]:
        count = int(mask.sum())
        return {"count": count, "rate_of_all_test_origins": count / total if total else 0.0}

    distance_summary: dict[str, float | int | None] = {
        "origins_with_contaminated_train_session": origins_with_train_session,
        "origins_without_contaminated_train_session": total - origins_with_train_session,
        "minimum": float(distance.min()) if len(distance) else None,
        "median": float(np.median(distance)) if len(distance) else None,
        "p95": float(np.quantile(distance, 0.95)) if len(distance) else None,
        "maximum": float(distance.max()) if len(distance) else None,
    }
    return {
        "nearest_contaminated_train_origin_distance_seconds": distance_summary,
        "test_origins_with_nearest_distance_le_60_seconds": count_rate(distance <= 60),
        "test_origins_with_nearest_distance_le_120_seconds": count_rate(distance <= 120),
        "test_origins_with_nearest_distance_le_300_seconds": count_rate(distance <= 300),
        "nearest_context_overlap_seconds": {
            "definition": "max(0, 300 - absolute origin-time difference)",
            "median": float(np.median(overlap)) if len(overlap) else None,
            "p05": float(np.quantile(overlap, 0.05)) if len(overlap) else None,
            "maximum": float(overlap.max()) if len(overlap) else None,
            "positive_overlap": count_rate(overlap > 0),
            "overlap_ge_240_seconds": count_rate(overlap >= 240),
        },
        "exact_target_timestamp_collisions": {
            "definition": "test target time equals any +60/+180/+300 target time of a contaminated training origin in the same session",
            "by_test_horizon": {
                str(int(horizon)): int(target_collision_by_horizon[position])
                for position, horizon in enumerate(HORIZONS_SECONDS)
            },
            "collided_target_slots": int(target_collision_by_horizon.sum()),
            "total_test_target_slots": int(total * len(HORIZONS_SECONDS)),
            "test_origins_with_any_collision": origins_with_any_target_collision,
            "test_origins_with_any_collision_rate": (
                origins_with_any_target_collision / total if total else 0.0
            ),
        },
    }


def construct_partition(
    *,
    dataset: np.ndarray,
    evaluation: np.ndarray,
    strict_partition: np.ndarray,
    users: np.ndarray,
    sessions: np.ndarray,
    origin_times: np.ndarray,
    namespace: str,
    train_threshold: float,
    validation_threshold: float,
) -> tuple[np.ndarray, dict[str, object]]:
    arrays = [dataset, evaluation, strict_partition, users, sessions, origin_times]
    require(len({len(item) for item in arrays}) == 1, "array-length mismatch")
    endomondo = dataset == 0
    clean_train = endomondo & (strict_partition == PARTITION_TRAIN)
    clean_validation = (
        endomondo
        & (strict_partition == PARTITION_VALIDATION)
        & (evaluation == 1)
    )
    clean_calibration = (
        endomondo
        & (strict_partition == PARTITION_CALIBRATION)
        & (evaluation == 1)
    )
    fixed_test = (
        endomondo & (strict_partition == PARTITION_TEST) & (evaluation == 1)
    )
    test_index = np.flatnonzero(fixed_test)
    require(len(test_index) > 0, "fixed strict-temporal test is empty")
    test_sessions = np.unique(sessions[test_index])
    contamination_candidate = (
        endomondo
        & (strict_partition == PARTITION_TEST)
        & (evaluation == 0)
        & np.isin(sessions, test_sessions)
    )
    contamination_index = np.flatnonzero(contamination_candidate)
    require(len(contamination_index) > 0, "contamination candidate pool is empty")
    assignments = stable_window_assignments(
        sessions[contamination_index],
        origin_times[contamination_index],
        namespace=namespace,
        train_threshold=train_threshold,
        validation_threshold=validation_threshold,
    )

    partition = np.zeros(len(dataset), dtype=np.uint8)
    partition[clean_train] = PARTITION_TRAIN
    partition[clean_validation] = PARTITION_VALIDATION
    partition[clean_calibration] = PARTITION_CALIBRATION
    partition[fixed_test] = PARTITION_TEST
    partition[contamination_index] = assignments
    require(
        np.array_equal(np.flatnonzero(partition == PARTITION_TEST), test_index),
        "fixed test row alignment changed",
    )
    require(
        not np.intersect1d(test_index, contamination_index).size,
        "exact test rows entered contamination pool",
    )

    contamination_train_index = contamination_index[
        assignments == PARTITION_TRAIN
    ]
    row_overlap = {
        key: value
        for key, value in pairwise_overlap_counts(
            np.arange(len(partition), dtype=np.int64), partition
        ).items()
    }
    session_overlap = pairwise_overlap_counts(sessions, partition)
    user_overlap = pairwise_overlap_counts(users, partition)
    counts = {
        "clean_train_rows": int(clean_train.sum()),
        "clean_validation_rows": int(clean_validation.sum()),
        "clean_calibration_rows": int(clean_calibration.sum()),
        "contamination_candidates": int(len(contamination_index)),
        "contamination_train_rows": int(
            np.count_nonzero(assignments == PARTITION_TRAIN)
        ),
        "contamination_validation_rows": int(
            np.count_nonzero(assignments == PARTITION_VALIDATION)
        ),
        "contamination_calibration_rows": int(
            np.count_nonzero(assignments == PARTITION_CALIBRATION)
        ),
        "leaky_train_rows": int(np.count_nonzero(partition == PARTITION_TRAIN)),
        "leaky_validation_rows": int(
            np.count_nonzero(partition == PARTITION_VALIDATION)
        ),
        "leaky_calibration_rows": int(
            np.count_nonzero(partition == PARTITION_CALIBRATION)
        ),
        "fixed_test_rows": int(len(test_index)),
    }
    audit: dict[str, object] = {
        "analysis_version": ANALYSIS_VERSION,
        "valid_for_generalization": False,
        "leaderboard_eligible": False,
        "invalid_reason": "same-session overlapping windows deliberately contaminate training, validation, and calibration",
        "hash_assignment": {
            "namespace": namespace,
            "algorithm": "SHA-256 first unsigned 64 bits",
            "train_threshold": train_threshold,
            "validation_threshold": validation_threshold,
            "calibration_threshold": 1.0,
            "model_seed_changes_split": False,
        },
        "counts": counts,
        "fixed_test": {
            "rows": int(len(test_index)),
            "sessions": int(len(test_sessions)),
            "users": int(np.unique(users[test_index]).size),
            "row_index_sha256_int64_little_endian": index_sha256(test_index),
            "exactly_matches_strict_temporal_evaluation_test": True,
        },
        "exact_row_overlap_counts": row_overlap,
        "session_overlap_counts": session_overlap,
        "user_overlap_counts": user_overlap,
        "test_sessions_with_contaminated_training_rows": int(
            np.intersect1d(
                test_sessions,
                np.unique(sessions[contamination_train_index]),
                assume_unique=True,
            ).size
        ),
        "proximity_and_collision": proximity_and_collision_audit(
            test_index,
            contamination_train_index,
            sessions,
            origin_times,
        ),
    }
    audit["all_assertions_pass"] = (
        not any(row_overlap.values())
        and audit["test_sessions_with_contaminated_training_rows"] > 0
        and session_overlap["train_test"] > 0
        and audit["fixed_test"]["exactly_matches_strict_temporal_evaluation_test"]
    )
    require(bool(audit["all_assertions_pass"]), "leaky partition assertions failed")
    return partition, audit


def validate_configuration(config: dict[str, object]) -> None:
    require(config.get("analysis_version") == ANALYSIS_VERSION, "config version")
    require(
        config.get("status")
        == "retrospective-negative-control-configuration-locked-before-GPU-execution",
        "config status",
    )
    require(config.get("valid_for_generalization") is False, "validity flag")
    require(
        config.get("required_acknowledgement_flag")
        == "--acknowledge-invalid-generalization",
        "acknowledgement flag",
    )
    split = config.get("split")
    require(isinstance(split, dict), "missing split config")
    require(split.get("model_seed_changes_split") is False, "split must be fixed")
    require(split.get("train_threshold") == 0.7, "train threshold")
    require(split.get("validation_threshold") == 0.85, "validation threshold")


def build(args: argparse.Namespace) -> dict[str, object]:
    config = json.loads(args.configuration.read_text(encoding="utf-8"))
    validate_configuration(config)
    array_dir = args.array_dir
    strict_path = args.strict_temporal_partition
    dataset = np.load(array_dir / "dataset_code.npy", mmap_mode="r")
    evaluation = np.load(array_dir / "evaluation_origin.npy", mmap_mode="r")
    users = np.load(array_dir / "user_index.npy", mmap_mode="r")
    sessions = np.load(array_dir / "session_index.npy", mmap_mode="r")
    origin_times = np.load(array_dir / "origin_time.npy", mmap_mode="r")
    strict_partition = np.load(strict_path, mmap_mode="r")
    split = config["split"]
    partition, payload = construct_partition(
        dataset=dataset,
        evaluation=evaluation,
        strict_partition=strict_partition,
        users=users,
        sessions=sessions,
        origin_times=origin_times,
        namespace=str(split["namespace"]),
        train_threshold=float(split["train_threshold"]),
        validation_threshold=float(split["validation_threshold"]),
    )
    expected_counts = config.get("expected_partition_counts")
    require(isinstance(expected_counts, dict), "missing expected counts")
    count_mismatches = {
        key: {"expected": int(value), "observed": int(payload["counts"].get(key, -1))}
        for key, value in expected_counts.items()
        if int(payload["counts"].get(key, -1)) != int(value)
    }
    fixed = config.get("fixed_test")
    require(isinstance(fixed, dict), "missing fixed-test config")
    fixed_hash = payload["fixed_test"]["row_index_sha256_int64_little_endian"]
    require(
        fixed_hash == fixed.get("row_index_sha256_int64_little_endian"),
        "fixed-test row-index hash mismatch",
    )
    require(not count_mismatches, f"partition count mismatch: {count_mismatches}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, partition)
    temporary.replace(args.output)
    payload.update(
        {
            "generated_at_utc": utc_now(),
            "configuration": str(args.configuration.resolve()),
            "configuration_sha256": sha256_file(args.configuration),
            "array_dir": str(array_dir.resolve()),
            "strict_temporal_partition": str(strict_path.resolve()),
            "strict_temporal_partition_sha256": sha256_file(strict_path),
            "output": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
            "expected_count_mismatches": count_mismatches,
        }
    )
    payload["all_assertions_pass"] = bool(payload["all_assertions_pass"]) and not count_mismatches
    atomic_json(args.audit, payload)
    require(bool(payload["all_assertions_pass"]), "partition audit failed")
    return payload


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(
        description="Build the deliberately invalid v0.28 same-session-window partition."
    )
    result.add_argument(
        "--configuration",
        type=Path,
        default=root / "configs" / "leaky_negative_control_v0_28_0.json",
    )
    result.add_argument(
        "--array-dir",
        type=Path,
        default=root / "outputs" / "features" / "model_arrays_v0_6_0",
    )
    result.add_argument(
        "--strict-temporal-partition",
        type=Path,
        default=(
            root
            / "outputs"
            / "features"
            / "model_arrays_v0_6_0"
            / "temporal_partition_strict.npy"
        ),
    )
    result.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "outputs"
            / "features"
            / "model_arrays_v0_6_0"
            / "temporal_partition_deliberately_leaky_v0_28_0.npy"
        ),
    )
    result.add_argument(
        "--audit",
        type=Path,
        default=root / "outputs" / "audit" / "deliberately_leaky_temporal_partition_v0_28_0.json",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), ensure_ascii=False))
