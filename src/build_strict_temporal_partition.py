from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "0.13.0"
ORDERED_CODES = (1, 2, 3, 4)
EXCLUDED_CODE = 5


def exclude_cross_boundary_overlaps(
    sessions: pd.DataFrame, session_codes: np.ndarray
) -> tuple[np.ndarray, set[int]]:
    strict = session_codes.copy()
    excluded: set[int] = set()
    eligible = sessions[
        (sessions["dataset"] == "Endomondo")
        & sessions["session_index"].map(lambda item: strict[int(item)] in ORDERED_CODES)
    ].copy()
    for _, user_sessions in eligible.groupby("user_index", sort=False):
        indices = user_sessions["session_index"].to_numpy(dtype=np.int64)
        starts = user_sessions["session_start_time"].to_numpy(dtype=np.float64)
        ends = user_sessions["session_end_time"].to_numpy(dtype=np.float64)
        changed = True
        while changed:
            changed = False
            codes = strict[indices]
            mark = np.zeros(len(indices), dtype=bool)
            for boundary in (1, 2, 3):
                left = (codes >= 1) & (codes <= boundary)
                right = (codes > boundary) & (codes <= 4)
                if not left.any() or not right.any():
                    continue
                first_right_start = starts[right].min()
                last_left_end = ends[left].max()
                mark |= left & (ends >= first_right_start)
                mark |= right & (starts <= last_left_end)
            if mark.any():
                newly_excluded = indices[mark & (strict[indices] != EXCLUDED_CODE)]
                if len(newly_excluded):
                    strict[newly_excluded] = EXCLUDED_CODE
                    excluded.update(int(item) for item in newly_excluded)
                    changed = True
    return strict, excluded


def validate_order(sessions: pd.DataFrame, strict_codes: np.ndarray) -> int:
    failures = 0
    endomondo = sessions[sessions["dataset"] == "Endomondo"]
    for _, user_sessions in endomondo.groupby("user_index", sort=False):
        indices = user_sessions["session_index"].to_numpy(dtype=np.int64)
        starts = user_sessions["session_start_time"].to_numpy(dtype=np.float64)
        ends = user_sessions["session_end_time"].to_numpy(dtype=np.float64)
        codes = strict_codes[indices]
        for boundary in (1, 2, 3):
            left = (codes >= 1) & (codes <= boundary)
            right = (codes > boundary) & (codes <= 4)
            if left.any() and right.any() and ends[left].max() >= starts[right].min():
                failures += 1
    return failures


def build(args: argparse.Namespace) -> dict[str, object]:
    original = np.load(args.array_dir / "temporal_partition.npy", mmap_mode="r")
    dataset = np.load(args.array_dir / "dataset_code.npy", mmap_mode="r")
    row_sessions = np.load(args.array_dir / "session_index.npy", mmap_mode="r")
    sessions = pd.read_csv(
        args.array_dir / "sessions.csv", dtype={"session_key": str}, low_memory=False
    )
    n_sessions = int(sessions["session_index"].max()) + 1
    session_codes = np.zeros(n_sessions, dtype=np.uint8)
    consistency_failures = 0
    for code in (1, 2, 3, 4, 5):
        selected_sessions = np.unique(row_sessions[(dataset == 0) & (original == code)])
        consistency_failures += int(np.count_nonzero(session_codes[selected_sessions]))
        session_codes[selected_sessions] = code
    if consistency_failures:
        raise AssertionError(f"sessions assigned to multiple temporal partitions: {consistency_failures}")

    strict_session_codes, excluded_sessions = exclude_cross_boundary_overlaps(
        sessions, session_codes
    )
    order_failures = validate_order(sessions, strict_session_codes)
    if order_failures:
        raise AssertionError(f"strict temporal ordering failures: {order_failures}")

    strict_rows = np.asarray(original).copy()
    endomondo_rows = np.flatnonzero(dataset == 0)
    strict_rows[endomondo_rows] = strict_session_codes[row_sessions[endomondo_rows]]
    changed_rows = int(np.count_nonzero(strict_rows != original))
    if np.any(strict_rows[dataset == 1] != original[dataset == 1]):
        raise AssertionError("external rows were modified")

    counts_before = {
        str(code): int(np.count_nonzero((dataset == 0) & (original == code)))
        for code in (1, 2, 3, 4, 5)
    }
    counts_after = {
        str(code): int(np.count_nonzero((dataset == 0) & (strict_rows == code)))
        for code in (1, 2, 3, 4, 5)
    }
    excluded_users = sessions.loc[
        sessions["session_index"].isin(excluded_sessions), "user_index"
    ].nunique()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, strict_rows)
    temporary.replace(args.output)
    payload: dict[str, object] = {
        "version": VERSION,
        "source_partition": str(args.array_dir / "temporal_partition.npy"),
        "rule": "exclude all sessions touching or overlapping a boundary between ordered temporal partitions",
        "ordered_partition_codes": list(ORDERED_CODES),
        "excluded_partition_code": EXCLUDED_CODE,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "excluded_boundary_sessions": len(excluded_sessions),
        "excluded_boundary_users": int(excluded_users),
        "changed_rows": changed_rows,
        "session_partition_consistency_failures": consistency_failures,
        "strict_order_failures": order_failures,
        "external_rows_modified": 0,
        "all_assertions_pass": True,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--array-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--audit", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args())))
