"""Summarize causally available prior-workout counts in final test regimes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.19.0"


def summarize_regime(
    regime: str,
    session_ids: np.ndarray,
    session_prior_count: np.ndarray,
    session_users: pd.Series,
) -> dict:
    unique_sessions = np.unique(session_ids).astype(np.int64, copy=False)
    counts = session_prior_count[unique_sessions].astype(np.int64, copy=False)
    users = session_users.loc[unique_sessions].to_numpy(dtype=np.int64)
    frame = pd.DataFrame({"user_index": users, "prior_count": counts})
    user_summary = frame.groupby("user_index", sort=False).prior_count.agg(
        sessions="size",
        any_history=lambda values: bool((values > 0).any()),
        all_history=lambda values: bool((values > 0).all()),
        median_prior="median",
    )
    q1, median, q3 = np.quantile(counts, [0.25, 0.5, 0.75])
    user_q1, user_median, user_q3 = np.quantile(
        user_summary.median_prior.to_numpy(), [0.25, 0.5, 0.75]
    )
    return {
        "analysis_version": VERSION,
        "regime": regime,
        "users": int(frame.user_index.nunique()),
        "sessions": int(len(unique_sessions)),
        "sessions_with_history": int((counts > 0).sum()),
        "sessions_without_history": int((counts == 0).sum()),
        "sessions_with_history_percent": float((counts > 0).mean() * 100),
        "prior_count_session_q1": float(q1),
        "prior_count_session_median": float(median),
        "prior_count_session_q3": float(q3),
        "sessions_prior_0": int((counts == 0).sum()),
        "sessions_prior_1_4": int(((counts >= 1) & (counts <= 4)).sum()),
        "sessions_prior_5_9": int(((counts >= 5) & (counts <= 9)).sum()),
        "sessions_prior_10_plus": int((counts >= 10).sum()),
        "users_with_any_history": int(user_summary.any_history.sum()),
        "users_with_history_in_all_test_sessions": int(user_summary.all_history.sum()),
        "user_median_prior_count_q1": float(user_q1),
        "user_median_prior_count_median": float(user_median),
        "user_median_prior_count_q3": float(user_q3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--array-dir",
        type=Path,
        default=ROOT / "outputs" / "features" / "model_arrays_v0_6_0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "results" / "history_availability_v0_19_0.csv",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "outputs" / "audit" / "history_availability_v0_19_0.json",
    )
    args = parser.parse_args()

    array_dir = args.array_dir
    session_index = np.load(array_dir / "session_index.npy", mmap_mode="r")
    session_prior_count = np.load(array_dir / "session_prior_count.npy", mmap_mode="r")
    evaluation_origin = np.load(array_dir / "evaluation_origin.npy", mmap_mode="r")
    dataset_code = np.load(array_dir / "dataset_code.npy", mmap_mode="r")
    unseen_partition = np.load(array_dir / "unseen_user_partition.npy", mmap_mode="r")
    temporal_partition = np.load(array_dir / "temporal_partition_strict.npy", mmap_mode="r")
    sessions = pd.read_csv(array_dir / "sessions.csv", usecols=["session_index", "user_index"])
    session_users = sessions.set_index("session_index").user_index

    row_masks = {
        "strict_temporal_test":
            (dataset_code == 0) & (evaluation_origin == 1) & (temporal_partition == 4),
        "unseen_user_test":
            (dataset_code == 0) & (evaluation_origin == 1) & (unseen_partition == 4),
    }
    rows = [
        summarize_regime(name, session_index[mask], session_prior_count, session_users)
        for name, mask in row_masks.items()
    ]
    frame = pd.DataFrame(rows)

    assertions = {
        "strict_temporal_sessions": int(frame.loc[frame.regime == "strict_temporal_test", "sessions"].iloc[0]) == 16012,
        "strict_temporal_users": int(frame.loc[frame.regime == "strict_temporal_test", "users"].iloc[0]) == 948,
        "unseen_user_sessions": int(frame.loc[frame.regime == "unseen_user_test", "sessions"].iloc[0]) == 15026,
        "unseen_user_users": int(frame.loc[frame.regime == "unseen_user_test", "users"].iloc[0]) == 105,
        "category_totals_equal_sessions": all(
            int(row.sessions_prior_0 + row.sessions_prior_1_4 + row.sessions_prior_5_9 + row.sessions_prior_10_plus)
            == int(row.sessions)
            for row in frame.itertuples()
        ),
        "history_counts_nonnegative": bool((session_prior_count >= 0).all()),
        "no_identifiers_in_output": not bool(
            {"user_index", "session_index", "user_id", "session_key"} & set(frame.columns)
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, lineterminator="\n")
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": VERSION,
        "source_array_version": "0.6.0",
        "selection": "Endomondo 300-s evaluation origins in final test partitions",
        "aggregation": "unique evaluation sessions; user summaries give users equal rows",
        "rows": rows,
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
    }
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    print(args.audit)
    return 0 if audit["all_assertions_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

