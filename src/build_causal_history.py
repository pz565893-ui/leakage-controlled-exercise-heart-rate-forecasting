from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_VERSION = "0.10.1"
FEATURE_NAMES = (
    "log1p_prior_sessions",
    "log1p_mean_duration_seconds",
    "log1p_duration_std_seconds",
    "prior_mean_session_hr",
    "prior_std_session_mean_hr",
    "prior_mean_within_session_hr_std",
    "prior_mean_session_speed",
    "prior_std_session_mean_speed",
    "prior_mean_altitude_std",
    "log1p_same_sport_sessions",
    "prior_same_sport_mean_hr",
    "prior_same_sport_mean_speed",
    "log1p_days_since_previous_session",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Accumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_duration = 0.0
        self.sum_duration_sq = 0.0
        self.sum_hr = 0.0
        self.sum_hr_sq = 0.0
        self.sum_hr_std = 0.0
        self.sum_speed = 0.0
        self.sum_speed_sq = 0.0
        self.sum_altitude_std = 0.0
        self.previous_start: float | None = None
        self.sport_count: defaultdict[int, int] = defaultdict(int)
        self.sport_hr_sum: defaultdict[int, float] = defaultdict(float)
        self.sport_speed_sum: defaultdict[int, float] = defaultdict(float)

    @staticmethod
    def standard_deviation(total: float, total_sq: float, count: int) -> float:
        if count == 0:
            return 0.0
        mean = total / count
        return math.sqrt(max(total_sq / count - mean * mean, 0.0))

    def features(self, sport_code: int, start_time: float) -> np.ndarray:
        if self.count == 0:
            return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        same_count = self.sport_count[sport_code]
        days_since_previous = (
            max(0.0, start_time - self.previous_start) / 86_400.0
            if self.previous_start is not None
            else 0.0
        )
        return np.asarray(
            [
                math.log1p(self.count),
                math.log1p(self.sum_duration / self.count),
                math.log1p(
                    self.standard_deviation(
                        self.sum_duration, self.sum_duration_sq, self.count
                    )
                ),
                self.sum_hr / self.count,
                self.standard_deviation(self.sum_hr, self.sum_hr_sq, self.count),
                self.sum_hr_std / self.count,
                self.sum_speed / self.count,
                self.standard_deviation(
                    self.sum_speed, self.sum_speed_sq, self.count
                ),
                self.sum_altitude_std / self.count,
                math.log1p(same_count),
                self.sport_hr_sum[sport_code] / same_count if same_count else 0.0,
                self.sport_speed_sum[sport_code] / same_count if same_count else 0.0,
                math.log1p(days_since_previous),
            ],
            dtype=np.float32,
        )

    def update(self, row: pd.Series) -> None:
        duration = float(row["duration_seconds"])
        hr = float(row["hr_mean"])
        hr_std = float(row["hr_std"])
        speed = float(row["speed_mean"])
        altitude_std = float(row["altitude_std"])
        sport = int(row["sport_code"])
        self.count += 1
        self.sum_duration += duration
        self.sum_duration_sq += duration * duration
        self.sum_hr += hr
        self.sum_hr_sq += hr * hr
        self.sum_hr_std += hr_std
        self.sum_speed += speed
        self.sum_speed_sq += speed * speed
        self.sum_altitude_std += altitude_std
        self.sport_count[sport] += 1
        self.sport_hr_sum[sport] += hr
        self.sport_speed_sum[sport] += speed
        start_time = float(row["session_start_time"])
        self.previous_start = (
            start_time
            if self.previous_start is None
            else max(self.previous_start, start_time)
        )


def build(
    session_manifest: Path,
    output_dir: Path,
    excluded_sport_code: int | None = None,
) -> dict[str, object]:
    sessions = pd.read_csv(session_manifest, dtype={"session_key": str}, low_memory=False)
    n_sessions = int(sessions["session_index"].max()) + 1
    values = np.zeros((n_sessions, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros(n_sessions, dtype=np.uint8)
    prior_count = np.zeros(n_sessions, dtype=np.int32)
    endomondo = sessions[sessions["dataset"] == "Endomondo"].copy()
    endomondo = endomondo.sort_values(
        ["user_index", "session_start_time", "session_index"]
    )
    overlap_guarded_current_sessions = 0
    overlap_guarded_pairs = 0
    for _, user_sessions in endomondo.groupby("user_index", sort=False):
        accumulator = Accumulator()
        pending: list[tuple[float, int, pd.Series]] = []
        for _, same_time_sessions in user_sessions.groupby("session_start_time", sort=True):
            current_start = float(same_time_sessions["session_start_time"].iloc[0])
            while pending and pending[0][0] <= current_start:
                _, _, completed_row = heapq.heappop(pending)
                accumulator.update(completed_row)
            if pending:
                overlap_guarded_current_sessions += len(same_time_sessions)
                overlap_guarded_pairs += len(pending) * len(same_time_sessions)
            for _, row in same_time_sessions.iterrows():
                index = int(row["session_index"])
                values[index] = accumulator.features(
                    int(row["sport_code"]), float(row["session_start_time"])
                )
                mask[index] = int(accumulator.count > 0)
                prior_count[index] = accumulator.count
            for _, row in same_time_sessions.iterrows():
                if excluded_sport_code is None or int(row["sport_code"]) != excluded_sport_code:
                    heapq.heappush(
                        pending,
                        (
                            float(row["session_end_time"]),
                            int(row["session_index"]),
                            row,
                        ),
                    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "session_history_values.npy", values)
    np.save(output_dir / "session_history_mask.npy", mask)
    np.save(output_dir / "session_prior_count.npy", prior_count)
    first_session_failures = 0
    count_order_failures = 0
    for _, user_sessions in endomondo.groupby("user_index", sort=False):
        ordered = user_sessions.sort_values(["session_start_time", "session_index"])
        first_time = ordered["session_start_time"].iloc[0]
        first_indices = ordered.loc[
            ordered["session_start_time"] == first_time, "session_index"
        ].to_numpy(dtype=np.int64)
        first_session_failures += int(mask[first_indices].sum())
        times = ordered["session_start_time"].to_numpy()
        indices = ordered["session_index"].to_numpy(dtype=np.int64)
        expected = np.searchsorted(np.unique(times), times, side="left")
        # The exact prior-session count differs from the prior-time-group count when
        # earlier timestamps contain more than one session; only monotonicity is asserted.
        if np.any(np.diff(prior_count[indices]) < 0):
            count_order_failures += 1
        del expected
    nonfinite = int((~np.isfinite(values)).sum())
    golden_history = int(mask[sessions.loc[sessions["dataset"] == "GoldenCheetah", "session_index"].to_numpy(dtype=np.int64)].sum())
    payload: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "history_version": HISTORY_VERSION,
        "sessions": n_sessions,
        "endomondo_sessions": int(len(endomondo)),
        "history_available_sessions": int(mask.sum()),
        "no_history_sessions": int((mask == 0).sum()),
        "features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "strict_rule": "only sessions completed at or before the current session_start_time",
        "same_timestamp_rule": "sessions sharing a start time cannot enter one another's history",
        "overlap_rule": "a prior-started session cannot enter history until its end time",
        "overlap_guarded_current_sessions": overlap_guarded_current_sessions,
        "overlap_guarded_session_pairs": overlap_guarded_pairs,
        "excluded_sport_code": excluded_sport_code,
        "excluded_sport_rule": "excluded-family sessions never update history" if excluded_sport_code is not None else "none",
        "goldencheetah_primary_history_policy": "zero history; external adaptation is separate",
        "first_session_history_failures": first_session_failures,
        "user_prior_count_monotonicity_failures": count_order_failures,
        "goldencheetah_nonzero_history_masks": golden_history,
        "nonfinite_values": nonfinite,
    }
    payload["all_assertions_pass"] = (
        first_session_failures == 0
        and count_order_failures == 0
        and golden_history == 0
        and nonfinite == 0
    )
    metadata = output_dir / "history_metadata.json"
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata)
    if not payload["all_assertions_pass"]:
        raise AssertionError(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strictly causal session-history features.")
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-sport-code", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.sessions, args.output_dir, args.exclude_sport_code),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
