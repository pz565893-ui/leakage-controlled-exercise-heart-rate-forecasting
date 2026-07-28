from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_strict_temporal_partition import (  # noqa: E402
    exclude_cross_boundary_overlaps,
    validate_order,
)


class StrictTemporalPartitionTests(unittest.TestCase):
    def test_boundary_overlap_sessions_are_excluded(self) -> None:
        sessions = pd.DataFrame(
            [
                [0, "Endomondo", 0, 0.0, 100.0],
                [1, "Endomondo", 0, 100.0, 200.0],
                [2, "Endomondo", 0, 300.0, 400.0],
            ],
            columns=[
                "session_index",
                "dataset",
                "user_index",
                "session_start_time",
                "session_end_time",
            ],
        )
        codes = np.asarray([1, 2, 3], dtype=np.uint8)
        strict, excluded = exclude_cross_boundary_overlaps(sessions, codes)
        self.assertEqual(excluded, {0, 1})
        self.assertTrue(np.array_equal(strict, np.asarray([5, 5, 3], dtype=np.uint8)))
        self.assertEqual(validate_order(sessions, strict), 0)

    def test_nonoverlapping_boundary_is_preserved(self) -> None:
        sessions = pd.DataFrame(
            [
                [0, "Endomondo", 0, 0.0, 99.0],
                [1, "Endomondo", 0, 100.0, 200.0],
            ],
            columns=[
                "session_index",
                "dataset",
                "user_index",
                "session_start_time",
                "session_end_time",
            ],
        )
        codes = np.asarray([1, 2], dtype=np.uint8)
        strict, excluded = exclude_cross_boundary_overlaps(sessions, codes)
        self.assertFalse(excluded)
        self.assertTrue(np.array_equal(strict, codes))


if __name__ == "__main__":
    unittest.main()
