from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_session_series import (  # noqa: E402
    assign_last_in_bin,
    build_series_row,
    decompress_float32,
    decompress_uint8,
    derive_distance_speed,
    derive_gps_speed,
    haversine_km,
)


class SessionSeriesTests(unittest.TestCase):
    def test_last_observation_is_retained_per_bin(self) -> None:
        values, mask = assign_last_in_bin(
            [1.0, 9.0, 10.0, 11.0],
            [100.0, 101.0, 102.0, 103.0],
            grid_start_bin=1,
            n_bins=2,
            lower=30.0,
            upper=240.0,
        )
        self.assertEqual(list(mask), [1, 1])
        self.assertEqual(list(values), [102.0, 103.0])

    def test_distance_speed_is_kmh_and_causal(self) -> None:
        speed = derive_distance_speed([0.0, 10.0, 20.0], [0.0, 0.1, 0.2])
        self.assertTrue(math.isnan(speed[0]))
        self.assertAlmostEqual(speed[1], 36.0)
        self.assertAlmostEqual(speed[2], 36.0)

    def test_haversine_and_gps_speed(self) -> None:
        distance = haversine_km(0.0, 0.0, 0.0, 0.001)
        self.assertGreater(distance, 0.1)
        self.assertLess(distance, 0.12)
        speed = derive_gps_speed([0.0, 10.0], [0.0, 0.0], [0.0, 0.001])
        self.assertTrue(math.isnan(speed[0]))
        self.assertGreater(speed[1], 35.0)
        self.assertLess(speed[1], 45.0)

    def test_compressed_sequences_round_trip(self) -> None:
        timestamps = [0.0, 10.0, 20.0, 30.0]
        row = build_series_row(
            "Endomondo",
            "1",
            timestamps,
            [100.0, 101.0, 102.0, 103.0],
            [10.0, 11.0, 12.0, 13.0],
            [math.nan, 5.0, 6.0, 7.0],
            "test",
        )
        n_bins = int(row[4])
        self.assertEqual(n_bins, 4)
        self.assertEqual(list(decompress_float32(row[11], n_bins)), [100.0, 101.0, 102.0, 103.0])
        self.assertEqual(list(decompress_uint8(row[12], n_bins)), [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
