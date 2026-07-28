from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_tabular_features import channel_features, feature_names  # noqa: E402


class TabularFeatureTests(unittest.TestCase):
    def test_constant_complete_channel(self) -> None:
        values = np.full((1, 30), 120.0, dtype=np.float32)
        mask = np.ones((1, 30), dtype=bool)
        output = channel_features(values, mask)[0]
        self.assertEqual(output[0], 120.0)
        self.assertEqual(output[1], 120.0)
        self.assertEqual(output[2], 0.0)
        self.assertEqual(output[5], 0.0)
        self.assertAlmostEqual(output[6], 0.0, places=7)
        self.assertEqual(output[7], 1.0)

    def test_slope_uses_observed_bin_times(self) -> None:
        values = np.zeros((1, 30), dtype=np.float32)
        mask = np.zeros((1, 30), dtype=bool)
        values[0, -3:] = [100.0, 110.0, 120.0]
        mask[0, -3:] = True
        output = channel_features(values, mask)[0]
        self.assertAlmostEqual(output[6], 1.0, places=5)
        self.assertEqual(output[0], 120.0)

    def test_feature_schema_has_39_columns(self) -> None:
        self.assertEqual(len(feature_names()), 39)


if __name__ == "__main__":
    unittest.main()
