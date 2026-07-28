from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_model_arrays import context_positions, masked_summary  # noqa: E402


class ModelArrayTests(unittest.TestCase):
    def test_context_positions_match_right_closed_grid(self) -> None:
        self.assertEqual(context_positions(600.0, 1, 100), (30, 60))

    def test_masked_summary_ignores_missing_values(self) -> None:
        mean, standard_deviation, count = masked_summary(
            [100.0, 999.0, 120.0], [1, 0, 1]
        )
        self.assertEqual(mean, 110.0)
        self.assertEqual(standard_deviation, 10.0)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
