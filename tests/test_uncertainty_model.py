from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_uncertainty_model import (  # noqa: E402
    HistoryQuantileTCN,
    apply_signal_input,
    conformal_thresholds,
    pinball_loss,
    prepare_history_batch,
    selection_metric_name,
    validation_selection_score,
)


class UncertaintyModelTests(unittest.TestCase):
    def test_heart_rate_only_ablation_masks_speed_and_altitude(self) -> None:
        sequence = torch.ones(2, 3, 6)
        output = apply_signal_input(sequence, "heart_rate_only")
        expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertTrue(torch.equal(output[0, 0], expected))

    def test_zero_initialized_quantiles_are_ordered(self) -> None:
        model = HistoryQuantileTCN().eval()
        with torch.no_grad():
            output = model(
                torch.randn(2, 30, 6),
                torch.tensor([1, 3]),
                torch.zeros(2, 1),
                torch.zeros(2, 13),
                torch.zeros(2, 1),
            )
        self.assertEqual(tuple(output.shape), (2, 3, 7))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_pinball_loss_is_zero_for_exact_quantiles(self) -> None:
        prediction = torch.zeros(4, 3, 7)
        target = torch.zeros(4, 3)
        self.assertEqual(float(pinball_loss(prediction, target)), 0.0)

    def test_conformal_adjustment_expands_undercoverage(self) -> None:
        prediction = np.zeros((20, 3, 7), dtype=np.float32)
        target = np.ones((20, 3), dtype=np.float32) * 10.0
        thresholds = conformal_thresholds(prediction, target)
        self.assertEqual(thresholds["0.9"], [10.0, 10.0, 10.0])

    def test_always_zero_selection_uses_only_zero_history_validation(self) -> None:
        self.assertEqual(
            selection_metric_name("always_zero"),
            "mean 1/3/5-min zero-history validation hierarchical MAE",
        )
        self.assertEqual(
            validation_selection_score(
                training_history_mode="always_zero",
                history_mae=None,
                zero_history_mae=7.25,
            ),
            7.25,
        )
        self.assertEqual(
            validation_selection_score(
                training_history_mode="mixed",
                history_mae=6.0,
                zero_history_mae=8.0,
            ),
            7.0,
        )

    def test_force_zero_history_batch_masks_every_history_feature(self) -> None:
        history_values = np.arange(39, dtype=np.float32).reshape(3, 13)
        history_mask = np.ones(3, dtype=np.uint8)
        session_index = np.array([0, 1, 2], dtype=np.int64)
        normalization = {
            "history_mean": np.zeros(13, dtype=float).tolist(),
            "history_std": np.ones(13, dtype=float).tolist(),
        }
        values, mask = prepare_history_batch(
            history_values,
            history_mask,
            session_index,
            np.array([0, 1, 2], dtype=np.int64),
            normalization,
            torch.device("cpu"),
            force_zero_history=True,
        )
        self.assertTrue(torch.equal(mask, torch.zeros_like(mask)))
        self.assertTrue(torch.equal(values, torch.zeros_like(values)))


if __name__ == "__main__":
    unittest.main()
