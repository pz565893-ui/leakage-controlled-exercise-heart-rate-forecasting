from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_neural_baselines import (  # noqa: E402
    GRUForecast,
    TCNForecast,
    TransformerForecast,
    last_observed_hr,
)


class NeuralBaselineTests(unittest.TestCase):
    def test_last_observed_hr_skips_missing_origin_bin(self) -> None:
        values = np.zeros((1, 30, 3), dtype=np.float32)
        masks = np.zeros((1, 30, 3), dtype=np.float32)
        values[0, 27, 0] = 133.0
        masks[0, 27, 0] = 1.0
        self.assertEqual(last_observed_hr(values, masks)[0], 133.0)

    def test_all_models_return_three_zero_initialized_residuals(self) -> None:
        sequence = torch.randn(2, 30, 6)
        sport = torch.tensor([1, 3])
        elapsed = torch.zeros(2, 1)
        for model in (GRUForecast(), TCNForecast(), TransformerForecast()):
            model.eval()
            with torch.no_grad():
                output = model(sequence, sport, elapsed)
            self.assertEqual(tuple(output.shape), (2, 3))
            self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_tcn_is_causal_with_respect_to_future_context_positions(self) -> None:
        model = TCNForecast().eval()
        model.head.layers[-1].weight.data.fill_(1.0)
        first = torch.zeros(1, 30, 6)
        second = first.clone()
        second[:, 20:, :] = 1.0
        sport = torch.tensor([1])
        elapsed = torch.zeros(1, 1)
        with torch.no_grad():
            early_first = model.blocks(model.input_projection(first.transpose(1, 2)))[:, :, 10]
            early_second = model.blocks(model.input_projection(second.transpose(1, 2)))[:, :, 10]
        self.assertTrue(torch.allclose(early_first, early_second, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
