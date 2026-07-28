from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aggregate_independent_zero_history import (  # noqa: E402
    AggregationBlockedError,
    aggregate,
    hierarchy_user_means,
    load_prediction_npz,
    output_paths,
    percentile_user_bootstrap,
    validate_frozen_config,
)


class IndependentZeroHistoryAggregationTests(unittest.TestCase):
    def config(self) -> dict[str, object]:
        return json.loads(
            (ROOT / "configs" / "independent_zero_history_v0_23_0.json").read_text(
                encoding="utf-8"
            )
        )

    def namespace(self, root: Path, allow_incomplete: bool) -> argparse.Namespace:
        return argparse.Namespace(
            independent_root=root / "independent",
            mixed_root=root / "mixed",
            array_dir=root / "arrays",
            config=ROOT / "configs" / "independent_zero_history_v0_23_0.json",
            output_dir=root / "aggregation",
            allow_incomplete=allow_incomplete,
            bootstrap_seed=20260722,
        )

    def test_frozen_config_declares_five_seeds_and_two_protocols(self) -> None:
        seeds, protocols = validate_frozen_config(self.config())
        self.assertEqual(seeds, [20260722, 20260723, 20260724, 20260725, 20260726])
        self.assertEqual(set(protocols), {"unseen_user", "strict_temporal"})

    def test_session_then_user_hierarchy_is_not_origin_weighted(self) -> None:
        losses = np.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        )
        users = np.array([1, 1, 1, 2])
        sessions = np.array([10, 10, 11, 20])
        user_ids, means = hierarchy_user_means(losses, users, sessions)
        self.assertTrue(np.array_equal(user_ids, [1, 2]))
        # User 1: mean within session 10 is [1,2], then equal-session mean with
        # session 11 [10,20] gives [5.5,11].
        self.assertTrue(np.allclose(means[0], [5.5, 11.0]))
        self.assertTrue(np.allclose(means[1], [30.0, 40.0]))

    def test_user_bootstrap_is_deterministic_and_centred_on_user_mean(self) -> None:
        values = np.array([-1.0, 0.0, 2.0, 3.0])
        first = percentile_user_bootstrap(values, replicates=1000, seed=7)
        second = percentile_user_bootstrap(values, replicates=1000, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], values.mean())
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])

    def test_prediction_loader_rejects_duplicate_row_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.npz"
            np.savez_compressed(
                path,
                row_index=np.array([1, 1], dtype=np.int64),
                zero_history_quantiles=np.full((2, 3, 7), 100.0, dtype=np.float32),
            )
            with self.assertRaises(AssertionError):
                load_prediction_npz(path, ("zero_history_quantiles",))

    def test_incomplete_mode_emits_only_progress_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.namespace(root, allow_incomplete=True)
            result = aggregate(args)
            self.assertEqual(result["status"], "pending")
            self.assertFalse(result["final_artifacts_emitted"])
            paths = output_paths(args.output_dir)
            self.assertTrue(paths["audit"].is_file())
            self.assertTrue(paths["progress"].is_file())
            for key in ("per_seed", "seed_summary", "user_seed_mean", "user_bootstrap"):
                self.assertFalse(paths[key].exists())

    def test_strict_mode_blocks_when_any_job_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(AggregationBlockedError):
                aggregate(self.namespace(Path(temporary), allow_incomplete=False))


if __name__ == "__main__":
    unittest.main()
