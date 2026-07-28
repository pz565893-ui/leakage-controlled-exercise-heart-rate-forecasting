from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_multiseed_paired_user_bootstrap_v0_25_0 import (  # noqa: E402
    ANALYSIS_VERSION,
    DEFAULT_SEEDS,
    HORIZONS,
    SPORT_FAMILIES,
    analyze,
    hierarchical_paired_mae,
    stable_seed,
)


def quantiles(median: np.ndarray) -> np.ndarray:
    return np.repeat(median[:, :, None], 7, axis=2)


def write_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def write_reference(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for family in SPORT_FAMILIES:
        for regime in (
            f"unseen_sport__{family}",
            f"joint_user_sport__{family}",
        ):
            for horizon in HORIZONS:
                rows.append(
                    {
                        "model_version": "0.12.0",
                        "held_sport_family": family,
                        "regime": regime,
                        "model": "ewma_alpha_0_1",
                        "horizon_seconds": horizon,
                        "mae_bpm": 5.0,
                        "rmse_bpm": 5.0,
                        "bias_bpm": -5.0,
                        "users": 2,
                        "sessions": 2,
                        "origins": 4,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_learned_reference(path: Path) -> None:
    rows: list[dict[str, object]] = []
    regimes = (
        "within_user_temporal_test",
        "unseen_user_test",
        "goldencheetah_frozen_external",
    )
    for seed in DEFAULT_SEEDS:
        for comparator, comparator_mae in (("gru", 2.0), ("tcn", 0.5)):
            for regime in regimes:
                for horizon in HORIZONS:
                    rows.append(
                        {
                            "seed": seed,
                            "comparator_model": comparator,
                            "regime": regime,
                            "horizon_seconds": horizon,
                            "main_mae_bpm": 1.0,
                            "comparator_mae_bpm": comparator_mae,
                            "main_minus_comparator_mae_bpm": 1.0 - comparator_mae,
                            "users": 2,
                        }
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root: Path) -> argparse.Namespace:
    array_dir = root / "arrays"
    prediction_root = root / "predictions"
    array_dir.mkdir(parents=True)
    n_rows = 20
    targets = np.full((n_rows, 3), 105.0, dtype=np.float32)
    dataset = np.zeros(n_rows, dtype=np.int8)
    dataset[8:12] = 1
    unseen = np.zeros(n_rows, dtype=np.int8)
    unseen[4:8] = 4
    external = np.zeros(n_rows, dtype=np.int8)
    external[8:12] = 1
    temporal = np.zeros(n_rows, dtype=np.int8)
    temporal[0:4] = 4
    users = np.asarray(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
        dtype=np.int32,
    )
    sessions = np.asarray(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
        dtype=np.int32,
    )
    values = np.full((n_rows, 30, 1), 100.0, dtype=np.float32)
    masks = np.ones((n_rows, 30, 1), dtype=bool)
    arrays = {
        "targets.npy": targets,
        "dataset_code.npy": dataset,
        "unseen_user_partition.npy": unseen,
        "primary_external_partition.npy": external,
        "temporal_partition_strict.npy": temporal,
        "user_index.npy": users,
        "session_index.npy": sessions,
        "sequence_values.npy": values,
        "sequence_masks.npy": masks,
    }
    for filename, value in arrays.items():
        np.save(array_dir / filename, value)

    temporal_rows = np.arange(0, 4, dtype=np.int64)
    development_rows = np.arange(4, 8, dtype=np.int64)
    external_rows = np.arange(8, 12, dtype=np.int64)
    comparator_rows = np.arange(4, 12, dtype=np.int64)
    sport_rows = np.arange(12, 20, dtype=np.int64)
    for seed_position, seed in enumerate(DEFAULT_SEEDS):
        offset = (-0.1, 0.0, 0.1)[seed_position]
        seed_root = prediction_root / f"seed_{seed}"
        temporal_main = np.full((4, 3), 106.0 + offset, dtype=np.float32)
        write_npz(
            seed_root / "temporal_main" / "predictions.npz",
            row_index=temporal_rows,
            history_quantiles=quantiles(temporal_main),
            zero_history_quantiles=quantiles(temporal_main),
        )
        development_main = np.full((4, 3), 106.0 + offset, dtype=np.float32)
        write_npz(
            seed_root / "unseen_main" / "development_predictions.npz",
            row_index=development_rows,
            history_quantiles=quantiles(development_main),
            zero_history_quantiles=quantiles(development_main),
        )
        external_main = np.full((4, 3), 106.0 + offset, dtype=np.float32)
        write_npz(
            seed_root / "unseen_main" / "external_predictions.npz",
            row_index=external_rows,
            zero_history_quantiles=quantiles(external_main),
        )
        for comparator, value in (("gru", 107.0), ("tcn", 105.5)):
            write_npz(
                seed_root / f"temporal_{comparator}" / "predictions.npz",
                row_index=temporal_rows,
                predictions=np.full((4, 3), value, dtype=np.float32),
            )
            write_npz(
                seed_root / f"unseen_{comparator}" / "predictions.npz",
                row_index=comparator_rows,
                predictions=np.full((8, 3), value, dtype=np.float32),
            )
        sport_main = np.full((8, 3), 106.0 + offset, dtype=np.float32)
        for family in SPORT_FAMILIES:
            write_npz(
                seed_root / "held_sport" / family / "predictions.npz",
                row_index=sport_rows,
                same_user_rows=np.asarray([4], dtype=np.int64),
                history_quantiles=quantiles(sport_main),
                zero_history_quantiles=quantiles(sport_main),
            )

    reference = root / "aligned_reference.csv"
    write_reference(reference)
    learned_reference = root / "learned_reference.csv"
    write_learned_reference(learned_reference)
    return argparse.Namespace(
        array_dir=array_dir,
        prediction_root=prediction_root,
        aligned_sport_reference=reference,
        learned_comparison_reference=learned_reference,
        seeds=list(DEFAULT_SEEDS),
        bootstrap_replicates=1_000,
        bootstrap_seed=20260725,
        learned_output=root / "outputs" / "model_v0_25_0.csv",
        sport_output=root / "outputs" / "sport_v0_25_0.csv",
        audit=root / "outputs" / "audit_v0_25_0.json",
    )


class MultiSeedPairedUserBootstrapTests(unittest.TestCase):
    def test_hierarchical_paired_mae_respects_equal_session_weighting(self) -> None:
        frame = hierarchical_paired_mae(
            main_prediction=np.asarray([1.0, 3.0, 5.0, 8.0]),
            comparator_prediction=np.asarray([2.0, 2.0, 2.0, 2.0]),
            target=np.zeros(4),
            users=np.asarray([0, 0, 0, 1]),
            sessions=np.asarray([0, 0, 1, 2]),
        )
        self.assertAlmostEqual(frame.loc[0, "main_mae_bpm"], 3.5)
        self.assertAlmostEqual(frame.loc[0, "comparator_mae_bpm"], 2.0)
        self.assertAlmostEqual(frame.loc[0, "delta_mae_bpm"], 1.5)

    def test_stable_seed_is_repeatable_and_label_specific(self) -> None:
        self.assertEqual(stable_seed(1, "a"), stable_seed(1, "a"))
        self.assertNotEqual(stable_seed(1, "a"), stable_seed(1, "b"))

    def test_complete_fixture_emits_aggregate_only_audited_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = build_fixture(Path(temporary))
            audit = analyze(args)
            self.assertEqual(audit["analysis_version"], ANALYSIS_VERSION)
            self.assertTrue(audit["all_assertions_pass"])
            self.assertEqual(audit["outputs"]["learned_comparator_rows"], 18)
            self.assertEqual(audit["outputs"]["sport_shift_rows"], 30)
            self.assertEqual(audit["outputs"]["joint_shift_rows_lt25_users"], 15)

            learned = pd.read_csv(args.learned_output)
            sport = pd.read_csv(args.sport_output)
            self.assertEqual(len(learned), 18)
            self.assertEqual(len(sport), 30)
            self.assertTrue(learned["main_mode"].eq("history_masked").all())
            self.assertAlmostEqual(
                learned[learned["comparator_model"] == "gru"]["delta_mae_bpm"].iloc[0],
                -1.0,
                places=5,
            )
            self.assertAlmostEqual(
                learned[learned["comparator_model"] == "tcn"]["delta_mae_bpm"].iloc[0],
                0.5,
                places=5,
            )
            self.assertTrue(np.allclose(sport["delta_mae_bpm"], -4.0))
            self.assertTrue(
                sport[sport["joint_support_caution_lt25_users"]]["analysis_role"]
                .eq("exploratory_joint_shift_lt25_users")
                .all()
            )
            forbidden = {
                "user",
                "user_id",
                "user_index",
                "session",
                "session_id",
                "session_index",
                "row_index",
            }
            self.assertFalse(forbidden.intersection(learned.columns))
            self.assertFalse(forbidden.intersection(sport.columns))
            parsed = json.loads(args.audit.read_text(encoding="utf-8"))
            self.assertGreater(len(parsed["input_sha256"]), 20)
            self.assertTrue(parsed["assertions"]["no_training_or_model_selection"])


if __name__ == "__main__":
    unittest.main()
