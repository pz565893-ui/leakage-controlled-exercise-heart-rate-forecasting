from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_independent_zero_history_queue import (  # noqa: E402
    EXPECTED_SEEDS,
    PHASES,
    QueueRunner,
    validate_frozen_configuration,
    zero_history_npz_postcondition_passed,
)


class IndependentZeroHistoryQueueTests(unittest.TestCase):
    def frozen_config(self) -> dict[str, object]:
        return json.loads(
            (ROOT / "configs" / "independent_zero_history_v0_23_0.json").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_configuration_is_accepted_and_mutation_rejected(self) -> None:
        config = self.frozen_config()
        validate_frozen_configuration(config)
        config["training"]["selection_metric"] = "mixed validation score"  # type: ignore[index]
        with self.assertRaises(AssertionError):
            validate_frozen_configuration(config)

    def test_prediction_archive_requires_only_registered_zero_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.npz"
            np.savez_compressed(
                valid,
                row_index=np.arange(4, dtype=np.int64),
                zero_history_quantiles=np.zeros((4, 3, 7), dtype=np.float32),
            )
            self.assertTrue(zero_history_npz_postcondition_passed(valid))

            mixed = root / "mixed.npz"
            np.savez_compressed(
                mixed,
                row_index=np.arange(4, dtype=np.int64),
                zero_history_quantiles=np.zeros((4, 3, 7), dtype=np.float32),
                history_quantiles=np.zeros((4, 3, 7), dtype=np.float32),
            )
            self.assertFalse(zero_history_npz_postcondition_passed(mixed))

    def test_dry_run_builds_all_ten_registered_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = QueueRunner(
                project_root=ROOT,
                configuration=ROOT
                / "configs"
                / "independent_zero_history_v0_23_0.json",
                specification=ROOT
                / "protocol"
                / "INDEPENDENT_ZERO_HISTORY_ABLATION_SPECIFICATION.md",
                output_root=Path(temporary) / "queue",
                dry_run=True,
            )
            runner.run(list(PHASES), None)
            tasks = runner.manifest["tasks"]
            self.assertEqual(len(tasks), 2 * len(EXPECTED_SEEDS))
            expected = {
                f"seed_{seed}/{protocol}"
                for seed in EXPECTED_SEEDS
                for protocol in PHASES
            }
            self.assertEqual(set(tasks), expected)
            self.assertTrue(all(task["status"] == "dry_run" for task in tasks.values()))
            self.assertTrue(
                all(
                    "--training-history-mode always_zero" in task["command"]
                    for task in tasks.values()
                )
            )
            self.assertTrue(
                all(
                    "test_predictions.npz" in task["command"]
                    for key, task in tasks.items()
                    if key.endswith("/unseen_user")
                )
            )

    @unittest.skipUnless(sys.platform == "win32", "Windows MAX_PATH regression")
    def test_longest_formal_model_artifact_stays_below_legacy_max_path(self) -> None:
        longest = (
            ROOT
            / "outputs"
            / "independent_zero_history_v0_23_0"
            / f"seed_{EXPECTED_SEEDS[0]}"
            / "unseen_user"
            / "m"
            / "history_normalization_unseen_user_train.json.tmp"
        )
        self.assertLess(
            len(str(longest)),
            260,
            f"formal artifact exceeds the traditional Windows MAX_PATH: {longest}",
        )


if __name__ == "__main__":
    unittest.main()
