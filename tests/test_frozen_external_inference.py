from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infer_frozen_external_uncertainty import (  # noqa: E402
    sha256_file,
    validate_freeze_record,
)


class FrozenExternalInferenceTests(unittest.TestCase):
    def make_record(self, root: Path) -> tuple[Path, dict[str, Path]]:
        artifacts = {
            name: root / f"{name}.bin"
            for name in (
                "checkpoint",
                "thresholds",
                "input_normalization",
                "history_normalization",
            )
        }
        for name, path in artifacts.items():
            path.write_bytes(name.encode("utf-8"))
        record = {
            "status": "frozen_before_external_inference",
            "seed": 7,
            "external_outcomes_used_for_selection": False,
            "artifacts": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in artifacts.items()
            },
            "source_code": {
                "training_script": {
                    "path": str((ROOT / "src" / "train_uncertainty_model.py").resolve()),
                    "sha256": sha256_file(
                        ROOT / "src" / "train_uncertainty_model.py"
                    ),
                },
                "external_inference_script": {
                    "path": str(
                        (ROOT / "src" / "infer_frozen_external_uncertainty.py").resolve()
                    ),
                    "sha256": sha256_file(
                        ROOT / "src" / "infer_frozen_external_uncertainty.py"
                    ),
                },
            },
        }
        record_path = root / "freeze.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        return record_path, artifacts

    def test_valid_freeze_record_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record, artifacts = self.make_record(Path(directory))
            result = validate_freeze_record(record, **artifacts)
            self.assertEqual(result["seed"], 7)

    def test_artifact_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record, artifacts = self.make_record(Path(directory))
            artifacts["checkpoint"].write_bytes(b"mutated")
            with self.assertRaises(AssertionError):
                validate_freeze_record(record, **artifacts)

    def test_external_selection_flag_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record, artifacts = self.make_record(Path(directory))
            payload = json.loads(record.read_text(encoding="utf-8"))
            payload["external_outcomes_used_for_selection"] = True
            record.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(AssertionError):
                validate_freeze_record(record, **artifacts)


if __name__ == "__main__":
    unittest.main()
