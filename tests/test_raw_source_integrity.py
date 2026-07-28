from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_raw_source_integrity import (  # noqa: E402
    atomic_write_json,
    build_audit,
    deterministic_tree_sha256,
)


class RawSourceIntegrityTests(unittest.TestCase):
    def _make_inputs(self, root: Path, reverse_creation: bool = False):
        hr = root / "private_hr_source.json"
        metadata = root / "private_metadata_source.json"
        golden = root / "private_user_collection"
        hr.write_bytes(b'{"userId": 123, "hr": [80, 81]}\n')
        metadata.write_bytes(b'{"userId": 123, "gender": "x"}\n')

        members = (
            ("athlete_b/second.csv", b"secs,hr\n0,80\n"),
            ("athlete_a/config.json", b'{"private": true}\n'),
            ("athlete_a/no_extension", b"opaque\n"),
        )
        for relative_path, content in reversed(members) if reverse_creation else members:
            path = golden / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return hr, metadata, golden, members

    def test_synthetic_audit_counts_hashes_and_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr, metadata, golden, members = self._make_inputs(root)
            payload = build_audit(hr, metadata, golden, chunk_size=7)

            self.assertEqual(
                payload["sources"]["endomondo_hr_json"]["sha256"],
                hashlib.sha256(hr.read_bytes()).hexdigest(),
            )
            golden_summary = payload["sources"]["goldencheetah_extracted"]
            self.assertEqual(golden_summary["file_count"], 3)
            self.assertEqual(
                golden_summary["bytes"], sum(len(content) for _, content in members)
            )
            self.assertEqual(
                golden_summary["extension_counts"],
                {".csv": 1, ".json": 1, "<none>": 1},
            )
            self.assertEqual(len(golden_summary["sha256"]), 64)
            self.assertEqual(len(golden_summary["tree_sha256"]), 64)

    def test_tree_hash_is_creation_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            hr1, metadata1, golden1, _ = self._make_inputs(first)
            hr2, metadata2, golden2, _ = self._make_inputs(
                second, reverse_creation=True
            )
            audit1 = build_audit(hr1, metadata1, golden1, chunk_size=5)
            audit2 = build_audit(hr2, metadata2, golden2, chunk_size=5)
            summary1 = audit1["sources"]["goldencheetah_extracted"]
            summary2 = audit2["sources"]["goldencheetah_extracted"]
            self.assertEqual(summary1["sha256"], summary2["sha256"])
            self.assertEqual(summary1["tree_sha256"], summary2["tree_sha256"])

    def test_tree_hash_detects_path_change(self) -> None:
        content_hash = hashlib.sha256(b"same").hexdigest()
        first = deterministic_tree_sha256([("a/file.csv", content_hash)])
        second = deterministic_tree_sha256([("b/file.csv", content_hash)])
        self.assertNotEqual(first, second)

    def test_serialized_audit_omits_paths_ids_and_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr, metadata, golden, _ = self._make_inputs(root)
            payload = build_audit(hr, metadata, golden, chunk_size=11)
            output = root / "audit.json"
            atomic_write_json(output, payload)
            serialized = output.read_text(encoding="utf-8")

            self.assertNotIn(str(root), serialized)
            self.assertNotIn("private_hr_source.json", serialized)
            self.assertNotIn("private_metadata_source.json", serialized)
            self.assertNotIn("athlete_a", serialized)
            self.assertNotIn("athlete_b", serialized)
            self.assertNotIn("userId", serialized)
            self.assertNotIn('"123"', serialized)
            parsed = json.loads(serialized)
            self.assertFalse(parsed["privacy"]["absolute_paths_written"])
            self.assertFalse(parsed["privacy"]["source_file_manifest_written"])


if __name__ == "__main__":
    unittest.main()
