from __future__ import annotations

import csv
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_public_repository_package import build, sha256_file  # noqa: E402


class PublicRepositoryPackageTests(unittest.TestCase):
    def test_build_copies_only_manifest_listed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "release"
            release.mkdir()
            readme = root / "README.md"
            readme.write_text("public\n", encoding="utf-8")
            (root / "private.txt").write_text("private\n", encoding="utf-8")

            manifest = release / "PUBLIC_RELEASE_INTEGRITY_v0_1_0.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "manifest_version",
                        "relative_path",
                        "category",
                        "size_bytes",
                        "sha256",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "manifest_version": "0.1.0",
                        "relative_path": "README.md",
                        "category": "release_metadata",
                        "size_bytes": readme.stat().st_size,
                        "sha256": sha256_file(readme),
                    }
                )
            (release / "PUBLIC_RELEASE_INTEGRITY_v0_1_0.audit.json").write_text(
                "{}\n", encoding="utf-8"
            )

            output = release / "code_repository_upload_v0_30_0"
            archive = release / "code_repository_upload_v0_30_0.zip"
            payload = build(root, output, archive)

            self.assertEqual(payload["status"], "PASS")
            self.assertTrue((output / "README.md").is_file())
            self.assertFalse((output / "private.txt").exists())
            with zipfile.ZipFile(archive) as zipped:
                names = set(zipped.namelist())
            self.assertIn("README.md", names)
            self.assertIn("release/PUBLIC_RELEASE_INTEGRITY_v0_1_0.csv", names)
            self.assertIn("UPLOAD_FILE_LIST.txt", names)

    def test_output_must_remain_under_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "release").mkdir()
            with self.assertRaisesRegex(RuntimeError, "release directory"):
                build(root, root / "outside", root / "release" / "bundle.zip")


if __name__ == "__main__":
    unittest.main()

