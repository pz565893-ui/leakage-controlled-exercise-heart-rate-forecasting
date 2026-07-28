"""Build a privacy-conservative code-repository upload bundle.

The builder consumes the verified public-release integrity manifest, copies
only manifest-listed files, adds the manifest and its audit, and writes a ZIP
whose paths are relative to the intended repository root. It never traverses
raw or row-level data directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.30.0"
STEM = "code_repository_upload_v0_30_0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def io_path(path: Path) -> Path:
    """Return a Windows extended-length path for filesystem I/O when needed."""

    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Public-release integrity manifest is empty")
    return rows


def safe_source(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise RuntimeError(f"Unsafe or missing manifest path: {relative}")
    return candidate


def build(root: Path, output_dir: Path, zip_path: Path) -> dict[str, object]:
    release = (root / "release").resolve()
    output_dir = output_dir.resolve()
    zip_path = zip_path.resolve()
    if output_dir.parent != release or zip_path.parent != release:
        raise RuntimeError("Package outputs must remain directly under the release directory")
    manifest = release / "PUBLIC_RELEASE_INTEGRITY_v0_1_0.csv"
    manifest_audit = release / "PUBLIC_RELEASE_INTEGRITY_v0_1_0.audit.json"
    rows = read_manifest(manifest)

    if output_dir.exists() or zip_path.exists():
        raise RuntimeError(
            "Output already exists; use a new release version instead of overwriting it"
        )

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Temporary output already exists: {temporary.name}")
    temporary.mkdir(parents=True)

    try:
        for row in rows:
            relative = row["relative_path"]
            source = safe_source(root, relative)
            if sha256_file(source) != row["sha256"]:
                raise RuntimeError(f"Manifest hash mismatch before copy: {relative}")
            destination = temporary / Path(relative)
            io_path(destination.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(io_path(source), io_path(destination))

        packaged_relatives = [row["relative_path"] for row in rows]
        for source in (manifest, manifest_audit):
            destination = temporary / "release" / source.name
            io_path(destination.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(io_path(source), io_path(destination))
            packaged_relatives.append(destination.relative_to(temporary).as_posix())

        file_list = sorted(packaged_relatives)
        io_path(temporary / "UPLOAD_FILE_LIST.txt").write_text(
            "\n".join(file_list) + "\n", encoding="utf-8"
        )
        temporary.replace(output_dir)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in file_list + ["UPLOAD_FILE_LIST.txt"]:
                archive.write(io_path(output_dir / relative), relative)

        category_counts = Counter(row["category"] for row in rows)
        payload: dict[str, object] = {
            "package_version": VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "directory": output_dir.name,
            "zip_file": zip_path.name,
            "zip_size_bytes": zip_path.stat().st_size,
            "zip_sha256": sha256_file(zip_path),
            "manifest_entries": len(rows),
            "package_files": len(file_list) + 1,
            "category_counts": dict(sorted(category_counts.items())),
            "privacy_boundary": (
                "manifest-listed aggregate/code/documentation files only; raw records, "
                "row-level arrays, identifiers, checkpoints, and TIFF masters excluded"
            ),
        }
        audit = release / "CODE_REPOSITORY_PACKAGE_v0_30_0.audit.json"
        audit.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload
    except Exception:
        if temporary.exists() and temporary.parent == release:
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "release" / STEM
    )
    zip_path = args.zip.resolve() if args.zip else root / "release" / f"{STEM}.zip"
    print(json.dumps(build(root, output_dir, zip_path), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
