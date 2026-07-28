"""Create a privacy-conservative integrity audit of the three raw data sources.

The inputs are opened read-only.  The JSON output contains only logical source
names, counts, byte totals, hash values, algorithms, and elapsed time.  It never
contains absolute paths, user identifiers, or a source-file name manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
AUDIT_VERSION = "0.1.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "audit" / "RAW_SOURCE_INTEGRITY_v0_1_0.json"
DEFAULT_CHUNK_SIZE = 16 * 1024 * 1024


class IntegrityAuditError(RuntimeError):
    """Raised when an input cannot be hashed as a stable, read-only snapshot."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    relative_path: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int


ProgressCallback = Callable[[str, int, int, int], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_reparse_point(path: Path) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _assert_plain_regular_file(path: Path, label: str) -> os.stat_result:
    if not path.exists():
        raise IntegrityAuditError(f"{label} is missing")
    if path.is_symlink() or _is_reparse_point(path):
        raise IntegrityAuditError(f"{label} must not be a symlink or reparse point")
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise IntegrityAuditError(f"{label} is not a regular file")
    return file_stat


def _snapshot_matches(snapshot: FileSnapshot, current: os.stat_result) -> bool:
    return (
        snapshot.size_bytes == current.st_size
        and snapshot.mtime_ns == current.st_mtime_ns
        and snapshot.device == current.st_dev
        and snapshot.inode == current.st_ino
    )


def _extension(path: Path | PurePosixPath) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else "<none>"


def sha256_file(
    snapshot: FileSnapshot,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    current = _assert_plain_regular_file(snapshot.path, "source file")
    if not _snapshot_matches(snapshot, current):
        raise IntegrityAuditError("source file changed before hashing")

    digest = hashlib.sha256()
    with snapshot.path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    current = _assert_plain_regular_file(snapshot.path, "source file")
    if not _snapshot_matches(snapshot, current):
        raise IntegrityAuditError("source file changed during hashing")
    return digest.hexdigest()


def snapshot_single_file(path: Path) -> FileSnapshot:
    file_stat = _assert_plain_regular_file(path, "single-file source")
    return FileSnapshot(
        path=path,
        relative_path="",
        size_bytes=file_stat.st_size,
        mtime_ns=file_stat.st_mtime_ns,
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
    )


def _walk_error(error: OSError) -> None:
    raise IntegrityAuditError("recursive source could not be enumerated") from error


def snapshot_recursive_files(root: Path) -> list[FileSnapshot]:
    if not root.exists():
        raise IntegrityAuditError("recursive source is missing")
    if root.is_symlink() or _is_reparse_point(root):
        raise IntegrityAuditError("recursive source root must not be a symlink or reparse point")
    if not root.is_dir():
        raise IntegrityAuditError("recursive source is not a directory")

    snapshots: list[FileSnapshot] = []
    normalized_paths: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False, onerror=_walk_error
    ):
        directory_path = Path(directory)
        directory_names.sort()
        file_names.sort()

        for name in directory_names:
            child = directory_path / name
            if child.is_symlink() or _is_reparse_point(child):
                raise IntegrityAuditError(
                    "recursive source contains a symlink or reparse-point directory"
                )

        for name in file_names:
            path = directory_path / name
            file_stat = _assert_plain_regular_file(path, "recursive source member")
            relative_path = unicodedata.normalize(
                "NFC", path.relative_to(root).as_posix()
            )
            if relative_path in normalized_paths:
                raise IntegrityAuditError(
                    "recursive source has duplicate normalized relative paths"
                )
            normalized_paths.add(relative_path)
            snapshots.append(
                FileSnapshot(
                    path=path,
                    relative_path=relative_path,
                    size_bytes=file_stat.st_size,
                    mtime_ns=file_stat.st_mtime_ns,
                    device=file_stat.st_dev,
                    inode=file_stat.st_ino,
                )
            )

    if not snapshots:
        raise IntegrityAuditError("recursive source contains no regular files")
    snapshots.sort(key=lambda item: item.relative_path)
    return snapshots


def _snapshot_signature(
    snapshots: Iterable[FileSnapshot],
) -> tuple[tuple[str, int, int, int, int], ...]:
    return tuple(
        (
            item.relative_path,
            item.size_bytes,
            item.mtime_ns,
            item.device,
            item.inode,
        )
        for item in snapshots
    )


def content_multiset_sha256(file_hashes: Iterable[str]) -> str:
    """Hash a sorted multiset of raw 32-byte SHA-256 digests."""
    digest = hashlib.sha256()
    for file_hash in sorted(file_hashes):
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest()


def deterministic_tree_sha256(entries: Iterable[tuple[str, str]]) -> str:
    """Hash normalized relative paths plus file hashes in deterministic order.

    Each entry is encoded as uint64-big-endian(path UTF-8 byte length), followed
    by the path bytes and the raw 32-byte file digest.  Length-prefixing prevents
    ambiguous record boundaries.
    """
    digest = hashlib.sha256()
    normalized: list[tuple[str, str]] = []
    for relative_path, file_hash in entries:
        normalized.append((unicodedata.normalize("NFC", relative_path), file_hash.lower()))
    for relative_path, file_hash in sorted(normalized, key=lambda item: (item[0], item[1])):
        path_bytes = relative_path.encode("utf-8")
        digest.update(len(path_bytes).to_bytes(8, byteorder="big", signed=False))
        digest.update(path_bytes)
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest()


def audit_single_file(
    logical_name: str,
    path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    started = perf_counter()
    snapshot = snapshot_single_file(path)
    file_hash = sha256_file(snapshot, chunk_size=chunk_size)
    if progress is not None:
        progress(logical_name, 1, 1, snapshot.size_bytes)
    return {
        "source_kind": "single_file",
        "sha256": file_hash,
        "bytes": snapshot.size_bytes,
        "file_count": 1,
        "extension_counts": {_extension(path): 1},
        "duration_seconds": round(perf_counter() - started, 6),
    }


def audit_recursive_source(
    logical_name: str,
    root: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    started = perf_counter()
    initial_snapshots = snapshot_recursive_files(root)
    total_files = len(initial_snapshots)
    extension_counts: Counter[str] = Counter()
    entries: list[tuple[str, str]] = []
    bytes_hashed = 0

    for index, snapshot in enumerate(initial_snapshots, start=1):
        file_hash = sha256_file(snapshot, chunk_size=chunk_size)
        entries.append((snapshot.relative_path, file_hash))
        extension_counts[_extension(PurePosixPath(snapshot.relative_path))] += 1
        bytes_hashed += snapshot.size_bytes
        if progress is not None and (
            index == total_files or index == 1 or index % 5000 == 0
        ):
            progress(logical_name, index, total_files, bytes_hashed)

    final_snapshots = snapshot_recursive_files(root)
    if _snapshot_signature(initial_snapshots) != _snapshot_signature(final_snapshots):
        raise IntegrityAuditError("recursive source changed during hashing")

    file_hashes = [file_hash for _, file_hash in entries]
    return {
        "source_kind": "recursive_file_set",
        "sha256": content_multiset_sha256(file_hashes),
        "tree_sha256": deterministic_tree_sha256(entries),
        "bytes": bytes_hashed,
        "file_count": total_files,
        "extension_counts": dict(sorted(extension_counts.items())),
        "duration_seconds": round(perf_counter() - started, 6),
    }


def build_audit(
    endomondo_hr: Path,
    endomondo_metadata: Path,
    goldencheetah_root: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    started = perf_counter()
    sources = {
        "endomondo_hr_json": audit_single_file(
            "endomondo_hr_json",
            endomondo_hr,
            chunk_size=chunk_size,
            progress=progress,
        ),
        "endomondo_metadata_json": audit_single_file(
            "endomondo_metadata_json",
            endomondo_metadata,
            chunk_size=chunk_size,
            progress=progress,
        ),
        "goldencheetah_extracted": audit_recursive_source(
            "goldencheetah_extracted",
            goldencheetah_root,
            chunk_size=chunk_size,
            progress=progress,
        ),
    }
    return {
        "audit_type": "raw_source_integrity",
        "audit_version": AUDIT_VERSION,
        "generated_at_utc": utc_now(),
        "read_only_inputs": True,
        "privacy": {
            "absolute_paths_written": False,
            "source_file_names_written": False,
            "source_file_manifest_written": False,
            "user_identifiers_written": False,
        },
        "hashing": {
            "file_sha256": "SHA-256 of raw file bytes",
            "recursive_set_sha256": (
                "SHA-256 of lexicographically sorted raw per-file SHA-256 digests; "
                "duplicate file contents are retained"
            ),
            "tree_sha256": (
                "SHA-256 over records sorted by NFC-normalized POSIX relative path "
                "then file SHA-256; record encoding is uint64_be(path_utf8_length) "
                "|| path_utf8 || raw_file_sha256"
            ),
            "recursive_symlink_policy": "reject symlinks and reparse points",
        },
        "sources": sources,
        "audit_duration_seconds": round(perf_counter() - started, 6),
    }


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def progress_to_stderr(
    logical_name: str, completed_files: int, total_files: int, bytes_hashed: int
) -> None:
    gibibytes = bytes_hashed / (1024**3)
    print(
        f"{logical_name}: {completed_files:,}/{total_files:,} files, "
        f"{gibibytes:.3f} GiB hashed",
        file=sys.stderr,
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Hash raw Endomondo and GoldenCheetah inputs without modifying them "
            "or emitting paths, identifiers, or file-name manifests."
        )
    )
    result.add_argument(
        "--endomondo-hr",
        type=Path,
        default=WORKSPACE_ROOT / "endomondoHR.json" / "endomondoHR.json",
    )
    result.add_argument(
        "--endomondo-metadata",
        type=Path,
        default=WORKSPACE_ROOT / "endomondoMeta.json" / "endomondoMeta.json",
    )
    result.add_argument(
        "--goldencheetah-root",
        type=Path,
        default=WORKSPACE_ROOT / "GoldenCheetah_extracted",
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--chunk-size-mib",
        type=int,
        default=DEFAULT_CHUNK_SIZE // (1024 * 1024),
    )
    result.add_argument("--quiet", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.chunk_size_mib <= 0:
        raise SystemExit("--chunk-size-mib must be positive")
    progress = None if args.quiet else progress_to_stderr
    payload = build_audit(
        args.endomondo_hr,
        args.endomondo_metadata,
        args.goldencheetah_root,
        chunk_size=args.chunk_size_mib * 1024 * 1024,
        progress=progress,
    )
    atomic_write_json(args.output, payload)
    completion = {
        "status": "ok",
        "audit_version": payload["audit_version"],
        "audit_duration_seconds": payload["audit_duration_seconds"],
        "logical_sources": sorted(payload["sources"]),
    }
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
