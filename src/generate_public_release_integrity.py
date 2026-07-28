"""Generate and verify a privacy-conservative public-release integrity manifest.

The default is an explicit allowlist.  Row-level/intermediate data locations are
never traversed, and candidate tabular/JSON files are inspected for direct
identifier, exact-time, geolocation, or source-path fields before hashing.
Only repository-relative paths are written to the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_VERSION = "0.1.0"
DEFAULT_MANIFEST_NAME = "PUBLIC_RELEASE_INTEGRITY_v0_1_0.csv"


class ReleaseSafetyError(RuntimeError):
    """Raised when an allowlisted file violates a public-release safety rule."""


@dataclass(frozen=True)
class AllowRule:
    category: str
    patterns: tuple[str, ...]


DEFAULT_ALLOW_RULES = (
    AllowRule(
        "release_metadata",
        (
            ".gitignore",
            "CITATION.cff",
            "DATA_SOURCES.md",
            "README.md",
            "REPRODUCING.md",
            "ENVIRONMENT.md",
            "LICENSE_DECISION_REQUIRED.md",
            "requirements-lock.txt",
            "PUBLIC_RELEASE_MANIFEST.md",
            "REPOSITORY_UPLOAD_GUIDE.md",
        ),
    ),
    AllowRule("analysis_code", ("src/*.py",)),
    AllowRule("tests", ("tests/*.py",)),
    AllowRule(
        "configuration",
        ("configs/study.yaml",),
    ),
    AllowRule("protocol", ("protocol/*.md",)),
    AllowRule(
        "references",
        (
            "references/references.bib",
            "references/goldencheetah_opendata.bib",
            "references/LITERATURE_SEARCH_REPORT.md",
            "references/PRIOR_WORK_COMPARISON.md",
            "references/TARGETED_LITERATURE_UPDATE_2026-07-23.md",
        ),
    ),
    AllowRule(
        "aggregate_results",
        (
            "outputs/results/ablation_hr_only_interval_v0_14_0.csv",
            "outputs/results/ablation_hr_only_point_v0_14_0.csv",
            "outputs/results/ablation_hr_only_probabilistic_v0_14_0.csv",
            "outputs/results/clustered_calibration_bootstrap_v0_20_0.csv",
            "outputs/results/dense_origin_interval_v0_15_0.csv",
            "outputs/results/dense_origin_point_v0_15_0.csv",
            "outputs/results/external_sport_standardization_v0_20_1.csv",
            "outputs/results/external_sport_uncertainty_bootstrap_v0_21_0.csv",
            "outputs/results/external_sport_uncertainty_standardization_v0_24_0.csv",
            "outputs/results/figure3_uncertainty_bootstrap_v0_18_0.csv",
            "outputs/results/gru_user_generalization_metrics_v0_9_0.csv",
            "outputs/results/history_availability_v0_19_0.csv",
            "outputs/results/horizon_specific_eligibility_v0_29_0.csv",
            "outputs/results/horizon_specific_frozen_model_per_seed_v0_30_0.csv",
            "outputs/results/horizon_specific_frozen_model_summary_v0_30_0.csv",
            "outputs/results/multiseed_balanced_calibration_differences_v0_24_0.csv",
            "outputs/results/multiseed_balanced_calibration_difference_summary_v0_24_0.csv",
            "outputs/results/multiseed_balanced_calibration_per_seed_v0_24_0.csv",
            "outputs/results/multiseed_balanced_calibration_summary_v0_24_0.csv",
            "outputs/results/multiseed_paired_model_comparisons_v0_25_0.csv",
            "outputs/results/multiseed_paired_sport_shift_v0_25_0.csv",
            "outputs/results/matched_sport_availability_v0_27_0.csv",
            "outputs/results/naive_baseline_metrics_v0_5_0.csv",
            "outputs/results/paired_model_comparisons_v0_11_0.csv",
            "outputs/results/persistence_conformal_baseline_v0_26_0.csv",
            "outputs/results/probabilistic_metrics_v0_11_0.csv",
            "outputs/results/recorded_gender_differences_v0_16_0.csv",
            "outputs/results/signal_ablation_paired_v0_14_0.csv",
            "outputs/results/sport_shift_aligned_baselines_v0_12_0.csv",
            "outputs/results/sport_shift_interval_v0_12_0.csv",
            "outputs/results/sport_shift_mae_bootstrap_v0_19_0.csv",
            "outputs/results/sport_shift_point_v0_12_0.csv",
            "outputs/results/sport_shift_uncertainty_bootstrap_v0_17_0.csv",
            "outputs/results/source_shift_characterization_v0_21_0.csv",
            "outputs/results/source_shift_session_distributions_v0_21_0.csv",
            "outputs/results/source_shift_sport_composition_v0_21_0.csv",
            "outputs/results/tcn_user_generalization_metrics_v0_9_0.csv",
            "outputs/results/temporal_aligned_baselines_v0_13_0.csv",
            "outputs/results/temporal_paired_comparisons_v0_13_0.csv",
            "outputs/results/temporal_probabilistic_metrics_v0_13_0.csv",
            "outputs/results/temporal_uncertainty_interval_v0_13_0.csv",
            "outputs/results/temporal_uncertainty_point_v0_13_0.csv",
            "outputs/results/transformer_user_generalization_metrics_v0_9_0.csv",
            "outputs/results/uncertainty_interval_metrics_v0_11_0.csv",
            "outputs/results/uncertainty_point_metrics_v0_11_0.csv",
            "outputs/results/xgboost_user_generalization_metrics_v0_8_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/main_history_difference_summary_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/main_history_seed_paired_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/main_vs_comparator_seed_paired_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/main_vs_comparator_summary_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/per_seed_metrics_long_v0_22_0.csv",
            "outputs/q1_multiseed_v0_21_0/aggregation/seed_variability_summary_v0_22_0.csv",
            "outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrasts_per_seed_v0_23_0.csv",
            "outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrast_seed_summary_v0_23_0.csv",
            "outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrast_user_bootstrap_v0_23_0.csv",
            "outputs/deliberately_leaky_negative_control_v0_28_0/aggregation/paired_metrics_per_seed_v0_28_0.csv",
            "outputs/deliberately_leaky_negative_control_v0_28_0/aggregation/paired_metrics_seed_summary_v0_28_0.csv",
            "outputs/deliberately_leaky_negative_control_v0_28_0/aggregation/paired_user_bootstrap_v0_28_0.csv",
            "outputs/deliberately_leaky_negative_control_v0_28_0/aggregation/interval_diagnostics_per_seed_v0_28_0.csv",
        ),
    ),
    AllowRule(
        "validated_audits",
        (
            "outputs/audit/FINAL_RESULTS_VALIDATION.md",
            "outputs/audit/RAW_SOURCE_INTEGRITY_v0_1_0.json",
            "outputs/audit/REPORTED_NUMBER_VALIDATION.json",
            "outputs/audit/PMEA_SUBMISSION_VALIDATION.json",
            "outputs/audit/persistence_conformal_baseline_v0_26_0.json",
            "outputs/audit/matched_sport_availability_v0_27_0.json",
        ),
    ),
    AllowRule(
        "figures",
        (
            "figures/Figure_1_study_design.pdf",
            "figures/Figure_1_study_design.png",
            "figures/Figure_1_study_design.svg",
            "figures/Figure_2_primary_performance.pdf",
            "figures/Figure_2_primary_performance.png",
            "figures/Figure_2_primary_performance.svg",
            "figures/Figure_3_sport_shift.pdf",
            "figures/Figure_3_sport_shift.png",
            "figures/Figure_3_sport_shift.svg",
            "figures/Figure_4_uncertainty_calibration.pdf",
            "figures/Figure_4_uncertainty_calibration.png",
            "figures/Figure_4_uncertainty_calibration.svg",
            "figures/Supplementary_Figure_1_ablation_sensitivity.pdf",
            "figures/Supplementary_Figure_1_ablation_sensitivity.png",
            "figures/Supplementary_Figure_1_ablation_sensitivity.svg",
            "figures/FIGURE_CONTRACT.md",
            "figures/FIGURE_QA.md",
            "figures/source_data/Figure_2_paired_effect_source.csv",
            "figures/source_data/Figure_2_point_forecast_source.csv",
            "figures/source_data/Figure_3_sport_shift_source.csv",
            "figures/source_data/Figure_3_support_source.csv",
            "figures/source_data/Figure_3_sport_shift_PMEA_source.csv",
            "figures/source_data/Figure_3_sport_shift_PMEA_support_source.csv",
            "figures/source_data/Figure_4_coverage_source.csv",
            "figures/source_data/Figure_4_probabilistic_source.csv",
            "figures/source_data/Figure_4_width_source.csv",
            "figures/source_data/Supplementary_Figure_1_history_source.csv",
            "figures/source_data/Supplementary_Figure_1_signal_source.csv",
            "figures/source_data/Supplementary_Figure_1_stride_source.csv",
        ),
    ),
    AllowRule(
        "manuscript_targets",
        (
            "manuscript/main_manuscript.md",
            "manuscript/supplementary_material.md",
            "manuscript/highlights.txt",
        ),
    ),
)


BLOCKED_PREFIXES = (
    "outputs/manifests/",
    "outputs/origins/",
    "outputs/features/",
    "outputs/predictions/",
    "outputs/models/",
    "outputs/docx_render/",
    "outputs/docx_render_",
    "outputs/docx_render_word/",
    "notebooks/",
    "scripts/",
    "data/raw/",
    "endomondohr.json/",
    "endomondometa.json/",
    "goldencheetah_extracted/",
)
BLOCKED_DIRECTORY_NAMES = {
    ".git",
    ".idea",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "hrf_env",
}
BLOCKED_SUFFIXES = {
    ".fit",
    ".gpx",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".pyc",
    ".sqlite",
    ".tcx",
}


# These working-tree artifacts can expose linkable free text or have unresolved
# provenance/licensing.  They stay blocked even if a future custom allowlist
# accidentally names them explicitly.
SENSITIVE_RELEASE_PATHS = {
    "configs/sport_ontology_v0_2_0.csv",
    "figures/Graphical_Abstract.pdf",
    "figures/Graphical_Abstract.png",
    "figures/Graphical_Abstract.svg",
    "figures/Graphical_Abstract.tiff",
    "figures/source_data/Graphical_Abstract_source.csv",
    "outputs/results/recorded_gender_subgroups_v0_16_0.csv",
    "outputs/audit/external_sport_uncertainty_standardization_v0_24_0.json",
    "outputs/audit/multiseed_balanced_calibration_v0_24_0.json",
    "outputs/audit/multiseed_paired_user_bootstrap_v0_25_0.audit.json",
}

MINIMUM_AGGREGATE_GROUP_SIZE = 10


SENSITIVE_FIELD_NAMES = {
    "activity_id",
    "athlete_id",
    "byte_offset",
    "csv_local_datetime",
    "csv_relative_path",
    "end_timestamp",
    "file_path",
    "gender",
    "latitude",
    "longitude",
    "metadata_ride_index",
    "metadata_utc_datetime",
    "origin_time",
    "record_index",
    "session_id",
    "session_key",
    "source_key",
    "source_path",
    "start_timestamp",
    "timestamp",
    "user_id",
    "user_index",
}
SENSITIVE_FIELD_SUFFIXES = ("_activity_id", "_athlete_id", "_session_id", "_user_id")


TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".txt",
    ".yaml",
    ".yml",
}
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
UNC_PATH = re.compile(r"(?<![\\])\\\\[A-Za-z0-9_.\-$]+[\\/]")
POSIX_HOME_PATH = re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users|mnt)/[A-Za-z0-9_.\-]+/")
FILE_URI = re.compile(r"\bfile://", flags=re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_field(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lstrip("\ufeff").lower()).strip("_")


def is_sensitive_field(value: str) -> bool:
    normalized = normalize_field(value)
    return normalized in SENSITIVE_FIELD_NAMES or normalized.endswith(SENSITIVE_FIELD_SUFFIXES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ReleaseSafetyError(f"file escapes repository root: {path.name}")
    relative = resolved_path.relative_to(resolved_root)
    if path.is_symlink():
        raise ReleaseSafetyError(f"symbolic links are not allowed: {relative.as_posix()}")
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseSafetyError(f"unsafe release path: {relative.as_posix()}")
    return relative.as_posix()


def assert_path_allowed(relative_path: str) -> None:
    lowered = relative_path.lower()
    parts = {part.lower() for part in Path(relative_path).parts}
    if any(lowered.startswith(prefix) for prefix in BLOCKED_PREFIXES):
        raise ReleaseSafetyError(f"blocked release location: {relative_path}")
    if parts & {name.lower() for name in BLOCKED_DIRECTORY_NAMES}:
        raise ReleaseSafetyError(f"blocked generated/private directory: {relative_path}")
    if Path(relative_path).suffix.lower() in BLOCKED_SUFFIXES:
        raise ReleaseSafetyError(f"blocked row-level/binary artifact type: {relative_path}")
    if relative_path in SENSITIVE_RELEASE_PATHS:
        raise ReleaseSafetyError(f"privacy- or rights-blocked release artifact: {relative_path}")


def assert_no_local_absolute_path(path: Path) -> None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return
    text = path.read_text(encoding="utf-8")
    if (
        WINDOWS_ABSOLUTE_PATH.search(text)
        or UNC_PATH.search(text)
        or POSIX_HOME_PATH.search(text)
        or FILE_URI.search(text)
    ):
        raise ReleaseSafetyError(f"local absolute path or file URI found in {path.name}")


def assert_safe_csv_header(path: Path) -> None:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), [])
    sensitive = sorted(field for field in header if is_sensitive_field(field))
    if sensitive:
        names = ", ".join(normalize_field(field) for field in sensitive)
        raise ReleaseSafetyError(f"sensitive tabular field(s) in {path.name}: {names}")


def assert_minimum_group_support(path: Path) -> None:
    """Reject aggregate rows that report a user subgroup smaller than k=10."""

    if path.suffix.lower() not in {".csv", ".tsv"}:
        return
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        support_fields = [
            field
            for field in (reader.fieldnames or [])
            if normalize_field(field) in {"users", "n_users", "user_count"}
            or normalize_field(field).endswith("_users")
        ]
        for row_number, row in enumerate(reader, start=2):
            for field in support_fields:
                raw_value = str(row.get(field, "")).strip()
                if not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if 0 <= value < MINIMUM_AGGREGATE_GROUP_SIZE:
                    raise ReleaseSafetyError(
                        f"aggregate user support below k={MINIMUM_AGGREGATE_GROUP_SIZE} "
                        f"in {path.name} at row {row_number}: {field}={raw_value}"
                    )


def iter_json_keys(value) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from iter_json_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_json_keys(nested)


def assert_safe_json_keys(path: Path) -> None:
    if path.suffix.lower() != ".json":
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    sensitive = sorted({normalize_field(key) for key in iter_json_keys(payload) if is_sensitive_field(key)})
    if sensitive:
        raise ReleaseSafetyError(
            f"sensitive JSON key(s) in {path.name}: {', '.join(sensitive)}"
        )


def inspect_candidate(path: Path, root: Path) -> str:
    relative = relative_posix(path, root)
    assert_path_allowed(relative)
    assert_safe_csv_header(path)
    assert_minimum_group_support(path)
    assert_safe_json_keys(path)
    assert_no_local_absolute_path(path)
    return relative


def collect_entries(
    root: Path,
    rules: tuple[AllowRule, ...] = DEFAULT_ALLOW_RULES,
) -> list[dict[str, object]]:
    root = root.resolve()
    entries_by_path: dict[str, dict[str, object]] = {}
    category_by_path: dict[str, str] = {}

    for rule in rules:
        for pattern in rule.patterns:
            matches = [path for path in sorted(root.glob(pattern)) if path.is_file()]
            if not matches:
                raise ReleaseSafetyError(
                    f"required allowlist pattern has no files: {pattern}"
                )
            for path in matches:
                relative = inspect_candidate(path, root)
                existing_category = category_by_path.get(relative)
                if existing_category and existing_category != rule.category:
                    raise ReleaseSafetyError(
                        f"file assigned to multiple categories: {relative} "
                        f"({existing_category}, {rule.category})"
                    )
                category_by_path[relative] = rule.category
                entries_by_path[relative] = {
                    "manifest_version": MANIFEST_VERSION,
                    "relative_path": relative,
                    "category": rule.category,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }

    if not entries_by_path:
        raise ReleaseSafetyError("allowlist produced no release files")
    return [entries_by_path[path] for path in sorted(entries_by_path)]


def write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "manifest_version",
                "relative_path",
                "category",
                "size_bytes",
                "sha256",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def path_label(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(root.resolve()):
        return resolved.relative_to(root.resolve()).as_posix()
    return path.name


def generate_manifest(root: Path, output: Path, audit: Path) -> dict[str, object]:
    entries = collect_entries(root)
    write_csv_atomic(output, entries)
    category_counts: dict[str, int] = {}
    for entry in entries:
        category = str(entry["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    payload: dict[str, object] = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "manifest_file": path_label(output, root),
        "manifest_sha256": sha256_file(output),
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "category_counts": category_counts,
        "safety_policy": {
            "selection": "explicit default allowlist",
            "paths": "repository-relative only; symlinks and local absolute paths rejected",
            "tabular_guard": "direct identifiers, exact participant times, geolocation, and source paths rejected",
            "minimum_group_support": f"reported user subgroups must have k >= {MINIMUM_AGGREGATE_GROUP_SIZE}",
            "blocked_artifacts": "raw, row-level, model-array, origin, prediction, manifest, and checkpoint locations excluded by default",
        },
    }
    write_json_atomic(audit, payload)
    return payload


def expected_serialized_rows(root: Path) -> list[dict[str, str]]:
    return [
        {key: str(value) for key, value in entry.items()}
        for entry in collect_entries(root)
    ]


def verify_manifest(root: Path, manifest: Path) -> dict[str, object]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        observed = list(csv.DictReader(handle))
    expected = expected_serialized_rows(root)
    if observed != expected:
        observed_by_path = {row.get("relative_path", ""): row for row in observed}
        expected_by_path = {row["relative_path"]: row for row in expected}
        missing = sorted(set(expected_by_path) - set(observed_by_path))
        unexpected = sorted(set(observed_by_path) - set(expected_by_path))
        changed = sorted(
            path
            for path in set(expected_by_path) & set(observed_by_path)
            if expected_by_path[path] != observed_by_path[path]
        )
        raise ReleaseSafetyError(
            "integrity manifest mismatch: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    return {
        "manifest_version": MANIFEST_VERSION,
        "status": "PASS",
        "manifest_file": path_label(manifest, root),
        "manifest_sha256": sha256_file(manifest),
        "file_count": len(expected),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in expected),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generate or verify the privacy-conservative public-release integrity manifest."
    )
    result.add_argument("--root", type=Path, default=ROOT)
    result.add_argument("--output", type=Path)
    result.add_argument("--audit", type=Path)
    result.add_argument("--verify", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    root = args.root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root / "release" / DEFAULT_MANIFEST_NAME
    )
    audit = (
        args.audit.resolve()
        if args.audit
        else output.with_suffix(".audit.json")
    )
    payload = verify_manifest(root, output) if args.verify else generate_manifest(root, output, audit)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
