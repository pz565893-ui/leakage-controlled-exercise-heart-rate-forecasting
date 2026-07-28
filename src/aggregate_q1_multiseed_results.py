from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AGGREGATION_VERSION = "0.22.0"
SOURCE_EXPERIMENT_VERSION = "0.21.0"
HORIZONS = (60, 180, 300)
CANONICAL_AGGREGATION = (
    "origin-within-session, session-within-user, equal-user mean"
)
IDENTIFIER_COLUMNS = {
    "model_version",
    "analysis_version",
    "seed",
    "protocol",
    "regime",
    "mode",
    "model",
    "horizon_seconds",
    "nominal_coverage",
    "calibrated",
    "users",
    "sessions",
    "origins",
    "aggregation",
    "held_sport_code",
    "held_sport_family",
    "sport_family",
}
NONNEGATIVE_METRICS = {
    "mae_bpm",
    "rmse_bpm",
    "absolute_coverage_error",
    "mean_interval_width_bpm",
    "mean_width_bpm",
    "conformal_adjustment_bpm",
    "wis_bpm",
    "weighted_interval_score_bpm",
    "interval_score_bpm",
}
UNIT_INTERVAL_METRICS = {
    "picp",
    "coverage",
    "absolute_coverage_error",
    "coverage_error",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_layout() -> dict[str, dict[str, object]]:
    common_point = {
        "glob": "point_metrics.csv",
        "kind": "point",
        "phase": "evaluation",
        "required": True,
    }
    common_audit = {
        "glob": "audit.json",
        "kind": "audit",
        "role": "general",
        "required": True,
    }
    optional_interval = {
        "glob": "interval_metrics.csv",
        "kind": "interval",
        "phase": "evaluation",
        "required": False,
    }
    return {
        "unseen_main": {
            "seed_group": "primary_models",
            "directory_pattern": "seed_{seed}/unseen_main",
            "model": "history_quantile_tcn",
            "role": "main",
            "files": {
                "development_point": {
                    "glob": "development_point_metrics.csv",
                    "kind": "point",
                    "phase": "development",
                    "required": True,
                    "required_slices": [
                        {
                            "regimes": [
                                "unseen_user_validation",
                                "unseen_user_test",
                            ],
                            "modes": ["history_informed", "zero_history"],
                        }
                    ],
                },
                "development_interval": {
                    "glob": "development_interval_metrics.csv",
                    "kind": "interval",
                    "phase": "development",
                    "required": True,
                },
                "development_audit": {
                    "glob": "development_audit.json",
                    "kind": "audit",
                    "role": "development",
                    "required": True,
                },
                "freeze_record": {
                    "glob": "freeze_record.json",
                    "kind": "freeze",
                    "role": "freeze",
                    "required": True,
                },
                "external_point": {
                    "glob": "external_point_metrics.csv",
                    "kind": "point",
                    "phase": "external",
                    "required": True,
                    "required_slices": [
                        {
                            "regimes": ["goldencheetah_frozen_external"],
                            "modes": ["zero_history"],
                        }
                    ],
                },
                "external_interval": {
                    "glob": "external_interval_metrics.csv",
                    "kind": "interval",
                    "phase": "external",
                    "required": True,
                },
                "external_audit": {
                    "glob": "external_audit.json",
                    "kind": "audit",
                    "role": "external",
                    "required": True,
                },
            },
        },
        "unseen_gru": {
            "seed_group": "learned_comparators",
            "directory_pattern": "seed_{seed}/unseen_gru",
            "model": "gru",
            "role": "comparator",
            "files": {
                "point": {
                    **common_point,
                    "required_slices": [
                        {
                            "regimes": [
                                "unseen_user_validation",
                                "unseen_user_test",
                                "goldencheetah_frozen_external",
                            ]
                        }
                    ],
                },
                "audit": common_audit,
                "interval": optional_interval,
            },
        },
        "unseen_tcn": {
            "seed_group": "learned_comparators",
            "directory_pattern": "seed_{seed}/unseen_tcn",
            "model": "tcn",
            "role": "comparator",
            "files": {
                "point": {
                    **common_point,
                    "required_slices": [
                        {
                            "regimes": [
                                "unseen_user_validation",
                                "unseen_user_test",
                                "goldencheetah_frozen_external",
                            ]
                        }
                    ],
                },
                "audit": common_audit,
                "interval": optional_interval,
            },
        },
        "temporal_main": {
            "seed_group": "primary_models",
            "directory_pattern": "seed_{seed}/temporal_main",
            "model": "history_quantile_tcn",
            "role": "main",
            "files": {
                "point": {
                    **common_point,
                    "required_slices": [
                        {
                            "regimes": ["within_user_temporal_test"],
                            "modes": ["history_informed", "zero_history"],
                        }
                    ],
                },
                "audit": common_audit,
                "interval": optional_interval,
            },
        },
        "temporal_gru": {
            "seed_group": "learned_comparators",
            "directory_pattern": "seed_{seed}/temporal_gru",
            "model": "gru",
            "role": "comparator",
            "files": {
                "point": {
                    **common_point,
                    "required_slices": [
                        {"regimes": ["within_user_temporal_test"]}
                    ],
                },
                "audit": common_audit,
                "interval": optional_interval,
            },
        },
        "temporal_tcn": {
            "seed_group": "learned_comparators",
            "directory_pattern": "seed_{seed}/temporal_tcn",
            "model": "tcn",
            "role": "comparator",
            "files": {
                "point": {
                    **common_point,
                    "required_slices": [
                        {"regimes": ["within_user_temporal_test"]}
                    ],
                },
                "audit": common_audit,
                "interval": optional_interval,
            },
        },
        "held_sport": {
            "seed_group": "held_sport_models",
            "family_group": "held_sport_main.sport_families",
            "directory_pattern": "seed_{seed}/held_sport/{family}",
            "model": "history_quantile_tcn",
            "role": "main",
            "files": {
                "point": {
                    **common_point,
                    "required_slices": [
                        {
                            "regimes": [
                                "unseen_sport__{family}",
                                "joint_user_sport__{family}",
                            ],
                            "modes": ["history_informed", "zero_history"],
                        }
                    ],
                },
                "audit": common_audit,
                "interval": optional_interval,
            },
        },
    }


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_layout(path: Path | None) -> dict[str, dict[str, object]]:
    layout: dict[str, Any] = default_layout()
    if path is not None:
        override = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(override, dict), "layout override must be a JSON object")
        experiments = override.get("experiments", override)
        require(isinstance(experiments, dict), "layout experiments must be an object")
        layout = deep_merge(layout, experiments)
    layout = {
        name: value
        for name, value in layout.items()
        if bool(value.get("enabled", True))
    }
    require(bool(layout), "no enabled experiments in layout")
    for name, specification in layout.items():
        for field in ["seed_group", "directory_pattern", "model", "role", "files"]:
            require(field in specification, f"{name}: layout missing {field}")
        require(isinstance(specification["files"], dict), f"{name}: files")
    return layout


def nested_get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for piece in dotted.split("."):
        if not isinstance(current, dict) or piece not in current:
            raise KeyError(dotted)
        current = current[piece]
    return current


@dataclass(frozen=True)
class ExpectedJob:
    job_id: str
    experiment: str
    seed: int
    family: str
    model: str
    role: str
    directory: Path
    files: dict[str, dict[str, object]]


def build_expected_jobs(
    config: dict[str, Any],
    layout: dict[str, dict[str, object]],
    root: Path,
) -> list[ExpectedJob]:
    seed_config = config.get("seeds")
    require(isinstance(seed_config, dict), "configuration has no seeds object")
    jobs: list[ExpectedJob] = []
    for experiment, specification in layout.items():
        seed_group = str(specification["seed_group"])
        require(seed_group in seed_config, f"missing seed group {seed_group}")
        seeds = [int(seed) for seed in seed_config[seed_group]]
        require(len(seeds) == len(set(seeds)) and seeds, f"invalid {seed_group}")
        if "family_group" in specification:
            families = [str(value) for value in nested_get(
                config, str(specification["family_group"])
            )]
            require(families and len(families) == len(set(families)), "invalid families")
        else:
            families = [""]
        for seed in seeds:
            for family in families:
                relative = str(specification["directory_pattern"]).format(
                    seed=seed, family=family
                )
                job_id = relative.replace("\\", "/")
                jobs.append(
                    ExpectedJob(
                        job_id=job_id,
                        experiment=experiment,
                        seed=seed,
                        family=family,
                        model=str(specification["model"]),
                        role=str(specification["role"]),
                        directory=root / relative,
                        files=dict(specification["files"]),
                    )
                )
    require(len(jobs) == len({job.job_id for job in jobs}), "duplicate expected jobs")
    return jobs


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def resolve_files(
    job: ExpectedJob, root: Path
) -> tuple[dict[str, Path], list[dict[str, object]], list[str]]:
    resolved: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for key, specification in job.files.items():
        pattern = str(specification["glob"])
        matches = sorted(path for path in job.directory.glob(pattern) if path.is_file())
        required = bool(specification.get("required", False))
        if len(matches) > 1:
            errors.append(f"{key}: glob matched {len(matches)} files")
        elif len(matches) == 1:
            path = matches[0]
            resolved[key] = path
            records.append(
                {
                    "key": key,
                    "glob": pattern,
                    "required": required,
                    "status": "present",
                    "path": relative_path(path, root),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            records.append(
                {
                    "key": key,
                    "glob": pattern,
                    "required": required,
                    "status": "missing" if required else "optional_absent",
                    "path": None,
                    "bytes": None,
                    "sha256": None,
                }
            )
    return resolved, records, errors


def load_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{label}: JSON root is not an object")
    return payload


def extract_seed(payload: dict[str, Any]) -> int | None:
    candidates = [payload.get("seed")]
    resolved = payload.get("resolved_training_parameters")
    if isinstance(resolved, dict):
        candidates.append(resolved.get("seed"))
    for value in candidates:
        if value is not None:
            return int(value)
    return None


def validate_job_json(
    job: ExpectedJob,
    resolved: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for key, specification in job.files.items():
        if key not in resolved or specification.get("kind") not in {"audit", "freeze"}:
            continue
        payload = load_json(resolved[key], f"{job.job_id}/{key}")
        payloads[key] = payload
        observed_seed = extract_seed(payload)
        require(observed_seed is not None, f"{job.job_id}/{key}: audit seed missing")
        require(observed_seed == job.seed, f"{job.job_id}/{key}: seed mismatch")
        if specification.get("kind") == "audit":
            require(
                payload.get("all_assertions_pass") is True,
                f"{job.job_id}/{key}: assertions did not pass",
            )
        observed_model = payload.get("model")
        if observed_model is not None:
            require(
                str(observed_model) == job.model,
                f"{job.job_id}/{key}: model mismatch",
            )
        observed_family = payload.get("held_sport_family")
        if job.family and observed_family is not None:
            require(
                str(observed_family) == job.family,
                f"{job.job_id}/{key}: family mismatch",
            )
    if job.experiment == "unseen_main":
        development = payloads["development_audit"]
        freeze = payloads["freeze_record"]
        external = payloads["external_audit"]
        require(
            development.get("development_only") is True,
            f"{job.job_id}: development-only flag missing",
        )
        require(
            development.get("external_inference_performed") is False,
            f"{job.job_id}: external inference contaminated development audit",
        )
        require(
            freeze.get("status") == "frozen_before_external_inference",
            f"{job.job_id}: invalid freeze status",
        )
        require(
            freeze.get("external_outcomes_used_for_selection") is False,
            f"{job.job_id}: external outcomes used for selection",
        )
        require(
            freeze.get("external_adaptation_or_recalibration_allowed") is False,
            f"{job.job_id}: freeze allowed external adaptation",
        )
        require(
            external.get("external_adaptation_or_recalibration") is False,
            f"{job.job_id}: external adaptation/recalibration occurred",
        )
        require(
            external.get("freeze_record_sha256") == sha256_file(resolved["freeze_record"]),
            f"{job.job_id}: external audit freeze hash mismatch",
        )
        development_record = freeze.get("development_audit")
        require(
            isinstance(development_record, dict)
            and development_record.get("sha256")
            == sha256_file(resolved["development_audit"]),
            f"{job.job_id}: frozen development audit hash mismatch",
        )
        output_record = external.get("outputs")
        require(isinstance(output_record, dict), f"{job.job_id}: external outputs missing")
        for file_key, hash_key in [
            ("external_point", "point_metrics_sha256"),
            ("external_interval", "interval_metrics_sha256"),
        ]:
            require(
                output_record.get(hash_key) == sha256_file(resolved[file_key]),
                f"{job.job_id}: {file_key} hash mismatch",
            )
    return payloads


def normalize_bool(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return "true"
    if text in {"false", "0"}:
        return "false"
    raise ValueError(f"invalid boolean value: {value}")


def metric_columns(frame: pd.DataFrame, kind: str) -> list[str]:
    candidates = [
        column
        for column in frame.columns
        if column not in IDENTIFIER_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]
    if kind == "point":
        required = {"mae_bpm", "rmse_bpm", "bias_bpm"}
        require(required.issubset(candidates), f"point CSV missing {required - set(candidates)}")
    else:
        require("picp" in candidates, "interval CSV missing picp")
    require(bool(candidates), "metric CSV has no numeric metrics")
    return candidates


def validate_metric_values(metric: str, values: np.ndarray) -> None:
    require(np.isfinite(values).all(), f"{metric}: non-finite values")
    if metric in NONNEGATIVE_METRICS:
        require(bool((values >= 0).all()), f"{metric}: negative values")
    if metric in UNIT_INTERVAL_METRICS:
        require(bool(((values >= 0) & (values <= 1)).all()), f"{metric}: outside [0,1]")


def validate_required_slices(
    frame: pd.DataFrame,
    slices: list[dict[str, object]],
    family: str,
) -> None:
    for specification in slices:
        regimes = [str(value).format(family=family) for value in specification["regimes"]]
        modes = [str(value) for value in specification.get("modes", [])]
        for regime in regimes:
            for horizon in HORIZONS:
                if modes:
                    for mode in modes:
                        mask = (
                            frame["regime"].astype(str).eq(regime)
                            & frame["horizon_seconds"].eq(horizon)
                            & frame["mode"].astype(str).eq(mode)
                        )
                        require(
                            int(mask.sum()) == 1,
                            f"missing/duplicate required slice {regime}/{mode}/{horizon}",
                        )
                else:
                    mask = frame["regime"].astype(str).eq(regime) & frame[
                        "horizon_seconds"
                    ].eq(horizon)
                    require(
                        int(mask.sum()) == 1,
                        f"missing/duplicate required slice {regime}/{horizon}",
                    )


def normalize_metric_file(
    *,
    job: ExpectedJob,
    key: str,
    specification: dict[str, object],
    path: Path,
    root: Path,
) -> tuple[list[dict[str, object]], int]:
    kind = str(specification["kind"])
    frame = pd.read_csv(path)
    require(not frame.empty, f"{job.job_id}/{key}: empty CSV")
    required_columns = {"regime", "horizon_seconds", "users", "sessions", "origins"}
    require(
        required_columns.issubset(frame.columns),
        f"{job.job_id}/{key}: missing {required_columns - set(frame.columns)}",
    )
    frame["horizon_seconds"] = pd.to_numeric(
        frame["horizon_seconds"], errors="raise"
    ).astype(int)
    require(
        set(frame["horizon_seconds"]).issubset(HORIZONS),
        f"{job.job_id}/{key}: unexpected horizon",
    )
    for support in ["users", "sessions", "origins"]:
        numeric = pd.to_numeric(frame[support], errors="raise")
        require(bool((numeric > 0).all()), f"{job.job_id}/{key}: invalid {support}")
        frame[support] = numeric.astype(int)
    if "seed" in frame.columns:
        observed = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
        require(observed == {job.seed}, f"{job.job_id}/{key}: CSV seed mismatch")
    if "model" in frame.columns:
        require(
            set(frame["model"].astype(str)) == {job.model},
            f"{job.job_id}/{key}: CSV model mismatch",
        )
    if job.family and "held_sport_family" in frame.columns:
        require(
            set(frame["held_sport_family"].astype(str)) == {job.family},
            f"{job.job_id}/{key}: CSV family mismatch",
        )
    if "mode" not in frame.columns:
        frame["mode"] = "not_applicable"
    if "nominal_coverage" not in frame.columns:
        frame["nominal_coverage"] = np.nan
    if "calibrated" not in frame.columns:
        frame["calibrated"] = np.nan
    if specification.get("required_slices"):
        validate_required_slices(
            frame,
            list(specification["required_slices"]),
            job.family,
        )
    metrics = metric_columns(frame, kind)
    for metric in metrics:
        values = pd.to_numeric(frame[metric], errors="raise").to_numpy(dtype=np.float64)
        validate_metric_values(metric, values)
    source_hash = sha256_file(path)
    source_path = relative_path(path, root)
    rows: list[dict[str, object]] = []
    for _, record in frame.iterrows():
        coverage = record.get("nominal_coverage")
        coverage_value: object = "" if pd.isna(coverage) else float(coverage)
        calibrated = normalize_bool(record.get("calibrated"))
        aggregation = record.get("aggregation", CANONICAL_AGGREGATION)
        if pd.isna(aggregation) or not str(aggregation).strip():
            aggregation = CANONICAL_AGGREGATION
        for metric in metrics:
            rows.append(
                {
                    "aggregation_version": AGGREGATION_VERSION,
                    "seed": job.seed,
                    "experiment": job.experiment,
                    "family": job.family,
                    "phase": str(specification.get("phase", "evaluation")),
                    "source_kind": kind,
                    "model": job.model,
                    "source_model_version": (
                        ""
                        if pd.isna(record.get("model_version"))
                        else str(record.get("model_version"))
                    ),
                    "source_analysis_version": (
                        ""
                        if pd.isna(record.get("analysis_version"))
                        else str(record.get("analysis_version"))
                    ),
                    "protocol": (
                        job.experiment
                        if pd.isna(record.get("protocol"))
                        else str(record.get("protocol"))
                    ),
                    "regime": str(record["regime"]),
                    "mode": str(record["mode"]),
                    "horizon_seconds": int(record["horizon_seconds"]),
                    "nominal_coverage": coverage_value,
                    "calibrated": calibrated,
                    "metric": metric,
                    "value": float(record[metric]),
                    "users": int(record["users"]),
                    "sessions": int(record["sessions"]),
                    "origins": int(record["origins"]),
                    "aggregation": str(aggregation),
                    "source_file": source_path,
                    "source_sha256": source_hash,
                }
            )
    key_columns = [
        "seed",
        "experiment",
        "family",
        "phase",
        "source_kind",
        "model",
        "regime",
        "mode",
        "horizon_seconds",
        "nominal_coverage",
        "calibrated",
        "metric",
    ]
    normalized = pd.DataFrame(rows)
    require(
        not normalized.duplicated(key_columns).any(),
        f"{job.job_id}/{key}: duplicate normalized metric key",
    )
    return rows, int(len(frame))


def validate_audit_row_counts(
    job: ExpectedJob,
    payloads: dict[str, dict[str, Any]],
    csv_rows: dict[str, int],
) -> None:
    if job.experiment == "unseen_main":
        development = payloads["development_audit"]
        if "point_metric_rows" in development:
            require(
                int(development["point_metric_rows"]) == csv_rows["development_point"],
                f"{job.job_id}: development point row count mismatch",
            )
        if "uncertainty_metric_rows" in development:
            require(
                int(development["uncertainty_metric_rows"])
                == csv_rows["development_interval"],
                f"{job.job_id}: development interval row count mismatch",
            )
        return
    audit = payloads.get("audit")
    if audit is None:
        return
    point_key = "point"
    for audit_key in ["point_metric_rows", "metric_rows"]:
        if audit_key in audit and point_key in csv_rows:
            require(
                int(audit[audit_key]) == csv_rows[point_key],
                f"{job.job_id}: point row count mismatch",
            )
            break
    if "interval_metric_rows" in audit and "interval" in csv_rows:
        require(
            int(audit["interval_metric_rows"]) == csv_rows["interval"],
            f"{job.job_id}: interval row count mismatch",
        )


def inspect_job(
    job: ExpectedJob,
    root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    resolved, files, errors = resolve_files(job, root)
    missing_required = [
        str(record["key"])
        for record in files
        if record["required"] and record["status"] != "present"
    ]
    present_count = sum(record["status"] == "present" for record in files)
    metric_rows: list[dict[str, object]] = []
    csv_rows: dict[str, int] = {}
    audit_seeds: dict[str, int] = {}
    if not missing_required and not errors:
        try:
            payloads = validate_job_json(job, resolved)
            audit_seeds = {
                key: int(extract_seed(payload))
                for key, payload in payloads.items()
                if extract_seed(payload) is not None
            }
            for key, specification in job.files.items():
                if key not in resolved or specification.get("kind") not in {
                    "point",
                    "interval",
                }:
                    continue
                normalized, row_count = normalize_metric_file(
                    job=job,
                    key=key,
                    specification=specification,
                    path=resolved[key],
                    root=root,
                )
                metric_rows.extend(normalized)
                csv_rows[key] = row_count
            validate_audit_row_counts(job, payloads, csv_rows)
        except Exception as error:  # progress manifest must survive partial invalid jobs
            errors.append(f"{type(error).__name__}: {error}")
            metric_rows = []
    if errors:
        status = "invalid"
    elif missing_required:
        status = "incomplete" if job.directory.exists() or present_count else "pending"
    else:
        status = "complete"
    record: dict[str, object] = {
        "job_id": job.job_id,
        "experiment": job.experiment,
        "seed": job.seed,
        "family": job.family or None,
        "model": job.model,
        "role": job.role,
        "status": status,
        "directory_exists": job.directory.exists(),
        "required_files": sum(bool(value.get("required")) for value in job.files.values()),
        "present_files": present_count,
        "missing_required_files": missing_required,
        "csv_rows": csv_rows,
        "normalized_metric_rows": len(metric_rows),
        "audit_seeds": audit_seeds,
        "errors": errors,
        "files": files,
    }
    return record, metric_rows


SUMMARY_GROUP_COLUMNS = [
    "experiment",
    "family",
    "phase",
    "source_kind",
    "model",
    "source_model_version",
    "source_analysis_version",
    "protocol",
    "regime",
    "mode",
    "horizon_seconds",
    "nominal_coverage",
    "calibrated",
    "metric",
    "aggregation",
]


def summarize_seed_variability(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(long_rows)
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(SUMMARY_GROUP_COLUMNS, sort=True, dropna=False):
        seeds = sorted(int(value) for value in group["seed"].unique())
        require(len(group) == len(seeds), f"duplicate seed metrics for {key}")
        for support in ["users", "sessions", "origins"]:
            require(group[support].nunique() == 1, f"support changed across seeds: {key}")
        values = group["value"].to_numpy(dtype=np.float64)
        row = {
            "aggregation_version": AGGREGATION_VERSION,
            **dict(zip(SUMMARY_GROUP_COLUMNS, key)),
            "n_seeds": len(seeds),
            "seeds": ";".join(str(seed) for seed in seeds),
            "value_median": float(np.median(values)),
            "value_minimum": float(np.min(values)),
            "value_maximum": float(np.max(values)),
            "users": int(group["users"].iloc[0]),
            "sessions": int(group["sessions"].iloc[0]),
            "origins": int(group["origins"].iloc[0]),
            "seed_inferential_test": False,
        }
        rows.append(row)
    return rows


def main_history_pairs(
    long_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.DataFrame(long_rows)
    selected = frame.loc[
        (frame["source_kind"] == "point")
        & (frame["model"] == "history_quantile_tcn")
        & (frame["metric"] == "mae_bpm")
        & frame["mode"].isin(["history_informed", "zero_history"])
    ].copy()
    identity = [
        "seed",
        "experiment",
        "family",
        "phase",
        "regime",
        "horizon_seconds",
    ]
    require(
        not selected.duplicated(identity + ["mode"]).any(),
        "duplicate main history metric key",
    )
    complete_pair_keys: list[tuple[object, ...]] = []
    for key, group in selected.groupby(identity, sort=False, dropna=False):
        if set(group["mode"]) != {"history_informed", "zero_history"}:
            continue
        for support in ["users", "sessions", "origins"]:
            require(
                group[support].nunique() == 1,
                f"history support mismatch for {key}: {support}",
            )
        complete_pair_keys.append(key if isinstance(key, tuple) else (key,))
    pivot = selected.pivot_table(
        index=identity,
        columns="mode",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.dropna(subset=["history_informed", "zero_history"])
    require(
        len(pivot) == len(complete_pair_keys),
        "main history pairing lost a complete mode pair",
    )
    support = (
        selected.groupby(identity, sort=False, dropna=False)[
            ["users", "sessions", "origins"]
        ]
        .first()
        .reset_index()
    )
    pivot = pivot.merge(support, on=identity, how="left", validate="one_to_one")
    per_seed: list[dict[str, object]] = []
    for _, record in pivot.sort_values(identity).iterrows():
        per_seed.append(
            {
                "aggregation_version": AGGREGATION_VERSION,
                "seed": int(record["seed"]),
                "experiment": str(record["experiment"]),
                "family": str(record["family"]),
                "phase": str(record["phase"]),
                "regime": str(record["regime"]),
                "horizon_seconds": int(record["horizon_seconds"]),
                "history_informed_mae_bpm": float(record["history_informed"]),
                "zero_history_mae_bpm": float(record["zero_history"]),
                "history_minus_zero_mae_bpm": float(
                    record["history_informed"] - record["zero_history"]
                ),
                "users": int(record["users"]),
                "sessions": int(record["sessions"]),
                "origins": int(record["origins"]),
                "difference_direction": "negative favors history-informed",
                "seed_inferential_test": False,
            }
        )
    per_seed_frame = pd.DataFrame(per_seed)
    group_columns = [
        "experiment",
        "family",
        "phase",
        "regime",
        "horizon_seconds",
    ]
    summaries: list[dict[str, object]] = []
    for key, group in per_seed_frame.groupby(group_columns, sort=True, dropna=False):
        seeds = sorted(int(value) for value in group["seed"])
        require(len(seeds) == len(set(seeds)), f"duplicate history seed pair: {key}")
        difference = group["history_minus_zero_mae_bpm"].to_numpy()
        summaries.append(
            {
                "aggregation_version": AGGREGATION_VERSION,
                **dict(zip(group_columns, key)),
                "n_seed_pairs": len(seeds),
                "paired_seeds": ";".join(str(seed) for seed in seeds),
                "history_mae_median_bpm": float(
                    np.median(group["history_informed_mae_bpm"])
                ),
                "zero_history_mae_median_bpm": float(
                    np.median(group["zero_history_mae_bpm"])
                ),
                "difference_median_bpm": float(np.median(difference)),
                "difference_minimum_bpm": float(np.min(difference)),
                "difference_maximum_bpm": float(np.max(difference)),
                "difference_direction": "negative favors history-informed",
                "seed_inferential_test": False,
            }
        )
    return per_seed, summaries


def comparator_pairs(
    long_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.DataFrame(long_rows)
    point = frame.loc[
        (frame["source_kind"] == "point") & (frame["metric"] == "mae_bpm")
    ].copy()
    comparisons = [
        ("unseen_main", "unseen_gru", "gru"),
        ("unseen_main", "unseen_tcn", "tcn"),
        ("temporal_main", "temporal_gru", "gru"),
        ("temporal_main", "temporal_tcn", "tcn"),
    ]
    rows: list[dict[str, object]] = []
    for main_experiment, comparator_experiment, comparator_model in comparisons:
        main = point.loc[
            (point["experiment"] == main_experiment)
            & (point["model"] == "history_quantile_tcn")
            & (point["mode"] == "zero_history")
        ].copy()
        comparator = point.loc[
            (point["experiment"] == comparator_experiment)
            & (point["model"] == comparator_model)
        ].copy()
        merge_keys = ["seed", "regime", "horizon_seconds"]
        merged = main.merge(
            comparator,
            on=merge_keys,
            how="inner",
            suffixes=("_main", "_comparator"),
            validate="one_to_one",
        )
        for _, record in merged.sort_values(merge_keys).iterrows():
            for support in ["users", "sessions", "origins"]:
                require(
                    int(record[f"{support}_main"])
                    == int(record[f"{support}_comparator"]),
                    f"{main_experiment}/{comparator_model}: {support} mismatch",
                )
            rows.append(
                {
                    "aggregation_version": AGGREGATION_VERSION,
                    "seed": int(record["seed"]),
                    "main_experiment": main_experiment,
                    "comparator_experiment": comparator_experiment,
                    "comparator_model": comparator_model,
                    "regime": str(record["regime"]),
                    "horizon_seconds": int(record["horizon_seconds"]),
                    "main_mode": "zero_history",
                    "main_mae_bpm": float(record["value_main"]),
                    "comparator_mae_bpm": float(record["value_comparator"]),
                    "main_minus_comparator_mae_bpm": float(
                        record["value_main"] - record["value_comparator"]
                    ),
                    "users": int(record["users_main"]),
                    "sessions": int(record["sessions_main"]),
                    "origins": int(record["origins_main"]),
                    "difference_direction": "negative favors main zero-history model",
                    "seed_inferential_test": False,
                }
            )
    frame_pairs = pd.DataFrame(rows)
    group_columns = [
        "main_experiment",
        "comparator_experiment",
        "comparator_model",
        "regime",
        "horizon_seconds",
    ]
    summaries: list[dict[str, object]] = []
    for key, group in frame_pairs.groupby(group_columns, sort=True):
        seeds = sorted(int(value) for value in group["seed"])
        require(len(seeds) == len(set(seeds)), f"duplicate comparator seed pair: {key}")
        difference = group["main_minus_comparator_mae_bpm"].to_numpy()
        summaries.append(
            {
                "aggregation_version": AGGREGATION_VERSION,
                **dict(zip(group_columns, key)),
                "main_mode": "zero_history",
                "n_seed_pairs": len(seeds),
                "paired_seeds": ";".join(str(seed) for seed in seeds),
                "main_mae_median_bpm": float(np.median(group["main_mae_bpm"])),
                "comparator_mae_median_bpm": float(
                    np.median(group["comparator_mae_bpm"])
                ),
                "difference_median_bpm": float(np.median(difference)),
                "difference_minimum_bpm": float(np.min(difference)),
                "difference_maximum_bpm": float(np.max(difference)),
                "difference_direction": "negative favors main zero-history model",
                "seed_inferential_test": False,
            }
        )
    return rows, summaries


def expected_seed_count(
    experiment: str, config: dict[str, Any], layout: dict[str, dict[str, object]]
) -> int:
    seed_group = str(layout[experiment]["seed_group"])
    return len(config["seeds"][seed_group])


def validate_complete_summaries(
    *,
    variability: list[dict[str, object]],
    history_summary: list[dict[str, object]],
    comparator_summary: list[dict[str, object]],
    config: dict[str, Any],
    layout: dict[str, dict[str, object]],
) -> None:
    for row in variability:
        require(
            int(row["n_seeds"])
            == expected_seed_count(str(row["experiment"]), config, layout),
            f"incomplete variability seed set: {row['experiment']}",
        )
    for row in history_summary:
        require(
            int(row["n_seed_pairs"])
            == expected_seed_count(str(row["experiment"]), config, layout),
            f"incomplete history seed pairs: {row['experiment']}",
        )
    for row in comparator_summary:
        expected = expected_seed_count(
            str(row["comparator_experiment"]), config, layout
        )
        require(
            int(row["n_seed_pairs"]) == expected,
            f"incomplete comparator seed pairs: {row['comparator_experiment']}",
        )


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "per_seed_long": output_dir / "per_seed_metrics_long_v0_22_0.csv",
        "variability_summary": output_dir / "seed_variability_summary_v0_22_0.csv",
        "history_per_seed": output_dir / "main_history_seed_paired_v0_22_0.csv",
        "history_summary": output_dir / "main_history_difference_summary_v0_22_0.csv",
        "comparator_per_seed": output_dir
        / "main_vs_comparator_seed_paired_v0_22_0.csv",
        "comparator_summary": output_dir
        / "main_vs_comparator_summary_v0_22_0.csv",
        "audit": output_dir / "aggregation_audit_v0_22_0.json",
        "progress": output_dir / "progress_manifest.json",
    }


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    config = load_json(args.config, "configuration")
    require(
        str(config.get("analysis_version")) == SOURCE_EXPERIMENT_VERSION,
        "source configuration version mismatch",
    )
    layout = load_layout(args.layout_config)
    jobs = build_expected_jobs(config, layout, args.root)
    paths = output_paths(args.output_dir)
    job_records: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for job in jobs:
        record, rows = inspect_job(job, args.root)
        job_records.append(record)
        long_rows.extend(rows)
    counts = {
        status: sum(record["status"] == status for record in job_records)
        for status in ["pending", "incomplete", "invalid", "complete"]
    }
    complete = counts["complete"] == len(jobs)
    queue_manifest = args.root / "queue_manifest.json"
    progress: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "aggregation_version": AGGREGATION_VERSION,
        "source_experiment_version": SOURCE_EXPERIMENT_VERSION,
        "status": "complete" if complete else "in_progress",
        "all_expected_jobs_complete": complete,
        "expected_jobs": len(jobs),
        "status_counts": counts,
        "configuration": {
            "path": args.config.as_posix(),
            "sha256": sha256_file(args.config),
        },
        "layout": {
            "source": (
                args.layout_config.as_posix()
                if args.layout_config is not None
                else "built-in default layout"
            ),
            "canonical_sha256": sha256_json(layout),
        },
        "queue_manifest": {
            "present": queue_manifest.exists(),
            "sha256": sha256_file(queue_manifest) if queue_manifest.exists() else None,
        },
        "final_artifacts_emitted": False,
        "existing_final_artifacts_before_run": [
            path.name
            for key, path in paths.items()
            if key not in {"progress", "audit"} and path.exists()
        ],
        "statistical_policy": {
            "seeds_are_independent_participants": False,
            "hypothesis_tests_over_seeds": False,
            "confidence_intervals_over_seeds": False,
            "reported_seed_summaries": ["per-seed", "median", "minimum", "maximum"],
        },
        "jobs": job_records,
    }
    if not complete:
        progress["final_artifact_note"] = (
            "Final aggregation tables were not written because one or more expected "
            "jobs are pending, incomplete, or invalid. Any pre-existing final tables "
            "must be treated as stale until this manifest reports complete."
        )
        atomic_json(paths["progress"], progress)
        return progress

    require(bool(long_rows), "complete jobs produced no metric rows")
    long_frame = pd.DataFrame(long_rows)
    long_key = [
        "seed",
        "experiment",
        "family",
        "phase",
        "source_kind",
        "model",
        "source_model_version",
        "source_analysis_version",
        "protocol",
        "regime",
        "mode",
        "horizon_seconds",
        "nominal_coverage",
        "calibrated",
        "metric",
    ]
    require(not long_frame.duplicated(long_key).any(), "duplicate global long-table key")
    long_rows = long_frame.sort_values(long_key).to_dict(orient="records")
    variability = summarize_seed_variability(long_rows)
    history_per_seed, history_summary = main_history_pairs(long_rows)
    comparator_per_seed, comparator_summary = comparator_pairs(long_rows)
    require(history_per_seed and history_summary, "no main history pairs")
    require(comparator_per_seed and comparator_summary, "no comparator pairs")
    validate_complete_summaries(
        variability=variability,
        history_summary=history_summary,
        comparator_summary=comparator_summary,
        config=config,
        layout=layout,
    )
    tables = {
        "per_seed_long": long_rows,
        "variability_summary": variability,
        "history_per_seed": history_per_seed,
        "history_summary": history_summary,
        "comparator_per_seed": comparator_per_seed,
        "comparator_summary": comparator_summary,
    }
    for key, rows in tables.items():
        atomic_csv(paths[key], rows)
    input_files: dict[str, str] = {}
    for record in job_records:
        for file_record in record["files"]:
            if file_record["status"] == "present":
                input_files[str(file_record["path"])] = str(file_record["sha256"])
    output_hashes = {
        paths[key].name: sha256_file(paths[key]) for key in tables
    }
    forbidden_columns = {
        "p_value",
        "pvalue",
        "confidence_interval",
        "ci_low",
        "ci_high",
        "standard_error",
    }
    observed_columns = set().union(*(set(rows[0]) for rows in tables.values()))
    require(
        not forbidden_columns.intersection(observed_columns),
        "seed-inferential columns are forbidden",
    )
    audit: dict[str, object] = {
        "generated_at_utc": utc_now(),
        "aggregation_version": AGGREGATION_VERSION,
        "source_experiment_version": SOURCE_EXPERIMENT_VERSION,
        "status": "complete",
        "configuration_sha256": sha256_file(args.config),
        "layout_canonical_sha256": sha256_json(layout),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "expected_jobs": len(jobs),
        "completed_jobs": counts["complete"],
        "expected_seed_groups": config["seeds"],
        "expected_held_sport_families": config.get("held_sport_main", {}).get(
            "sport_families", []
        ),
        "table_rows": {key: len(rows) for key, rows in tables.items()},
        "input_sha256": dict(sorted(input_files.items())),
        "output_sha256": output_hashes,
        "difference_definitions": {
            "history": (
                "history-informed MAE minus forced-zero-history MAE; negative favors "
                "history-informed"
            ),
            "comparator": (
                "main forced-zero-history MAE minus comparator MAE for the identical "
                "seed, regime, horizon, and evaluation support; negative favors main"
            ),
        },
        "statistical_policy": {
            "purpose": "optimization-seed robustness, not participant-level inference",
            "seeds_treated_as_independent_participants": False,
            "hypothesis_tests_over_seeds": False,
            "p_values_over_seeds": False,
            "confidence_intervals_over_seeds": False,
            "reported_across_seed_statistics": ["median", "minimum", "maximum"],
            "participant_level_inference_source": (
                "prespecified user-cluster analyses in the underlying evaluation, "
                "not this seed aggregator"
            ),
        },
        "assertions": {
            "all_expected_jobs_complete": True,
            "all_audits_passed": True,
            "all_audit_seeds_match_directories": True,
            "unseen_main_frozen_before_external_inference": True,
            "no_external_selection_adaptation_or_recalibration": True,
            "external_freeze_and_metric_hashes_match": True,
            "required_metric_slices_present_once": True,
            "metric_values_finite_and_in_valid_ranges": True,
            "evaluation_support_constant_across_seeds": True,
            "history_differences_seed_paired": True,
            "model_comparisons_seed_and_support_paired": True,
            "all_expected_seed_pairs_present": True,
            "no_seed_inferential_tests_or_intervals": True,
        },
        "all_assertions_pass": True,
    }
    atomic_json(paths["audit"], audit)
    progress["final_artifacts_emitted"] = True
    progress["final_artifacts"] = {
        path.name: sha256_file(path)
        for key, path in paths.items()
        if key not in {"progress"} and path.exists()
    }
    progress["final_artifact_note"] = (
        "All expected jobs and strict audit checks passed; final tables are current."
    )
    atomic_json(paths["progress"], progress)
    return progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Q1 multi-seed experiments without treating optimization "
            "seeds as independent study participants."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/q1_multiseed_v0_21_0"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1_multiseed_v0_21_0.json"),
    )
    parser.add_argument(
        "--layout-config",
        type=Path,
        default=None,
        help=(
            "Optional JSON override for experiment directory and file-glob patterns. "
            "The built-in layout follows seed_<seed>/{unseen_main,...}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/q1_multiseed_v0_21_0/aggregation"),
    )
    return parser.parse_args()


def main() -> None:
    progress = aggregate(parse_args())
    print(
        json.dumps(
            {
                "status": progress["status"],
                "expected_jobs": progress["expected_jobs"],
                "status_counts": progress["status_counts"],
                "final_artifacts_emitted": progress["final_artifacts_emitted"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
