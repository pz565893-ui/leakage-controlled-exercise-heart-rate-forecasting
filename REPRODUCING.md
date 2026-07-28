# Reproducing the study

This project supports two reproducibility levels. The stored aggregate-results lane is fully testable without participant-level records. A full raw-to-model rerun additionally requires the two third-party datasets, substantial local storage, a CUDA-capable GPU, and a final author-approved run configuration.

## 1. Quick verification from released aggregate results

Run from the repository root in PowerShell with 64-bit Python 3.12.10:

```powershell
py -3.12 -m venv .venv
$StudyPython = (Resolve-Path -LiteralPath '.\.venv\Scripts\python.exe').Path
& $StudyPython -m pip install --upgrade pip
& $StudyPython -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
& $StudyPython -m pip install -r requirements-lock.txt
& $StudyPython -m pip check
& $StudyPython -m unittest discover -s tests -v
& $StudyPython src\make_publication_figures.py
& $StudyPython src\build_supplementary_material.py
& $StudyPython src\validate_reported_numbers.py
& $StudyPython src\build_pmea_docx.py
& $StudyPython src\build_pmea_supplementary_docx.py
& $StudyPython src\validate_pmea_submission.py
```

Expected checks for the v0.30.0 release candidate:

- all 189 current unit tests pass;
- `outputs/audit/REPORTED_NUMBER_VALIDATION.json` reports `PASS` with 562/562 checks;
- four main figures and one supplementary figure are rebuilt in PDF/SVG/TIFF/PNG from aggregate tables; the graphical abstract remains an internal optional artifact and is not part of the default submission route;
- the PMEA-facing supplement has a continuous major-table sequence S1--S18, with S18a and S18b presenting target-availability sensitivity results;
- the main manuscript and PMEA supplementary Word files are created and pass the scientific/format validator.

The Word files must then be exported with a supported Word-compatible renderer and visually inspected. Final page-level QA records, accessibility reports, and the DOCX structure audit must be regenerated after the last source change; earlier render reports are not evidence for a newer Word build.

## 2. Raw-data preprocessing and model-training order

Place the third-party data outside the repository as described in `DATA_SOURCES.md`. The processing order is:

1. link GoldenCheetah metadata to activity files;
2. build session-quality manifests;
3. audit exact and high-similarity signals within and across sources, and apply only documented exclusions;
4. build the sport ontology and user/session split manifests;
5. construct past-only forecast origins;
6. build session-series storage, completed-workout history, model arrays, and tabular features;
7. fit causal signal, XGBoost, GRU, point-TCN, Transformer, and quantile-TCN models;
8. fit the strict-temporal and five leave-one-sport-family-out models;
9. run frozen GoldenCheetah inference without tuning or recalibration;
10. compute probabilistic metrics, seed-matched paired user bootstraps, zero-history-trained contrasts, the independent persistence--conformal baseline, matched-origin sport-availability sensitivity, horizon-specific five-seed frozen-model sensitivity, subgroup descriptions, and publication artifacts;
11. run the locked deliberately leaky same-test-session negative control only as an explicitly invalid diagnostic analysis, never as model-selection or leaderboard evidence.

The relevant command-line programs are in `src/`; each exposes `--help`. The authoritative artifact map is `protocol/RESULT_ARTIFACT_AUTHORITY.md`, and the leakage invariants are in `protocol/LEAKAGE_CONTROL_CONTRACT.md`.

The formal multiseed runs are declared by machine-readable configurations, queue manifests, command logs, selected-checkpoint audits, seed values, epoch histories, row counts, overlap checks, and immutable external-freeze records. A release claim is valid only for jobs that pass the strict postcondition and aggregation audits; incomplete queue directories and intermediate manuscript values are not release evidence. The v0.28.0 negative control has a separate invalid-control authority tier and cannot support a claim that contaminated splitting is valid.

## 3. Frozen cross-source evaluation boundary

GoldenCheetah is used before evaluation only for source-integrity checks, metadata linkage, sport-label harmonization, quality eligibility, and cross-source duplicate auditing. No GoldenCheetah forecast error, interval coverage, or other predictive outcome may choose the architecture, optimizer, hyperparameters, checkpoint, normalization, or CQR adjustment. Every unseen-user seed must create a checkpoint-freeze record before its cross-source inference containing:

- resolved configuration hash and source-code hashes;
- selected checkpoint, normalization, ontology, and calibration-threshold SHA-256 values;
- selection metric and development partition;
- timestamp and command line;
- confirmation that no GoldenCheetah outcome was used for selection.

The cross-source inference command may be run only after that record is immutable. No GoldenCheetah fine-tuning or recalibration is permitted for the reported analysis.

## 4. Resource expectations

- Tested full-training platform: Windows, Python 3.12.10, NVIDIA RTX 5060 Laptop GPU with 8 GB VRAM, PyTorch 2.11.0+cu128.
- Quick tests and aggregate-result validation can run on CPU.
- The local working project used approximately 10 GB for generated outputs; the three source collections together occupy approximately 25 GB. Allow at least 50 GB free scratch space for a clean full rebuild.
- Full model training was not benchmarked for a CPU-only environment.

## 5. Privacy-safe public release

Do not upload the working `outputs/` tree. It contains stable pseudonymous identifiers, exact times, source paths, row-level arrays, and large prediction files. Follow `PUBLIC_RELEASE_MANIFEST.md` and generate the explicit allowlist only after all manuscript files are stable:

```powershell
& $StudyPython src\generate_public_release_integrity.py
& $StudyPython src\generate_public_release_integrity.py --verify
& $StudyPython src\build_public_repository_package.py
```

The generated integrity CSV identifies safe release candidates by repository-relative path, size, category, and SHA-256. The package builder then creates `release/code_repository_upload_v0_30_0.zip`; extract and upload its contents, not the ZIP itself. Inclusion does not grant a licence or override third-party data terms.

## 6. Remaining release blockers

Before public archiving, the authors must select a software licence, confirm rights for aggregate artifacts, add the final repository URL to `CITATION.cff`, create a tagged Git release, mint a persistent DOI, and test the archive outside the author account. Checkpoints remain outside the default package.
