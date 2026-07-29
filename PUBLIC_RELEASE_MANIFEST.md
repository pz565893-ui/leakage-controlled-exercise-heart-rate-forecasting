# Public release manifest policy

**Policy version:** 0.1.0  
**Status:** released under the MIT License; version 0.30.0 archived at https://doi.org/10.5281/zenodo.21649896

This document defines the default public-release boundary for the *Physiological Measurement* submission package. It is intentionally conservative: release files are selected through an explicit allowlist, while raw and row-level artifacts are excluded even if they are present in the working directory.

The integrity generator is:

```powershell
python src/generate_public_release_integrity.py
python src/generate_public_release_integrity.py --verify
```

It writes:

- `release/PUBLIC_RELEASE_INTEGRITY_v0_1_0.csv` — version, repository-relative file path, category, byte size, and SHA-256;
- `release/PUBLIC_RELEASE_INTEGRITY_v0_1_0.audit.json` — manifest hash, file count, total size, category counts, and the enforced safety policy.

The CSV is deterministic for unchanged inputs. It never writes local absolute paths, source participant identifiers, session/activity identifiers, exact participant times, geolocation fields, source activity paths, or linkable raw workout labels. Generation fails if a required allowlist pattern is missing, if a candidate CSV/TSV or JSON exposes a blocked field, if a reported user subgroup has fewer than 10 members, or if an allowlisted text file contains an author-machine absolute path.

## Default allowlist

Only the following classes are selected by default:

- release metadata: `.gitignore`, `README.md`, `REPRODUCING.md`, `DATA_SOURCES.md`, `ENVIRONMENT.md`, `requirements-lock.txt`, `CITATION.cff`, `LICENSE`, `REPOSITORY_UPLOAD_GUIDE.md`, and this policy;
- repository-local analysis code under `src/*.py`;
- unit tests under `tests/*.py`;
- `configs/study.yaml`; the full raw-label ontology is deliberately excluded, while its deterministic mapping rules remain in the released analysis code;
- protocol Markdown files;
- the consolidated bibliography, GoldenCheetah dataset citation, literature-search report, prior-work comparison note, and documented targeted literature update;
- authoritative aggregate result CSVs explicitly enumerated in the generator from `protocol/RESULT_ARTIFACT_AUTHORITY.md`;
  - the v0.22 multiseed release subset comprises `seed_variability_summary_v0_22_0.csv`, both history summary/seed-paired tables, both model-comparator summary/seed-paired tables, and `per_seed_metrics_long_v0_22_0.csv`;
  - the v0.23 zero-history-trained release subset comprises `strategy_contrasts_per_seed_v0_23_0.csv`, `strategy_contrast_seed_summary_v0_23_0.csv`, and `strategy_contrast_user_bootstrap_v0_23_0.csv`;
  - the v0.24 frozen-prediction release subset comprises the per-seed and summary balanced-calibration tables, their aggregate difference tables, and the point/interval sport-composition standardization tables;
  - the v0.25 release subset comprises aggregate seed-averaged paired-user comparator and held-sport/joint-shift confidence-interval tables;
  - v0.26--v0.28 add the persistence--conformal baseline, matched-origin sport-availability sensitivity, and explicitly invalid same-test-session contamination negative control;
  - v0.29--v0.30 add horizon-specific eligibility counts and five-seed frozen-model sensitivity results without retraining, model selection, normalization refitting, calibration, or external-source adaptation;
  - these files enter the allowlist only after their defining audits pass; training-run, queue, and participant-level aggregation audits remain excluded;
- the raw-source integrity audit, final results validation, reported-number validation, PMEA submission validation, and path-scrubbed v0.26/v0.27 analysis audits; the v0.29/v0.30 process audits remain private because they preserve local raw-input and checkpoint locations, while their identifier-free aggregate results are released;
- publication figures in PDF, SVG, and PNG, figure QA/contract files, and figure source-data CSVs, excluding 600-dpi TIFF masters and the optional graphical abstract;
- `manuscript/main_manuscript.md`, `manuscript/supplementary_material.md`, and `manuscript/highlights.txt` as integrity targets.

Inclusion in the integrity manifest means only that a file passed this technical allowlist. It does **not** grant a licence or override third-party data terms.

## Default blocklist

The following are excluded from the default public release:

- raw Endomondo or GoldenCheetah records and any copied raw-data directory;
- `outputs/manifests/`, `outputs/origins/`, `outputs/features/`, and `outputs/predictions/`;
- model checkpoints and normalization/calibration files under `outputs/models/`, which are intentionally excluded from the public release because they are not required to reproduce the deposited aggregate evidence and may retain training-derived artefacts;
- `configs/sport_ontology_v0_2_0.csv`, because GoldenCheetah raw labels include linkable free-text workout titles, routes, dates, or places;
- row-level audit databases or CSVs containing source keys, user IDs, session/activity IDs, exact times, paths, offsets, gender records, or coordinates;
- the v0.24/v0.25 process-audit JSON files, because these retained execution records can include author-machine paths; their identifier-free aggregate CSVs remain allowlisted;
- notebooks and notebook-creation helpers until saved author-machine paths are scrubbed;
- render caches, Python caches, temporary files, and superseded/pilot artifacts;
- 600-dpi TIFF publication masters, which remain in the private manuscript workspace and are unnecessary for code execution or aggregate-result verification;
- result files containing `pilot`, the superseded `paired_user_bootstrap_v0_11_0.csv`, and per-family run-level sport CSVs already represented by validated aggregate tables;
- `strategy_contrast_user_seed_mean_v0_23_0.csv`, because it is a per-user bootstrap input containing a pseudonymous user index;
- `recorded_gender_subgroups_v0_16_0.csv`, because it contains a reported subgroup with fewer than 10 users;
- `figures/Graphical_Abstract.*` and `figures/source_data/Graphical_Abstract_source.csv` until the author approves provenance, rights, and public release;
- queue/progress manifests, run-level audit JSON files, run-level model outputs, and prediction arrays from the v0.22/v0.23 experiment directories;
- `.npy`, `.npz`, `.pt`, `.sqlite`, `.fit`, `.tcx`, `.gpx`, and similar row-level or device files.

Stable public-platform IDs remain linkable pseudonyms. They are treated as non-release fields even when names are absent.

## Resolved release decisions

The following fields are deliberately unresolved and must not be inferred by the code:

1. **Software licence:** resolved as MIT for the deposited software and associated documentation.
2. **Repository and archive:** `https://github.com/pz565893-ui/leakage-controlled-exercise-heart-rate-forecasting`; version 0.30.0 archived at `https://doi.org/10.5281/zenodo.21649896`.
3. **Checkpoint release:** excluded from the default public repository and not required for this release.
4. **Third-party data terms:** FitRec/Endomondo is limited by its source page to academic, non-commercial use without redistribution. GoldenCheetah is currently labelled CC0 by OSF, but its raw records are still not mirrored under this privacy-conservative project policy. Neither source is relicensed by this project.

The released tag was generated and verified from the privacy-conservative integrity manifest before archiving. Post-release DOI metadata updates must be regenerated and verified on the repository default branch without moving the archived `v0.30.0` tag. The repository landing page should record creators, title, version, publication year, resource type, description, licence/rights, related dataset identifiers, and the related manuscript identifier when available.

## Release gate

A public release passes this policy only when:

- generation and `--verify` both return `status: PASS`;
- the final tag/commit matches the archived repository record;
- the selected licence and persistent DOI have been inserted by the authors;
- the archive has been tested from outside the author account;
- no blocked directory or participant-level field has been manually added after manifest generation.
