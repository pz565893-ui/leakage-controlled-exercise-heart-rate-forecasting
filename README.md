# Leakage-Controlled Exercise Heart-Rate Forecasting

This directory contains the reproducible study and manuscript workspace for:

> **Boundary-dependent reliability of exercise heart-rate forecasts across users, sports, and data sources: a leakage-controlled study**

Chinese working title:

> **用户、运动项目与数据来源边界下的运动心率预测可靠性：一项数据泄漏控制研究**

## Current status

- The first-submission target is now *Physiological Measurement*. The controlling journal profile is `protocol/PMEA_SUBMISSION_PROFILE.md`; legacy BSPC files are retained only as provenance. The source manuscript now uses the required structured abstract, includes a SAGER-oriented sex/gender statement, moves the IOP generative-AI disclosure to the Acknowledgements, and applies a 30-user journal-facing reporting threshold to human-group outcome comparisons.

- The formal v0.22.0 multiseed analysis is complete: all 37 declared unseen-user, strict-temporal, learned-comparator, held-sport, and seed-specific frozen cross-source jobs passed their postconditions and aggregation audit. The zero-history-trained v0.23.0 analysis is also complete for all 10 declared jobs. Frozen-prediction v0.24.0 calibration/composition analyses and v0.25.0 paired-user comparisons were completed without model retraining or GoldenCheetah adaptation or recalibration. The independent persistence--conformal baseline (v0.26.0), matched-origin sport-availability sensitivity (v0.27.0), and deliberately invalid same-test-session contamination negative control (v0.28.0) are complete and audited. Horizon-specific eligibility (v0.29.0) and five-seed frozen-main-model sensitivity (v0.30.0) are also complete; all 15 frozen batch replays matched the original common cohorts exactly, and no training, selection, normalization refitting, calibration, or external-source adaptation occurred.
- The final past-only origin index contains 7,635,176 accepted origins. The Endomondo and GoldenCheetah complete 300-s pools contain 1,001,128 and 537,672 origins, respectively; the supported-three-family frozen cross-source evaluation contains 531,725 origins from 31,851 sessions and 144 users.
- XGBoost, GRU, point TCN, Transformer, persistence, EWMA, linear trend, and the history-capable seven-quantile TCN have been evaluated under matched past-only inputs and partitions where applicable.
- The audited v0.22.0--v0.30.0 summaries are integrated into the Markdown manuscript and supplement. The reported-number validator passes all 562 checks against the authoritative artifacts, including journal-facing checks that retain low-support results in the audit source while excluding them from PMEA outcome reporting.
- Data-boundary, strict-temporal, prediction-integrity, calibration-sensitivity, source-shift, cross-source duplicate, release-privacy, and PMEA presentation checks have executable audits; all 189 current unit tests pass.
- The exact raw-input snapshot is recorded by privacy-conservative SHA-256 audits covering 24,833,933,570 bytes and 51,622 files without releasing file-name inventories or participant identifiers.
- At the 28 July 2026 high-impact pre-submission build, the English main text contains 7,017 words, the structured abstract contains 244 words, and the PMEA-facing supplement contains 30 rendered tables under the continuous major sequence S1--S18. The cover letter, four main figures, one supplementary figure, and source-data tables are prepared.
- Editable Word-build pipelines exist for the PMEA main manuscript and supplementary material. Figure 3 now presents sport-family distribution shifts and Figure 4 presents uncertainty calibration, matching their first-citation order. The v0.30.0 documents retain one concise AI-use disclosure in Acknowledgements, use correct range typography, and omit the internal author reminder.
- The release policy now excludes the full raw-label ontology because GoldenCheetah workout labels can contain linkable free text; it also blocks reported user groups below 10, the internal graphical abstract, row-level files, and checkpoints. The non-identifying raw-source integrity audit remains reproducible and is included.
- A deterministic privacy-conservative repository builder creates `release/code_repository_upload_v0_30_0.zip` from a SHA-256 allowlist. Raw Endomondo/GoldenCheetah records, row-level predictions, identifiers, local paths, checkpoints, and TIFF masters are excluded. The deposited software and associated documentation are released under the MIT License; no third-party raw data are relicensed. Author names, affiliations, contact emails, PANG KEREN's ORCID, the no-specific-funding statement, and the no-competing-interests declaration are integrated. Remaining submission blockers are the institutional ethics determination, author-specific CRediT roles, any additional acknowledgements, and a persistent release DOI.

## Author handoff

- Complete the Chinese one-time form: `manuscript/AUTHOR_INPUT_FORM_CN.md`.
- Follow the target-journal profile: `protocol/PMEA_SUBMISSION_PROFILE.md`.
- Track all remaining gates: `protocol/PMEA_SUBMISSION_READINESS_CHECKLIST.md`.

## Evidence-first workflow

1. Audit source fields, sampling, heart-rate coverage, session duration, sport labels, and user history.
2. Freeze inclusion criteria and sport-family ontology.
3. Assign users and complete sessions to splits before generating windows.
4. Implement common baselines and the uncertainty-aware history-capable model.
5. Evaluate temporal, unseen-user, unseen-sport, joint-shift, and frozen cross-source protocols.
6. Draft Results from executed evidence, then complete the Introduction, Discussion, title, and Abstract.

## Directory map

- `protocol/`: study design, terminology, and leakage-control rules.
- `configs/`: machine-readable study parameters.
- `notebooks/`: executable data and experiment audit trails.
- `scripts/`: reproducible data preparation and evaluation utilities.
- `src/`: modeling code.
- `outputs/`: generated audit tables, manifests, models, and figures.
- `manuscript/`: manuscript, supplement, and submission documents.
- `DATA_SOURCES.md`: third-party source, placement, size, and SHA-256 record.
- `REPRODUCING.md`: aggregate-result verification and full-pipeline boundary.
- `PUBLIC_RELEASE_MANIFEST.md`: privacy-conservative public-release policy.
- `REPOSITORY_UPLOAD_GUIDE.md`: exact instructions for reviewing and publishing the generated code-repository bundle.

Raw datasets remain outside this directory and are never modified.
