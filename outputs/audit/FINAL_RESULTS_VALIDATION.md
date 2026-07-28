# Final results validation report

**Assessment:** PASS for technical manuscript assembly; author-supplied submission fields remain open  
**Date:** 23 July 2026

## Evidence hierarchy

- The v0.22.0 multiseed aggregation is authoritative for principal point and interval summaries: five seeds for the history-capable temporal/unseen-user models and frozen cross-source inference, three matched seeds for GRU/point-TCN and held-sport experiments, and deterministic signal baselines.
- The v0.23.0 aggregation is authoritative for zero-history-trained models and training-strategy contrasts. Paired per-user differences were averaged over five matched seeds before 10,000 user-bootstrap resamples; seeds were not resampled.
- Frozen-prediction v0.24.0 analyses are authoritative for balanced-calibration sensitivity and cross-source sport-composition standardization; v0.25.0 is authoritative for seed-averaged paired-user comparator and held-sport confidence intervals. v0.26.0 adds the persistence-conformal baseline; v0.27.0 adds matched-origin sport-availability sensitivity; v0.28.0 adds the deliberately leaky same-test-session negative-control diagnostic.
- Seed 20260722 is retained only for explicitly labelled sensitivity analyses, user-bootstrap interval diagnostics, signal ablation, source characterization, and related descriptive analyses.

Both formal training queues are complete: 37/37 v0.22 jobs and 10/10 v0.23 jobs. Their strict aggregation audits pass and their frozen source hashes are unchanged; the post hoc v0.24/v0.25 audits also pass.

## Independent validation gates

- Forecast-origin construction contains 7,635,176 accepted causal origins. The reported unseen-user, strict-temporal, and frozen cross-source evaluation sets contain 101,184, 104,144, and 531,725 origins, respectively.
- All 169 repository unit tests pass; `pip check` reports no broken requirements.
- The machine-linked reported-number validator passes 473/473 checks, including Table 1, all v0.23 strategy contrasts, v0.24 calibration/composition analyses, v0.25 paired-user comparisons, v0.26 persistence baseline checks, v0.27 matched-sport sensitivity checks, v0.28 invalid-control diagnostics, multiseed interval and held-sport cells, and 47 provenance-path checks.
- DOCX structure validation passes. Canonical accessibility audits for the main manuscript, supplement, Highlights, and figure captions each report zero high-, medium-, and low-severity findings.
- Microsoft Word read-only renders were inspected page by page: 18 main-manuscript pages, 19 supplementary pages, one Highlights page, and two figure-caption pages. No clipping, overlap, unintended blank page, split table row, orphaned table header, or missing glyph remained.

## Primary point results

Hierarchical MAE in bpm; learned-model entries are medians across their declared seed sets:

| Regime/model | 1 min | 3 min | 5 min |
|---|---:|---:|---:|
| Strict temporal, history-informed | 6.012 | 7.557 | 8.245 |
| Strict temporal, zero-history-trained | 6.067 | 7.617 | 8.332 |
| Strict temporal, GRU | 6.054 | 7.581 | 8.300 |
| Strict temporal, persistence | 6.726 | 8.757 | 9.587 |
| Strict temporal, EWMA | 6.989 | 8.171 | 8.955 |
| Unseen user, history-informed | 5.849 | 7.407 | 8.157 |
| Unseen user, zero-history-trained | 5.866 | 7.487 | 8.285 |
| Unseen user, GRU | 5.902 | 7.423 | 8.216 |
| Frozen GoldenCheetah cross-source, history-masked | 7.465 | 10.206 | 11.214 |
| Frozen GoldenCheetah cross-source, zero-history-trained | 7.461 | 10.218 | 11.226 |
| Frozen GoldenCheetah cross-source, GRU | 7.527 | 10.245 | 11.260 |

Relative to zero-history-trained models, history-informed user effects were -0.044 [-0.059, -0.029], -0.063 [-0.091, -0.034], and -0.079 [-0.113, -0.046] bpm in the strict-temporal test. In unseen users they were -0.029 [-0.056, 0.006], -0.076 [-0.124, -0.036], and -0.112 [-0.187, -0.048] bpm. The 1-min unseen-user interval includes zero; completed history is incremental rather than clinically large.

On frozen GoldenCheetah, history-masked minus zero-history-trained effects were -0.006 [-0.015, 0.003], -0.014 [-0.029, 0.002], and -0.024 [-0.041, -0.008] bpm. Only the 5-min interval excludes zero, and the magnitude is too small to support a broad training-strategy superiority claim.

## Sport and uncertainty findings

Three-seed same-user held-sport 5-min MAE medians ranged from 7.617 bpm for running to 12.012 bpm for strength/cross-training. Joint user-sport 5-min medians ranged from 7.337 to 11.797 bpm; indoor/virtual cycling, walking/hiking, and strength/cross-training intersections contained only 18, 19, and 20 users and are explicitly cautionary.

Five-seed post-CQR 90% PICP medians were 0.896/0.897/0.892 in the strict-temporal test, 0.895/0.891/0.883 in unseen users, and 0.880/0.859/0.850 on frozen GoldenCheetah. Cross-source intervals widened to 29.55/37.50/40.56 bpm but still under-covered. These are empirical calibration results, not finite-sample user-level conformal guarantees.

## Claim permissions

| Candidate claim | Decision |
|---|---|
| Leakage-controlled forecasting was evaluated across temporal, user, sport, joint, and frozen cross-source boundaries | Supported |
| Completed prior-workout history provides a small forecasting improvement | Supported, with the 1-min unseen-user uncertainty stated |
| The main architecture dominates every learned comparator | Not supported |
| Forecast difficulty and interval behavior differ across sport families | Supported descriptively |
| Joint user-sport shift always worsens error | Not supported |
| Internal interval calibration transfers unchanged to GoldenCheetah | Not supported |
| Speed and altitude consistently improve forecasts | Not supported |
| Recorded-gender performance differs | Not supported |
| The system has clinical, diagnostic, safety, or exercise-prescription efficacy | Prohibited by study design |

## Final caution

The datasets contain self-selected users, heterogeneous and incompletely documented devices, observational workouts, and imbalanced recorded gender. GoldenCheetah is an independent data source but not a controlled device-validation cohort. The technical package is internally consistent; upload remains blocked until authors supply identities and affiliations, ethics determination, CRediT roles, funding, competing interests, acknowledgements, licence decisions, and a persistent repository DOI.
