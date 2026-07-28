# Result artifact authority map

Date: 2026-07-23  
Purpose: distinguish primary multiseed evidence, zero-history-trained strategy contrasts, frozen-prediction post hoc analyses, reference-seed sensitivities, and retained provenance

## Evidence hierarchy

1. **Primary multiseed evidence (v0.22).** The version 0.22 aggregation is the reporting authority for the history-capable model, learned comparators, uncertainty, and held-sport results. The history-capable model uses five seeds; GRU, point-TCN, and held-sport experiments use three seeds; deterministic baselines have no optimization seed. Seed ranges are descriptive stability ranges, not confidence intervals.
2. **Zero-history-trained strategy evidence (v0.23).** Version 0.23 is the reporting authority for contrasts between history-informed inference and a separate model trained, selected, and evaluated without prior-workout history. Its user-bootstrap table averages paired differences over the five matched seeds before resampling users; seeds are not resampled.
3. **Frozen-prediction uncertainty, baseline, and paired-comparison evidence (v0.24--v0.27).** Version 0.24 is the reporting authority for equal-user/equal-session calibration sensitivity and cross-source sport-composition standardization of point and interval metrics. Version 0.25 is the reporting authority for seed-averaged paired-user confidence intervals for learned comparators and held-sport/joint-shift comparisons. Version 0.26 adds an independently calibrated symmetric split-conformal persistence baseline. Version 0.27 adds a matched-origin, history-masked full-sport-versus-held-family sensitivity. These analyses reuse frozen observations or predictions; they do not tune, adapt, or recalibrate on GoldenCheetah outcomes.
4. **Deliberately invalid leakage negative control (v0.28).** Version 0.28 is the reporting authority only for the retrospective same-test-session-window contamination experiment. It compares freshly initialized zero-history-trained models with the corresponding clean v0.23 models on identical strict-temporal test origins. Because fitting, validation, and calibration were deliberately contaminated, v0.28 is excluded from valid-model rankings and cannot validate window-level splitting or estimate other leakage mechanisms. Its aggregate contrasts are bounded diagnostic evidence, not model-performance evidence.
5. **Reference-seed sensitivity evidence.** Versioned result tables under `outputs/results/` remain authoritative only for analyses explicitly labelled as reference-seed or secondary sensitivities. For optimized neural models, the frozen reference seed is 20260722. These tables must not replace the relevant v0.22--v0.28 authority tier.
6. **Deterministic and descriptive evidence.** Naive or otherwise deterministic baselines have no optimization seed. Source-integrity, composition, and availability summaries are descriptive audits rather than model-seed estimates.

## Authoritative final artifacts

| Result family | Authority and artifact(s) |
|---|---|
| Primary multiseed point, interval, probabilistic, comparator, and held-sport summaries | `outputs/q1_multiseed_v0_21_0/aggregation/seed_variability_summary_v0_22_0.csv` |
| Primary history-versus-within-checkpoint-zero contrasts | `main_history_seed_paired_v0_22_0.csv` and `main_history_difference_summary_v0_22_0.csv` in the v0.22 aggregation directory |
| Primary model-versus-comparator contrasts | `main_vs_comparator_seed_paired_v0_22_0.csv` and `main_vs_comparator_summary_v0_22_0.csv` in the v0.22 aggregation directory |
| Primary per-seed aggregate provenance | `per_seed_metrics_long_v0_22_0.csv` in the v0.22 aggregation directory; this contains aggregate metrics and repository-relative source labels, not participant-level predictions |
| Zero-history-trained strategy contrasts | `outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrasts_per_seed_v0_23_0.csv`, `strategy_contrast_seed_summary_v0_23_0.csv`, and `strategy_contrast_user_bootstrap_v0_23_0.csv` |
| Balanced calibration sensitivity | `outputs/results/multiseed_balanced_calibration_per_seed_v0_24_0.csv`, `multiseed_balanced_calibration_summary_v0_24_0.csv`, `multiseed_balanced_calibration_differences_v0_24_0.csv`, and `multiseed_balanced_calibration_difference_summary_v0_24_0.csv`; audit: `outputs/audit/multiseed_balanced_calibration_v0_24_0.json` |
| Cross-source sport-composition standardization of point and interval metrics | `outputs/results/external_sport_standardization_v0_20_1.csv`, `external_sport_uncertainty_bootstrap_v0_21_0.csv`, and `external_sport_uncertainty_standardization_v0_24_0.csv`; audit: `outputs/audit/external_sport_uncertainty_standardization_v0_24_0.json` |
| Seed-averaged paired-user comparator and held-sport effects | `outputs/results/multiseed_paired_model_comparisons_v0_25_0.csv` and `multiseed_paired_sport_shift_v0_25_0.csv`; audit: `outputs/audit/multiseed_paired_user_bootstrap_v0_25_0.audit.json` |
| Independent symmetric split-conformal persistence baseline | `outputs/results/persistence_conformal_baseline_v0_26_0.csv`; audit: `outputs/audit/persistence_conformal_baseline_v0_26_0.json` |
| Matched-origin sport-availability sensitivity | `outputs/results/matched_sport_availability_v0_27_0.csv`; audit: `outputs/audit/matched_sport_availability_v0_27_0.json` |
| Deliberately contaminated same-session-window negative control | `outputs/deliberately_leaky_negative_control_v0_28_0/aggregation/paired_metrics_seed_summary_v0_28_0.csv`, `paired_user_bootstrap_v0_28_0.csv`, and `interval_diagnostics_per_seed_v0_28_0.csv`; audit: `outputs/deliberately_leaky_negative_control_v0_28_0/aggregation/audit.json`. The participant-level `paired_user_seed_mean_v0_28_0.csv` is private and excluded from release. |
| Forecast-origin construction | `outputs/audit/forecast_origins_full_v0_3_1.json` and `outputs/origins/forecast_origins_v0_3_1.sqlite` (private process provenance; not public-release artifacts) |
| Split manifests | `outputs/audit/split_manifest_v0_2_0_summary.json` and version 0.2.0 manifest CSVs (private process provenance; not public-release artifacts) |
| Reference-seed strict temporal point, interval, probabilistic, and paired results | `temporal_uncertainty_point_v0_13_0.csv`, `temporal_uncertainty_interval_v0_13_0.csv`, `temporal_probabilistic_metrics_v0_13_0.csv`, and `temporal_paired_comparisons_v0_13_0.csv` |
| Reference-seed unseen-user and frozen cross-source point, interval, probabilistic, and paired results | `uncertainty_point_metrics_v0_11_0.csv`, `uncertainty_interval_metrics_v0_11_0.csv`, `probabilistic_metrics_v0_11_0.csv`, and `paired_model_comparisons_v0_11_0.csv` |
| Reference-seed learned-comparator source tables | version 0.8.0 XGBoost and version 0.9.0 GRU/TCN/Transformer metric CSVs; multiseed reporting is taken from v0.22 |
| Reference-seed sport and joint-shift source tables | aggregated version 0.12.0 point/interval CSVs plus `SPORT_SHIFT_VALIDATION_v0_12_0.json`; multiseed held-sport reporting is taken from v0.22 |
| Reference-seed sport and joint-shift uncertainty sensitivities | `sport_shift_mae_bootstrap_v0_19_0.csv` and `sport_shift_uncertainty_bootstrap_v0_17_0.csv` |
| Reference-seed main uncertainty-panel confidence intervals | `figure3_uncertainty_bootstrap_v0_18_0.csv` |
| Completed-workout history availability | `history_availability_v0_19_0.csv` (descriptive) |
| Reference-seed signal ablation | version 0.14.0 point/interval/probabilistic CSVs and `signal_ablation_paired_v0_14_0.csv` |
| Reference-seed dense-origin sensitivity | version 0.15.0 point/interval CSVs |
| Reference-seed recorded-gender descriptions | version 0.16.0 subgroup and difference CSVs |
| Figure values | CSVs in `figures/source_data/`, regenerated from the authority tier identified for each panel |

## Retained but not authoritative for primary manuscript claims

- Files containing `pilot` are feasibility or training-pipeline checks.
- Version 0.1.0 manifests and ontology outputs are retained for provenance; version 0.2.0 is final.
- Version 0.3.0 forecast-origin artifacts are superseded by version 0.3.1.
- `paired_user_bootstrap_v0_11_0.csv` preceded the final aligned prediction rebuild and is superseded by `paired_model_comparisons_v0_11_0.csv` within the reference-seed sensitivity tier.
- The reference-seed v0.11--v0.21 result tables support labelled sensitivities and provenance; they do not supersede the relevant v0.22--v0.28 evidence tier.
- Per-family sport-shift CSVs are retained as run-level outputs; the validated aggregated version 0.12.0 tables are the reference-seed reporting source.
- `outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrast_user_seed_mean_v0_23_0.csv` is an internal per-user bootstrap input and is excluded from public release.
- Queue manifests, progress manifests, aggregation audits, run-level metrics, checkpoints, and prediction arrays are process provenance, not public aggregate result files.
- `MANUSCRIPT_DRAFT.md` is an early scaffold; `main_manuscript.md` is the authoritative manuscript source.

## Release rule

The public integrity manifest uses an exact-file allowlist. It includes the enumerated non-identifying aggregate tables through v0.27 and may include only the explicitly enumerated aggregate v0.28 tables after privacy review. The v0.28 participant-level bootstrap input remains private. Execution-audit JSON files are released only when explicitly allowlisted and confirmed not to expose author-machine paths or participant-level identifiers. Other per-user bootstrap inputs, raw user/session identifiers, queue and progress manifests, run-level audits, full model arrays, raw-data-derived manifests, prediction archives, and checkpoints remain excluded. Reference-seed sensitivity tables may be released only when they are separately enumerated in the allowlist and pass the same privacy checks.
