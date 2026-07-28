# User-generalization and frozen-external model specification

> **Superseded for final reporting.** The consolidated locked protocol is `protocol/FINAL_ANALYSIS_SPECIFICATION.md`; it includes the corrected rule that a prior session enters history only after it has ended.

**Model-array version:** 0.6.0  
**Tabular baseline version:** 0.8.0  
**Neural baseline version:** 0.9.0  
**History/uncertainty model version:** 0.11.0  
**Status:** executed for unseen-user and frozen external protocols; sport-shift and temporal protocols remain separate required experiments.

## Training population and leakage control

All models in this experiment are fitted only on Endomondo users assigned to the unseen-user training partition. Validation, calibration, and test users are mutually disjoint. The 3,458,033 dense 60 s training origins are available for fitting; reported results use only 300 s evaluation origins. Normalization is estimated from training users only.

Training sampling and XGBoost loss weights make every training user contribute equal total weight and every session within a user contribute equal total weight. Validation selection uses the prespecified origin-within-session, session-within-user, equal-user aggregation.

## Common inputs

- thirty causal 10 s bins of heart rate, speed, and altitude;
- an observed/missing mask for each channel;
- log elapsed exercise time;
- canonical sport-family token.

The XGBoost baseline uses 39 causal lag summaries. GRU, TCN, and Transformer baselines use the identical standardized sequences. All point models predict residuals relative to the last valid context heart rate and clip final heart-rate predictions to 30–240 bpm.

## Neural comparison and frozen comparator

GRU, TCN, and lightweight Transformer share a 64-dimensional representation, 500,000 hierarchically sampled origins per epoch, a 20-epoch maximum, and validation early stopping. The GRU had the lowest validation composite MAE (7.400 bpm) and is therefore the frozen common-input neural comparator for paired primary comparisons; this choice was made without test or GoldenCheetah outcomes.

## Main model

The main model uses a causal TCN current-session encoder, a 13-feature strictly earlier-session history encoder, an explicit no-history embedding, 20% history dropout, sport and elapsed-time fusion, and seven direct residual quantiles at 1, 3, and 5 min. Raw quantiles are sorted along the quantile dimension to prevent crossing. Validation selection averages history-informed and forced-zero-history hierarchical median MAE.

Two unseen-user modes are evaluated from the same frozen checkpoint:

- **zero history:** the history mask is forced to zero;
- **history informed:** only sessions whose start time is strictly earlier than the current session are supplied.

GoldenCheetah primary external validation always uses zero history. Any external history adaptation is a separate secondary experiment.

## Calibration

Endomondo calibration users are disjoint from training and validation users. Central 50%, 80%, and 90% intervals use finite-sample CQR higher-quantile thresholds with nonnegative expansion only. Thresholds are frozen before unseen-user test and GoldenCheetah evaluation. GoldenCheetah is never recalibrated in the primary external analysis.

## Point results

User-session-hierarchical MAE (bpm):

| Regime/model | 1 min | 3 min | 5 min |
|---|---:|---:|---:|
| Unseen user – XGBoost | 5.926 | 7.493 | 8.354 |
| Unseen user – GRU | 5.860 | 7.412 | 8.216 |
| Unseen user – TCN | 5.872 | 7.448 | 8.265 |
| Unseen user – Transformer | 5.851 | 7.396 | 8.258 |
| Unseen user – main, zero history | 5.869 | 7.472 | 8.250 |
| Unseen user – main, history informed | **5.832** | **7.388** | **8.121** |
| GoldenCheetah – GRU | 7.501 | 10.258 | 11.260 |
| GoldenCheetah – main, zero history | **7.464** | **10.200** | **11.203** |

The history-informed versus zero-history paired user bootstrap mean-MAE differences are −0.037 bpm (95% CI −0.061 to −0.013), −0.085 (−0.119 to −0.051), and −0.130 (−0.203 to −0.072). These gains are small in absolute magnitude and must be described as incremental personalization gains, not large clinical effects.

## Uncertainty results and interpretation

On unseen users, calibrated history-informed coverage error ranges from 0.004 to 0.020 across the three nominal levels and horizons. On GoldenCheetah, 3–5 min coverage remains low by roughly 0.04–0.05 at several levels, demonstrating calibration degradation under dataset/device distribution shift. Mean user-level Spearman correlations between 90% interval width and absolute error are approximately 0.35, 0.32, and 0.29 internally and 0.37, 0.32, and 0.27 externally.

These results support uncertainty usefulness and reveal external miscalibration; they do not support a claim that interval calibration transfers perfectly.

## Statistical comparison

Point-model differences use 10,000 paired bootstrap replicates at the user level for 95% effect CIs. Two-sided paired Wilcoxon tests are Holm-adjusted within each three-horizon comparison family. Negative MAE differences favor the main model. Effect CIs remain the primary magnitude evidence; rank-test significance is not substituted for a nonzero mean-effect CI.

## Artifacts

- `src/train_xgboost_baseline.py`
- `src/train_neural_baselines.py`
- `src/build_causal_history.py`
- `src/train_uncertainty_model.py`
- `src/evaluate_probabilistic_metrics.py`
- `src/bootstrap_model_comparisons.py`
- `outputs/results/xgboost_user_generalization_metrics_v0_8_0.csv`
- `outputs/results/gru_user_generalization_metrics_v0_9_0.csv`
- `outputs/results/tcn_user_generalization_metrics_v0_9_0.csv`
- `outputs/results/transformer_user_generalization_metrics_v0_9_0.csv`
- `outputs/results/uncertainty_point_metrics_v0_11_0.csv`
- `outputs/results/uncertainty_interval_metrics_v0_11_0.csv`
- `outputs/results/probabilistic_metrics_v0_11_0.csv`
- `outputs/results/paired_user_bootstrap_v0_11_0.csv`
- `notebooks/04_user_generalization_external_results.ipynb`
