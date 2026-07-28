# Final analysis specification

**Locked:** 2026-07-22  
**Registration status:** retrospective consolidation of the executed versioned analyses; not a prospective preregistration  
**Target journal:** *Physiological Measurement* (first submission; journal-facing reporting overlay added 27 July 2026)  
**Development dataset:** Endomondo HR  
**Cross-source dataset:** GoldenCheetah OpenData  
**Primary seed:** 20260722

## Study question and boundary

The study evaluates past-only 1-, 3-, and 5-min exercise heart-rate forecasts under within-user temporal separation, unseen-user transfer, leave-one-sport-family-out transfer, joint user--sport shift, and frozen cross-source evaluation. The paper tests forecast accuracy and empirical interval calibration; it does not diagnose disease, prescribe training, establish physiological causality, or claim injury prevention.

## Forecasting task

Each forecast origin uses thirty right-closed 10-s bins covering the preceding 300 s; the last valid observation in each bin is retained and an empty bin remains zero-valued with a missingness mask. Dynamic channels are heart rate, speed, and altitude, each paired with an observed/missing mask. Elapsed session time and a protocol-defined sport-family token are added at fusion. Targets are heart rate at +60, +180, and +300 s. Exact targets are preferred; interpolation is allowed only between valid future heart-rate samples separated by at most 30 s. All three targets must be available, so every reported horizon uses the same complete-three-target cohort. Inputs never extend beyond the forecast origin.

Candidate origins are indexed every 60 s. Validation, calibration, test, and primary external metrics use origins aligned every 300 s. The 60-s evaluation-origin analysis is a frozen-checkpoint sensitivity test.

## Data and duplicate control

Before window construction, exact-signal fingerprints were formed from timestamp, heart rate, speed, altitude, latitude, and longitude while excluding identifiers and labels. Every cross-user group and every sport-conflicting group was excluded. Within-user, sport-consistent groups retained one deterministic canonical record.

After signal-quality, sport-ontology, and duplicate control, 201,823 Endomondo sessions and 32,587 GoldenCheetah sessions remained provisionally eligible. Final origin construction retained 5,008,341 Endomondo and 2,626,835 GoldenCheetah candidate origins; the primary 300-s subset contained 1,001,128 and 537,672 origins, respectively.

## Leakage-control rules

1. Users, sessions, and sport families were assigned before window construction.
2. A session never appeared in more than one partition of a given protocol.
3. Normalization, feature selection, model fitting, early stopping, and conformal calibration excluded all corresponding test data.
4. A prior workout entered user history only if its end time was at or before the current workout start time. A workout that merely started earlier but was still active was not eligible.
5. For leave-one-sport-family-out models, the held family contributed neither training, validation, or calibration origins nor historical-session summaries.
6. GoldenCheetah was not used for training, early stopping, model selection, or primary interval calibration.
7. Statistical resampling used users, not overlapping origins, as the resampling unit.

## Evaluation protocols

### Strict within-user temporal test

Complete sessions were ordered within each Endomondo user and allocated approximately 70%/10%/10%/10% to train, validation, calibration, and test. Forty-seven sessions that touched or crossed adjacent temporal boundaries were removed from the strict protocol. The final test contained 104,144 origins from 16,012 sessions and 948 users, with zero session overlap and zero temporal-order violations.

### Unseen-user test

Endomondo users were assigned by seeded hash to 759 training, 129 validation, 97 calibration, and 105 test users. User overlap was zero. The same frozen history-capable checkpoint was evaluated in two modes: history-informed inference supplied only completed prior workouts, whereas history-masked inference deliberately hid that input.

### Unseen-sport-family and joint shift

Five Endomondo families met the protocol-defined support threshold: outdoor cycling, indoor/virtual cycling, running, walking/hiking, and strength/cross-training. A separate model was fitted for each held family. The same-user sport-shift regime evaluated held-family sessions from users eligible to supply non-held training information; the joint-shift regime additionally restricted evaluation to unseen-user test users. Intersections with fewer than 25 users are labeled cautionary rather than broadly generalizable.

For the *Physiological Measurement* submission, outcome estimates for measurements on human groups are reported only when a cell contains at least 30 users, in accordance with the journal's stated group-size requirement. This is a journal-facing reporting overlay, not a change to the locked computation: the 18--20-user joint intersections remain preserved in the immutable analysis artifacts for auditability, but the submission presents only their support counts and omits their outcome estimates.

### Frozen cross-source evaluation

The complete 300-s GoldenCheetah pool contained 537,672 origins from 32,443 sessions and 144 users. After applying the primary model's supported-three-family scope, the frozen cross-source model evaluation contained 531,725 origins from 31,851 sessions and the same 144 users. The Endomondo checkpoint, normalizers, ontology mapping, and conformal thresholds were unchanged. The cross-source estimand used history-masked inference, with no GoldenCheetah recalibration or fine-tuning. Transport of the completed-history branch was not evaluated. Reference-seed GoldenCheetah outcomes were inspected before the final five-seed reporting plan was consolidated; the evaluation is therefore retrospectively frozen rather than prospectively sequestered, although each checkpoint and threshold was frozen before its corresponding cross-source inference.

## Models

The past-only reference models were persistence, an exponentially weighted moving average selected on Endomondo validation data, and linear extrapolation. Learned baselines were XGBoost, a two-layer 64-unit GRU, a 64-channel time-causal TCN, and a three-layer four-head lightweight Transformer. Learned point models predicted residuals from the last observed context heart rate.

The main model projected the six dynamic value/mask channels to 64 TCN channels and applied four residual temporal blocks with dilations 1, 2, 4, and 8. A 13-feature completed-workout summary was encoded by a two-layer 32-unit MLP; a learned 32-dimensional no-history vector represented absent history. The current-session representation, an eight-dimensional sport embedding, log elapsed time, history representation, and history-presence mask entered a 96-unit fusion head. The head directly estimated seven ordered residual quantiles (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, and 0.95) for all three horizons. Sorting along the quantile axis enforced non-crossing outputs.

Training used AdamW with learning rate 0.001 and weight decay 0.0001, batches of 2,048, hierarchical user/session sampling, 500,000 sampled origins per epoch for the main models, mixed precision, gradient-norm clipping at 1.0, and validation early stopping. Twenty per cent of available history masks were dropped during main-model fitting. Model selection averaged history-informed and history-masked hierarchical validation MAE.

## Calibration and outcomes

Central 50%, 80%, and 90% intervals were calibrated with split conformalized quantile regression. Nonconformity thresholds used the finite-sample higher quantile and nonnegative expansion; intervals were never narrowed. Calibration records were disjoint from model-selection and test records. In the unseen-user protocol, calibration users were additionally disjoint from validation and test users; in the strict temporal protocol, later calibration and test sessions belonged to the same users but were separated in time and by session.

The primary point metric is session-then-user hierarchical MAE in bpm. RMSE and signed bias are secondary. Distributional evaluation reports prediction interval coverage probability, absolute coverage error, mean interval width, pinball loss, weighted interval score, and within-user Spearman association between 90% interval width and absolute error.

Paired model differences use 10,000 user-level bootstrap replicates for 95% confidence intervals. Two-sided paired Wilcoxon tests are adjusted by Holm within each three-horizon comparison family. Effect estimates and bootstrap confidence intervals are primary; rank-test significance is supplementary.

## Locked sensitivities and subgroup boundary

- Heart-rate-only signal ablation retains elapsed time, sport family, and past-only completed-workout history while masking speed/altitude values and masks.
- Dense-origin sensitivity evaluates the frozen main checkpoint every 60 s instead of every 300 s.
- Dataset-provided recorded-gender metadata are retained for cohort characterization only. Because relevant strata do not consistently meet the journal's 30-person group requirement and do not represent a verified measure of sex assigned at birth or gender identity, the *Physiological Measurement* submission makes no sex- or gender-stratified outcome comparison.

## Version ledger

Ontology 0.2.0; split 0.2.0; forecast origins 0.3.1; session series 0.4.0; naive baselines 0.5.0; arrays 0.6.0; tabular features 0.7.0; XGBoost 0.8.0; neural baselines 0.9.0; completed-workout history 0.10.1; user/cross-source uncertainty model 0.11.0; sport shift 0.12.0; strict temporal protocol 0.13.0; signal ablation 0.14.0; dense-origin sensitivity 0.15.0; recorded-gender analysis 0.16.0; sport-shift uncertainty bootstrap 0.17.0; Figure 3 uncertainty bootstrap 0.18.0; sport-shift MAE bootstrap and history-availability summary 0.19.0; cross-source integrity and sport-composition analyses 0.20.0--0.21.0; five-seed aggregation 0.22.0; zero-history-trained strategy analysis 0.23.0; balanced calibration and interval standardization 0.24.0; paired-user bootstrap comparisons 0.25.0.
