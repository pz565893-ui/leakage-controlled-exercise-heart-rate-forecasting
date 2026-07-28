# Initial Study Protocol (Not Prospectively Registered)

**Status and timing:** This document records the initial protocol-led design but was not deposited or registered before all analyses. Reference-seed GoldenCheetah results were inspected before the final five-seed reporting configuration was consolidated. Each reported checkpoint was nevertheless frozen before its corresponding GoldenCheetah inference, and GoldenCheetah was not used for fitting, early stopping, adaptation, or recalibration. The final analysis specification and versioned audit trail, rather than this file alone, define the reported analysis.

## Working title

**Uncertainty-Aware Exercise Heart-Rate Forecasting under User and Sport Distribution Shifts: A Leakage-Controlled Multi-Dataset Study**

## One-sentence argument

In observational multi-sport wearable exercise streams, we will test whether a causal, history-conditioned forecaster with empirical prediction intervals can retain useful accuracy and interval coverage under user and sport distribution shifts, using leakage-controlled Endomondo experiments and a frozen cross-source evaluation on GoldenCheetah; conclusions will be limited to the recorded populations, sports, sensors, and forecast horizons.

## Primary research question

How much do point accuracy and interval calibration deteriorate when exercise heart-rate forecasts are transferred to unseen users, unseen sport families, and their intersection, and can causally available user history and calibrated predictive uncertainty reduce or identify this deterioration?

## Prespecified hypotheses

- **H1 — Shift penalty:** unseen-user, unseen-sport-family, and joint-shift evaluation will produce larger user-aggregated MAE than within-user temporal evaluation.
- **H2 — Causal personalization:** a user-history encoder will improve forecasts for history-informed users relative to an otherwise identical non-personalized model, without benefiting zero-shot users through identity leakage.
- **H3 — Uncertainty utility:** conformalized intervals will achieve smaller absolute coverage error than uncalibrated neural intervals on internal validation.
- **H4 — OOD awareness:** interval width will increase with realized error under user and sport shifts; this association will be evaluated rather than assumed.
- **H5 — External transportability:** a frozen Endomondo-trained model will show measurable degradation on GoldenCheetah, and its uncertainty outputs will reveal whether that degradation is accompanied by miscalibration.

No directional performance superiority is claimed before experiments are run.

## Datasets and roles

| Dataset | Role | Permitted use |
|---|---|---|
| Endomondo HR | model development | training, nested validation, hyperparameter selection, internal calibration, and prespecified internal tests |
| GoldenCheetah OpenData subset | external validation | frozen-model primary evaluation; optional recalibration only in a separately declared secondary protocol |

PAMAP2 and SportDB2 are not primary forecasting datasets unless the feasibility audit shows compatible longitudinal HR and workload signals at the required horizons. They may support a sensitivity analysis but will not be forced into the main study.

## Unit of analysis and target construction

- **Source grain:** repeated sensor observations nested within exercise sessions, nested within users.
- **Forecast example:** one forecast origin within an eligible session.
- **Context:** 300 s immediately preceding and including the forecast origin.
- **Targets:** HR at +60, +180, and +300 s.
- **Primary temporal grid:** 10 s. A deterministic 5,000-record Endomondo audit found a median observation interval of 8 s; the 10 s grid is retained as a conservative shared representation that limits interpolation and yields 30 context steps. This choice will be checked at 5 s and 30 s.
- **Candidate-origin stride:** 60 s for training-index coverage.
- **Primary evaluation stride:** 300 s for validation, calibration, and test reporting, reducing overlap between adjacent contexts and targets.
- **Target alignment:** exact observation when available; otherwise linear interpolation between valid future HR samples separated by at most 30 s.
- **Sensitivity grids:** 5 s and 30 s if computationally and statistically supported.
- **Target alignment:** nearest valid observation within a prespecified tolerance after causal resampling; tolerance will be fixed after the sampling-gap audit and before model comparison.

## Candidate causal inputs

### Shared core inputs

- past HR;
- elapsed session time;
- speed when observed, or a causally derived speed from prior distance/GPS samples;
- cumulative distance;
- altitude and causal altitude change/grade;
- canonical sport family;
- observation mask and time-since-last-observation for every dynamic channel.

### User-history inputs

- summaries or embeddings of complete sessions strictly preceding the indexed session;
- historical sport exposure, session duration, HR distribution, and workload–HR response computed only from eligible earlier observations;
- explicit no-history mask for zero-shot users.

### Extended inputs

Power and cadence will be evaluated only in a secondary sensor-rich analysis because they are not consistently shared between Endomondo and GoldenCheetah.

## Explicitly prohibited inputs

- speed, power, cadence, altitude, distance, or HR occurring after the forecast origin;
- statistics computed from the full indexed session;
- future or held-out sessions in user-history features;
- user means, standard deviations, or maxima computed before split assignment;
- external-test outcomes used for feature engineering, tuning, or primary interval calibration.

## Eligibility criteria

These thresholds are prospective and will be finalized from the blinded feasibility audit before modeling.

### Session inclusion

- identifiable user and session;
- canonical sport family available or mapped to an explicit `other` category;
- at least 600 s between the earliest usable context and the latest target;
- sufficient HR observations to construct the context and all three targets;
- monotonically orderable timestamps after exact-duplicate handling;
- no unresolved unit ambiguity in HR or time;
- context and horizon gaps below the prespecified maximum.

### Observation validity

- HR outside 30–240 bpm is flagged and excluded from target construction rather than silently clipped;
- negative elapsed time or distance increments are flagged;
- impossible or non-monotonic timestamps are audited;
- missing covariates are represented by masks, not filled using future observations.

## Sport-family ontology

Raw labels will be normalized before split assignment. The provisional families are:

1. outdoor cycling;
2. indoor/virtual cycling;
3. running;
4. walking/hiking;
5. swimming;
6. skiing;
7. strength/cross-training;
8. other/unknown.

The final ontology requires a versioned mapping table, minimum user/session support, and manual review of the highest-frequency unmatched labels. Free-text workout titles must not be treated as independent sport types.

## Evaluation protocols

### P1 — Within-user temporal forecasting

Complete sessions are ordered by time for each eligible user. Earlier sessions form development data and later sessions form validation/test data. Windows from one session may not cross partitions.

### P2 — Unseen-user forecasting

Users are assigned to train, validation/calibration, or test before window construction. No test-user session contributes to model fitting, preprocessing, early stopping, or hyperparameter selection.

Two inference settings are reported:

- zero-shot unseen user: no earlier sessions supplied;
- history-informed unseen user: only chronologically earlier sessions supplied to the frozen history encoder.

### P3 — Unseen-sport-family forecasting

A complete canonical sport family is withheld from model fitting. Leave-one-family-out experiments are run only for sport families meeting prespecified minimum support. Held-out sport tokens map to an unknown-sport representation unless a semantic encoding is prospectively justified.

### P4 — Joint user and sport shift

Test examples must belong both to users excluded from fitting and to a sport family excluded from fitting. The user–sport support table must be reported to prevent sparse intersections from being presented as broad generalization.

### P5 — Frozen external validation

The selected Endomondo model, preprocessing parameters, sport mapping, and conformal calibration thresholds are frozen before GoldenCheetah outcomes are inspected. Primary external results use no GoldenCheetah recalibration. A secondary adaptation analysis may use a user-disjoint GoldenCheetah calibration subset and must evaluate on untouched remaining users.

The initial primary external sport scope is outdoor cycling, running, and indoor or virtual cycling. Additional sport families require prespecified minimum user, session, heart-rate, and target-window support. Swimming is secondary at most unless the full census overturns its low audited heart-rate availability.

## Models

### Non-neural baselines

- persistence (last observed HR);
- exponentially weighted moving average;
- autoregressive / autoregressive-with-exogenous-input model;
- XGBoost using identical causal lag summaries.

The executed v0.5.0 persistence, EWMA, and linear-trend references use only the 300 s evaluation-origin subset for reported metrics. EWMA alpha is frozen from the Endomondo unseen-user validation partition using the mean user-session-hierarchical MAE across all three horizons; Endomondo test and GoldenCheetah outcomes are excluded from selection. Full definitions and verified point estimates are recorded in `protocol/NAIVE_BASELINE_SPECIFICATION.md`.

### Neural baselines

- GRU;
- temporal convolutional network (TCN);
- PatchTST or a lightweight Transformer after fixed-grid resampling.

### Main model

The primary architecture will use:

1. a TCN current-session encoder;
2. a user-history encoder with a no-history mask;
3. gated fusion of current context, causal user history, sport family, and missingness indicators;
4. a direct multi-horizon quantile head.

Quantiles `[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]` will support 50%, 80%, and 90% intervals. Split conformalized quantile regression will be applied using a calibration partition that is disjoint at the protocol’s required grouping level.

## Outcomes and statistical analysis

### Primary point outcome

- user-aggregated MAE in bpm at the 5-min horizon on the 300 s evaluation-origin subset.

### Secondary point outcomes

- MAE at 1 and 3 min;
- RMSE, median absolute error, signed bias, and horizon-wise error curves.

### Uncertainty outcomes

- PICP and absolute coverage error at 50%, 80%, and 90%;
- mean and normalized interval width;
- WIS and quantile/pinball loss;
- association between interval width and absolute forecast error.

Metrics are first summarized within sessions and then within users so that long sessions and highly active users do not dominate. Denser 60 s origins are training-only or descriptive unless explicitly labeled. Model differences will use user-clustered bootstrap confidence intervals with fixed seeds. Exact tests and multiplicity handling will be frozen before final experiments.

## Required ablations

- no user-history encoder;
- zeroed/unknown user history;
- no sport-family input;
- no missingness/time-gap masks;
- point head versus quantile head;
- uncalibrated versus conformalized intervals;
- random-window split as a deliberately leaky diagnostic, clearly separated from valid results;
- core shared sensors versus sensor-rich secondary inputs.

## External subgroup reporting

Sport-family and horizon analyses are required. Sex-stratified analysis is exploratory only and will be omitted from inferential claims when subgroup support is inadequate; the current GoldenCheetah subset is strongly male-dominated.

## Scope boundary

This is an observational forecasting study of wearable exercise records. It does not diagnose disease, establish causal physiology, prescribe training, or claim injury prevention. External validity is limited by self-selected platform users, device heterogeneity, missing sensors, and the sport families with adequate support.
