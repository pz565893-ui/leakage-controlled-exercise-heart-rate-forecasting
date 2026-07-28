# Terminology Ledger

This protocol ledger mirrors the manuscript-wide canonical vocabulary in `manuscript/TERMINOLOGY_LEDGER.md`; the manuscript ledger controls if wording differs.

| Canonical term | First-use definition | Avoid / previous variant | Decision |
|---|---|---|---|
| exercise heart-rate forecasting | forecasting future exercise heart rate from causal observations available up to the forecast origin | heart-rate prediction when referring to future values | Use *forecasting* for the task; reserve *prediction* for generic model outputs. |
| forecast origin | the latest time at which input information is available | current time, cut point | Use consistently in task definitions. |
| context window | the 5-min interval ending at the forecast origin | input window, past window | Use *context window*. |
| forecast horizon | 1, 3, or 5 min after the forecast origin | prediction length | Use horizons of 60, 180, and 300 s. |
| user shift | evaluation on users excluded from model fitting and hyperparameter selection | unseen-person split | Use *unseen-user evaluation* as the protocol label. |
| sport shift | evaluation on a canonical sport family excluded from model fitting | unseen activity, unseen sport type | Use *unseen-sport-family evaluation*. |
| joint shift | simultaneous unseen-user and unseen-sport-family evaluation | double distribution shift | Define once, then use *joint shift*. |
| history-masked inference | inference by the history-capable model with completed-workout input deliberately hidden | zero-shot user, cold-start user, forced-zero history | State the unseen-user boundary separately; unseen does not imply no deployment-time history. |
| history-informed inference | inference by the history-capable model using only causally completed prior workouts | personalized unseen user, few-shot user | No model-parameter update occurs at deployment. |
| zero-history-trained model | a separate model trained, selected, and evaluated without prior-workout input | independent zero, always-zero model | Keep distinct from history-masked inference. |
| current-session encoder | temporal encoder applied only to the causal context window | current training encoder | Do not use *training* to mean an exercise session. |
| user-history encoder | encoder of complete sessions strictly preceding the indexed session | personal encoder | History must be causally available. |
| predictive uncertainty | uncertainty attached to a future heart-rate forecast | confidence | Use *prediction interval* rather than *confidence interval*. |
| conformalized quantile regression (CQR) | quantile forecasting followed by split-conformal interval calibration | uncertainty head | Expand at first use. |
| leakage-controlled evaluation | evaluation in which split assignment, preprocessing, history construction, windowing, tuning, and calibration obey causal boundaries | leakage-free | Avoid absolute *leakage-free* claims; controls can be audited but never assumed perfect. |
| internal validation | validation within Endomondo under predefined shift protocols | development validation | Endomondo is the development dataset. |
| frozen cross-source evaluation | Endomondo-fitted models and thresholds applied to GoldenCheetah history-masked forecasts without adaptation or recalibration | external validation, device validation, independent test | This is source transport, not controlled device validation; the final evaluation was retrospectively frozen rather than prospectively sequestered. |
| sport family | a prespecified normalized category such as outdoor cycling, indoor cycling, running, walking/hiking, swimming, or skiing | raw sport label | Raw free-text labels are never treated as distinct sports. |
| heart rate (HR) | beats per minute (bpm) measured during an exercise session | pulse | Expand once and use HR thereafter. |
| mean absolute error (MAE) | mean absolute difference between observed and forecast HR in bpm | average error | Primary point-forecast metric. |
| prediction interval coverage probability (PICP) | empirical proportion of targets contained in a prediction interval | coverage | Report with nominal level and coverage gap. |
| weighted interval score (WIS) | proper score combining interval width and miscoverage | interval loss | Primary distributional score when multiple intervals are available. |
