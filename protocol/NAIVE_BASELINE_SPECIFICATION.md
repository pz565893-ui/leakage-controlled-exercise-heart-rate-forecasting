# Causal no-training baseline specification

**Baseline version:** 0.5.0  
**Input window index:** 0.3.1  
**Input feature series:** 0.4.0  
**Status:** executed and locked; these are reference baselines, not the main model.

## Common input and forecast origins

All baselines use the same thirty right-closed 10 s heart-rate bins covering `(t-300, t]`. Missing bins are skipped rather than filled with future or bidirectional information. Only the prespecified 300 s evaluation-origin subset is used for reported validation and test metrics.

## Models

- **Persistence:** repeats the last observed valid context heart rate at all horizons.
- **EWMA:** traverses valid bins chronologically and skips missing bins. A single smoothing parameter is used at all three horizons.
- **Linear trend:** fits ordinary least squares to valid context heart rate against causal bin-end time and extrapolates directly to +60, +180, and +300 s. Outputs are clipped to the prespecified valid range of 30–240 bpm.

## Frozen EWMA selection

Candidate values were `[0.1, 0.2, 0.3, 0.5, 0.7, 0.9]`. Selection used only Endomondo users in the unseen-user validation partition and the hierarchical MAE averaged across 1-, 3-, and 5-min horizons. The selected value was `alpha = 0.1` (aggregate validation MAE 8.078 bpm). The selection used 123,370 origins from 19,062 sessions and 127 users with accepted validation origins. No Endomondo test or GoldenCheetah outcome was used for selection.

## Aggregation

For each metric, origin-level errors are summarized within a session, session summaries are averaged within a user, and users receive equal weight. RMSE is computed within each session before the same session-then-user averaging. Primary point comparisons will later use user-clustered confidence intervals; the current baseline table contains point estimates only.

## Executed primary-reference results

| Regime | Model | 1-min MAE | 3-min MAE | 5-min MAE |
|---|---|---:|---:|---:|
| Within-user temporal test | Persistence | 6.732 | 8.758 | 9.577 |
| Within-user temporal test | EWMA | 6.994 | 8.173 | 8.944 |
| Unseen-user test | Persistence | 6.559 | 8.627 | 9.599 |
| Unseen-user test | EWMA | 6.654 | 7.902 | 8.798 |
| GoldenCheetah frozen external | Persistence | 7.929 | 11.282 | 12.569 |
| GoldenCheetah frozen external | EWMA | 8.670 | 10.911 | 12.011 |

Persistence is the stronger short-horizon reference, whereas the slowly varying EWMA is stronger at most 3- and 5-min comparisons. Linear trend is retained as a deliberately simple extrapolative baseline but is materially less stable, especially at 5 min. These observations do not establish superiority of the planned learned models.

## Integrity controls

- 1,538,800 expected evaluation origins and 1,538,800 prediction rows;
- zero missing feature series;
- zero context-mask mismatches against the locked origin index;
- zero duplicate prediction keys;
- zero null or out-of-range predictions;
- prediction SQLite integrity check `ok`;
- joined evaluation row count unchanged at 1,538,800;
- two independent hierarchical SQL recomputations matched the saved CSV exactly.

## Reproducible artifacts

- `src/run_naive_baselines.py`
- `tests/test_naive_baselines.py`
- `outputs/models/naive_baseline_tuning_v0_5_0.json`
- `outputs/predictions/naive_baselines_v0_5_0.sqlite`
- `outputs/results/naive_baseline_metrics_v0_5_0.csv`
- `outputs/audit/naive_baselines_v0_5_0.json`
- `notebooks/03_naive_baseline_results.ipynb`
