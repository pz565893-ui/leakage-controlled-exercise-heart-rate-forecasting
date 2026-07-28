# Protocol deviations and analysis-scope reconciliation

Date: 2026-07-22  
Status: final manuscript reconciliation

## Purpose

`STUDY_PROTOCOL.md` was an internal planning document, not a registered or time-stamped public protocol. It intentionally listed a broad set of desirable analyses before data construction and model execution. `FINAL_ANALYSIS_SPECIFICATION.md` records the analyses retained for the manuscript after feasibility, support, compute, and claim-scope review. This note prevents unexecuted early ideas from being silently presented as completed experiments.

## Executed as planned

- User-history availability was tested by forcing the selected checkpoint to its no-history state and comparing it with completed-session history under strict temporal and unseen-user protocols.
- Core-signal content was tested with the heart-rate-only versus multimodal ablation.
- Calibration was evaluated with both raw and conformalized interval metrics; frozen external results used the internal thresholds without recalibration.
- Five leave-one-sport-family-out experiments, joint user--sport intersections, a dense-origin sensitivity analysis, and descriptive recorded-gender analyses were executed.
- All primary splits, normalization, checkpoint selection, calibration, aggregation, and user-clustered uncertainty boundaries were retained.

## Early planned analyses not promoted to the final manuscript

| Early planning item | Final treatment | Rationale and consequence |
|---|---|---|
| Remove the history encoder entirely | Same checkpoint evaluated with history forced absent | This is the deployment-relevant counterfactual and avoids architecture/retraining differences. It supports an information-availability claim, not a component-capacity claim. |
| Remove sport-family input | Not executed | The manuscript makes no claim that the sport token is necessary or sufficient. Sport shift is evaluated by holding out complete families and mapping them to the unknown token. |
| Remove missingness/time-gap masks | Not executed | Removing masks would conflate representation with dataset-specific missingness. The manuscript claims only causal handling of missingness, not mask superiority. |
| Point head versus quantile head | Point TCN, GRU, Transformer, and XGBoost comparators were executed; an otherwise identical retrained point-head ablation was not | Results support uncertainty-aware output and calibration evaluation, not a causal claim that quantile loss improves MAE. |
| Deliberately leaky random-window split | Not executed | It answers no valid deployment question and would add a potentially misleading headline number. Leakage risk is demonstrated structurally through overlap, duplicate, history, and split audits rather than through an invalid benchmark. |

## Interpretation boundary

These deviations narrow the component-level claims. They do not change the primary deployment questions or the reported temporal, unseen-user, unseen-sport, joint-shift, and frozen-external results. The manuscript must not claim that sport encoding, missingness masks, or the quantile head independently improves point accuracy.
