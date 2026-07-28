# Leakage-Control Contract

Every reported experiment must satisfy the applicable items below. A failed item invalidates the corresponding result until rerun.

## Split before transformation

- Assign complete users, sport families, and sessions to partitions before generating windows.
- Do not allow overlapping windows from one session to appear in multiple partitions.
- Record immutable split manifests with dataset, user ID, session ID, sport family, session time, and partition.
- Detect exact signal duplicates before partition assignment. Exclude cross-user or sport-conflicting duplicate groups; retain at most one deterministic canonical record from a within-user, family-consistent group.

## Causal feature boundary

- For a forecast origin `t`, every dynamic input must have timestamp `<= t`.
- Target timestamps must be strictly greater than `t`.
- Interpolation may use past observations only; centered interpolation and backward filling from future samples are prohibited.
- Full-session totals, mean HR, maximum HR, duration, and post-session summaries are prohibited as inputs.

## User-history boundary

- Historical sessions must end before the indexed session begins.
- Test-user history may be supplied at inference only under the declared history-informed protocol.
- User identity embeddings learned for training users must not create a representation for unseen test users.
- Zero-shot evaluation must use a common unknown-user state and no hidden per-user adaptation.

## Preprocessing boundary

- Fit numerical scalers, learned imputers, feature selectors, and model encoders using development data only.
- Cross-source sport-label harmonization may inspect source label semantics and metadata counts only when it is outcome-blind, rule-based, versioned, and frozen before forecast-error or interval-coverage review. It must never use HR targets, prediction errors, coverage, or downstream model rankings.
- Ontology review-status fields are documentary workflow metadata. A semantic reinterpretation of a locked label requires a new ontology version and downstream rerun; it cannot modify the reported mapping in place.
- Preserve missingness masks.
- Do not normalize using the full indexed session or the full lifetime of a test user.
- Any dataset-specific unit conversion must be rule-based and fixed before outcome comparison.

## Tuning and calibration boundary

- Hyperparameter selection and early stopping use validation data only.
- Conformal thresholds use a dedicated calibration set with the same grouping boundary required by the protocol.
- The primary frozen cross-source evaluation uses Endomondo parameters and thresholds without GoldenCheetah adaptation or recalibration.
- Secondary external recalibration requires user-disjoint calibration and test subsets.

## Evaluation boundary

- Report user-aggregated metrics in addition to window-level descriptive metrics.
- Bootstrap at the user level, not the overlapping-window level.
- Do not select the best seed on the test set; report prespecified seeds or their distribution.
- Do not tune sport-label mappings after viewing forecast errors, coverage, or model performance.

## Automated assertions

The data pipeline must fail if any of the following occur:

1. a user prohibited by the protocol appears in both train and test;
2. a session appears in more than one partition;
3. an input timestamp exceeds its forecast origin;
4. a history session overlaps or follows the indexed session;
5. a target timestamp is not later than the forecast origin;
6. preprocessing metadata lists test examples among fitted records;
7. external-test users appear in the external calibration set;
8. duplicate `(dataset, user_id, session_id, forecast_origin)` keys exist.
9. members of an unresolved exact-signal duplicate group enter analysis;
10. an excluded duplicate copy is used to generate a training, calibration, or test window.

## Deliberately leaky comparator

A random-window split may be run only as a diagnostic demonstration of performance inflation. Its tables and figures must be labeled **invalid for generalization**, and it must never be presented as a competitive primary result.
