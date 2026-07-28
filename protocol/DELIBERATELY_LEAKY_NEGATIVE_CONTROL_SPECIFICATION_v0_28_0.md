# Deliberately leaky strict-temporal negative-control specification

**Analysis version:** 0.28.0  
**Status:** retrospective negative-control configuration locked before GPU execution  
**Validity:** invalid for generalization; diagnostic negative control only

## Purpose

This analysis deliberately violates the session-before-window boundary to quantify
the optimism observed when densely overlapping windows from a held-out exercise
session contaminate model fitting, early stopping, and conformal calibration. It
does not estimate a universal leakage bias and cannot enter the valid-model
leaderboard.

The design was added after reviewer-style audit of the completed primary analysis;
it is not a prospective preregistration. Its configuration was locked before any
v0.28 GPU training or v0.28 prediction result was generated.

## Fixed paired test support

The test set remains exactly the 104,144 lower-overlap origins from 16,012 sessions
and 948 users in the locked strict-temporal test. Its ordered NumPy row-index hash
is fixed in `configs/leaky_negative_control_v0_28_0.json`. Exact test rows remain
absent from fitting, normalization, validation, and calibration.

## Deliberate contamination

The contamination pool consists only of the 414,854 non-evaluation 60-s origins
from those same 16,012 test sessions. A fixed SHA-256 hash of the namespace,
session index, and integer origin second assigns these rows 70%/15%/15% to leaky
training, validation, and calibration. The assignment is independent of model
seed. Clean strict-temporal training rows remain in training; clean lower-overlap
validation and calibration rows remain in their original roles.

This construction intentionally produces context overlap and target-time reuse
between fitting records and fixed test records. Every such collision is audited.

## Model and inference

The model is the same quantile TCN used in the zero-history-trained v0.23
sensitivity. History is forced absent in every training, validation, calibration,
and test batch. The model starts from a fresh random initialization for each of
three locked seeds; warm-starting from a clean checkpoint is prohibited.

Input normalization is fitted only on the deliberately contaminated training
partition. CQR thresholds are fitted only on the deliberately contaminated
calibration partition. Because calibration and test share sessions and overlapping
windows, no conformal coverage guarantee is claimed. Raw and post-CQR interval
results are both retained.

## Paired analysis

Each leaky prediction is paired by row index with the same-seed v0.23 clean
zero-history-trained prediction. Losses are averaged within session and then
equally across sessions within user. Each user's paired effect is averaged across
the three matched seeds before a 10,000-replicate percentile user bootstrap.
`leaky - clean < 0` for MAE denotes optimistic error under the deliberately
contaminated pipeline.

Every output must state `valid_for_generalization=false` and
`leaderboard_eligible=false`.
