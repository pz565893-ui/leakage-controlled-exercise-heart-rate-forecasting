# Independent zero-history ablation specification

**Status:** frozen before execution of this added ablation  
**Date frozen:** 2026-07-22  
**Relationship to the main protocol:** reviewer-motivated supplementary analysis; not prospectively registered and not part of the original analysis plan

## Question

Does the history-conditioned training strategy outperform an otherwise matched model trained exclusively for zero-history deployment, or does the observed effect only reflect switching history on and off within a jointly trained checkpoint?

## Models and seeds

Train ten additional models after the ongoing Q1 multi-seed queue has completed:

- unseen-user protocol: seeds 20260722, 20260723, 20260724, 20260725, and 20260726;
- strict within-user temporal protocol: the same five seeds.

The model class, current-workout TCN, residual quantile head, sport representation, partitions, train-only normalization, hierarchical sampling weights, optimizer, learning-rate schedule, and quantile outputs must match the corresponding mixed-history main model. Every training and validation batch must force the history-presence mask to zero. Checkpoint selection must minimize zero-history validation MAE only. Passing `history_dropout=1.0` to the existing mixed-history selection rule is not an acceptable substitute.

Resolved training budget per seed:

- 500,000 sampled training origins per epoch;
- batch size 2,048;
- inference batch size 4,096;
- maximum 40 epochs;
- patience 4;
- initial learning rate 0.001.

## Evaluation

For each protocol and seed, save the checkpoint, resolved configuration, normalization state, audit, test predictions, point metrics, and interval metrics. Apply each unseen-user model to GoldenCheetah with the Endomondo preprocessing and calibration artifacts frozen; do not fine-tune or recalibrate on GoldenCheetah.

The primary endpoint is session-then-user hierarchical MAE at 5 min. The 1- and 3-min MAEs are secondary. Probabilistic secondary outcomes are pinball loss, WIS, empirical PICP, and mean interval width.

For matched seeds, calculate:

1. mixed model, history-informed minus mixed model, zero-history;
2. mixed model, zero-history minus independently trained zero-history model;
3. mixed model, history-informed minus independently trained zero-history model.

The first contrast remains an information-availability effect within one fitted checkpoint. The second estimates any cost of dual-mode training. The third is the net strategy contrast against a separately optimized cold-start model.

## Aggregation and uncertainty

Report every seed and the across-seed median, minimum, and maximum. Seeds characterize optimization variability and must not be treated as independent participants or used for a seed-level significance test. For user-level paired uncertainty, first average each user's paired loss difference across the five matched seeds, then apply 10,000 user-bootstrap replicates to those user-level mean differences. Report percentile 95% intervals and label them as sampling uncertainty conditional on the declared seed set.

## Claim boundary

If the independent comparator is not better, the manuscript may state that history-conditioned training showed a small net incremental effect under the declared architecture and protocols. If it matches or outperforms the history-informed model, the manuscript must retain only the within-checkpoint information-availability finding and explicitly state that the history strategy did not outperform a separately optimized cold-start system. In either case, practical importance must be judged from effect magnitude rather than statistical exclusion of zero alone.
