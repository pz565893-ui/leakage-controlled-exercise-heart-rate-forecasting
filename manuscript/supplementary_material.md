# Supplementary material

## Uncertainty-Aware Exercise Heart-Rate Forecasting under User and Sport Distribution Shifts: A Leakage-Controlled Multi-Dataset Study

**Target journal:** *Biomedical Signal Processing and Control*

## Supplementary methods

All supplementary results use the same session inclusion, exact-signal duplicate control, sport ontology, past-only 5-min context, 1/3/5-min targets, split-before-windowing rule, completed-session history rule, training-only normalization, checkpoint selection, and session-then-user hierarchical aggregation described in the main manuscript. The primary analyses required all three targets, so every horizon uses a common complete-three-target cohort. Table S19 isolates that eligibility decision with a fixed-session, parameter-free persistence diagnostic and a five-seed frozen-main-model sensitivity. The principal history-capable and zero-history-trained TCN models were repeated with five seeds, whereas GRU, point-TCN, and held-sport experiments used three seeds; deterministic baselines have no optimization seed, and explicitly labelled secondary sensitivities use the frozen reference seed 20260722. Seed summaries are reported as medians [minimum--maximum] and are descriptive stability ranges, not confidence intervals.

GoldenCheetah athlete-directory identifiers defined users. Within each directory, the local timestamp encoded by a CSV filename was linked to JSON ride metadata only when exactly one match arose after testing UTC offsets from -14:00 to +14:00 in 15-min steps. Of 51,470 CSVs in 150 directories, 50,002 (97.15%) were linked uniquely; 1,335 records had missing or invalid metadata, 111 had ambiguous matches, 14 were unmatched, and 8 had duplicate metadata matches. Signal-quality and duplicate controls then yielded 32,587 modelling sessions from 144 users. This outcome-blind linkage used no HR target values.

Raw observations were assigned to right-closed 10-s bins and the last valid HR, speed, or altitude value in each bin was retained; empty value fields remained zero with a separate observed/missing mask. The 13 completed-history variables were log prior-session count; log mean and log standard deviation of session duration; mean and standard deviation of prior session-mean HR; mean prior within-session HR standard deviation; mean and standard deviation of prior session-mean speed; mean prior within-session altitude standard deviation; log same-sport session count; same-sport mean HR; same-sport mean speed; and log days since the preceding session start. `log1p` was used for counts, duration quantities, and recency. For unseen test users, earlier workouts in the chronological evaluation stream could update these summaries only after ending; no model parameter was updated. In held-family experiments, the held family was absent from training, validation, calibration, and history updates; its sport token was remapped to learned other/unknown code 0, and 0.1 sport-token dropout exposed that code during training on nonheld families. GoldenCheetah uses history-masked inference from the history-capable model, so it evaluates cross-source transport without prior-workout input rather than transport of the completed-history branch. “Unseen user” means absent from fitting and is not synonymous with cold start or early onboarding.

The main network projected six value/mask channels to 64 channels and used four residual TCN blocks, each with two kernel-3 causal convolutions, channel layer normalization, GELU activation, and 0.1 dropout; block dilations were 1, 2, 4, and 8. The history encoder was a 13--32--32 GELU MLP with a learned 32-dimensional no-history vector. An eight-dimensional sport embedding, log elapsed time, the 64-dimensional current state, the history vector, and its presence mask entered a 96-unit GELU fusion layer with 0.1 dropout and 21 outputs. The mean pinball loss weighted all seven quantiles and three horizons equally; outputs were sorted by quantile. Formal main runs used AdamW with learning rate 0.001, weight decay 0.0001, mixed precision, gradient-norm clipping at 1.0, a plateau scheduler (factor 0.5, patience 1, minimum learning rate 0.00001), batch size 2,048, 500,000 sampled origins per epoch, at most 40 epochs, and early-stopping patience 4. Held-family runs used 250,000 samples per epoch, patience 3, and 0.1 sport-token dropout. GRU used two 64-unit layers; point-TCN used the same 64-channel four-block encoder; Transformer used three 64-dimensional, four-head layers with 128-unit feed-forward sublayers. XGBoost used up to 2,000 depth-8 histogram trees, learning rate 0.03, 0.85 row and column subsampling, and 100-round early stopping.

For the zero-history-trained strategy contrasts, paired per-user differences were first averaged over the five matched seeds and then bootstrapped over users with 10,000 replicates. These intervals therefore quantify user-sampling variation conditional on the declared seed set; seeds were not resampled and full optimization uncertainty is not covered. Reference-seed confidence intervals use 10,000 user-clustered bootstrap replicates conditional on that checkpoint. The mean-effect confidence interval is primary comparison evidence; Holm-adjusted paired Wilcoxon p-values are complementary rank evidence and do not override confidence intervals that include zero. CQR thresholds were estimated by pooling correlated calibration origins and therefore do not provide a finite-sample guarantee for equal-user PICP; interval tables report empirical post-CQR performance.

The authoritative multiseed artifacts are the v0.22 aggregation under `outputs/q1_multiseed_v0_21_0/aggregation` and the v0.23 zero-history-trained aggregation under `outputs/independent_zero_history_v0_23_0/aggregation`. Frozen-prediction v0.24 analyses add five-seed equal-user/equal-session calibration and reference-seed sport-composition standardization of interval metrics; v0.25 provides paired-user comparisons; v0.26 provides an independently calibrated persistence interval baseline without model fitting; and v0.27 provides a post hoc matched-origin sport-availability sensitivity from frozen predictions. Version 0.28 is a separately trained, deliberately invalid negative control in which non-evaluation origins from strict-temporal test sessions contaminate fitting, validation, and calibration; it is excluded from valid-model rankings. Version 0.29 provides a parameter-free, fixed-session target-availability sensitivity reconstructed from the raw HR streams. Version 0.30 applies the five frozen main-model seeds to only the additional horizon-eligible rows, with no training, checkpoint selection, normalization refit, calibration, or external adaptation. Fifteen full-batch inference replays and 45 common-cohort hierarchical-MAE checks reproduced the authoritative values exactly. The v0.24--v0.27 post-processing analyses did not adapt or recalibrate on GoldenCheetah, and v0.29 did not fit, adapt, or calibrate a model. The final reference-seed user/cross-source paired-comparison artifact is `outputs/results/paired_model_comparisons_v0_11_0.csv`; the earlier `paired_user_bootstrap_v0_11_0.csv` is retained for provenance but is superseded because it preceded the final aligned prediction rebuild. All values below are generated directly from the authoritative result files by `src/build_supplementary_material.py`.

## Table S1. Dataset construction and evaluation support

### Table S1a. Session and forecast-origin flow

| Dataset | Sessions entering origin construction | Users with accepted origins | Candidate origins | Accepted origins | Accepted (%) | Complete 300-s origin pool |
| --- | --- | --- | --- | --- | --- | --- |
| Endomondo | 201,823 | 1,085 | 17,192,690 | 5,008,341 | 29.1 | 1,001,128 |
| GoldenCheetah | 32,587 | 144 | 3,213,126 | 2,626,835 | 81.8 | 537,672 |

The session-eligible split contained 1,090 Endomondo users, whereas 1,085 users contributed at least one accepted forecast origin. Table S1a reports users represented after origin acceptance; the unseen-user assignments in Table S1b were made earlier, before origin construction.

### Table S1b. Principal partition support

| Partition/regime | Users | Primary boundary |
| --- | --- | --- |
| Unseen-user training | 759 | User disjoint |
| Unseen-user validation | 129 | User disjoint |
| Unseen-user calibration | 97 | User disjoint |
| Unseen-user test | 105 | User disjoint |
| Strict temporal test | 948 | Later sessions; crossing sessions excluded |
| Frozen GoldenCheetah cross-source | 144 | Three shared sport families; 531,725 origins; 31,851 sessions; history-masked; no adaptation or recalibration |

## Table S2. Full strict-temporal point-forecast metrics

MAE is in beats per minute (bpm). Learned-model entries are medians [seed range], with the number of seeds shown explicitly; deterministic baselines are single fixed evaluations.

| Model | 1-min MAE, bpm | 3-min MAE, bpm | 5-min MAE, bpm | Runs/seeds | Users | Sessions | Origins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| History-quantile TCN (history) | 6.012 [6.004 to 6.057] | 7.557 [7.548 to 7.563] | 8.245 [8.239 to 8.267] | 5 | 948 | 16,012 | 104,144 |
| History-capable TCN (history-masked) | 6.070 [6.044 to 6.106] | 7.650 [7.645 to 7.656] | 8.364 [8.355 to 8.379] | 5 | 948 | 16,012 | 104,144 |
| Zero-history-trained TCN | 6.067 [6.054 to 6.078] | 7.617 [7.606 to 7.632] | 8.332 [8.321 to 8.335] | 5 | 948 | 16,012 | 104,144 |
| GRU | 6.054 [6.048 to 6.082] | 7.581 [7.580 to 7.588] | 8.300 [8.294 to 8.306] | 3 | 948 | 16,012 | 104,144 |
| Point TCN | 6.104 [6.092 to 6.127] | 7.617 [7.613 to 7.625] | 8.333 [8.322 to 8.339] | 3 | 948 | 16,012 | 104,144 |
| Persistence | 6.726 | 8.757 | 9.587 | Deterministic | 948 | 16,012 | 104,144 |
| EWMA | 6.989 | 8.171 | 8.955 | Deterministic | 948 | 16,012 | 104,144 |
| Linear trend | 8.975 | 13.550 | 17.982 | Deterministic | 948 | 16,012 | 104,144 |

## Table S3. Full unseen-user and frozen cross-source point-forecast metrics

All GoldenCheetah rows use frozen Endomondo preprocessing and history-masked checkpoints. No target-source adaptation or recalibration was performed, and these rows do not test transfer of the completed-history branch. Learned-model entries are medians [seed range], with the number of seeds shown explicitly; single-run secondary comparators are labelled.

| Regime and model | 1-min MAE, bpm | 3-min MAE, bpm | 5-min MAE, bpm | Runs/seeds | Users | Sessions | Origins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Unseen user: history-quantile TCN (history) | 5.849 [5.821 to 5.867] | 7.407 [7.394 to 7.420] | 8.157 [8.127 to 8.212] | 5 | 105 | 15,026 | 101,184 |
| Unseen user: history-capable TCN (history-masked) | 5.883 [5.866 to 5.910] | 7.495 [7.475 to 7.514] | 8.303 [8.251 to 8.337] | 5 | 105 | 15,026 | 101,184 |
| Unseen user: zero-history-trained TCN | 5.866 [5.863 to 5.899] | 7.487 [7.463 to 7.494] | 8.285 [8.263 to 8.300] | 5 | 105 | 15,026 | 101,184 |
| Unseen user: GRU | 5.902 [5.860 to 5.915] | 7.423 [7.412 to 7.470] | 8.216 [8.211 to 8.256] | 3 | 105 | 15,026 | 101,184 |
| Unseen user: Point TCN | 5.897 [5.867 to 5.919] | 7.453 [7.448 to 7.478] | 8.257 [8.252 to 8.283] | 3 | 105 | 15,026 | 101,184 |
| Unseen user: XGBoost | 5.926 | 7.493 | 8.354 | 1 (20260722) | 105 | 15,026 | 101,184 |
| Unseen user: Transformer | 5.851 | 7.396 | 8.258 | 1 (20260722) | 105 | 15,026 | 101,184 |
| Unseen user: Persistence | 6.559 | 8.627 | 9.599 | Deterministic | 105 | 15,026 | 101,184 |
| Unseen user: EWMA | 6.654 | 7.902 | 8.798 | Deterministic | 105 | 15,026 | 101,184 |
| Unseen user: Linear trend | 8.552 | 13.122 | 17.538 | Deterministic | 105 | 15,026 | 101,184 |
| Frozen cross-source: history-capable TCN (history-masked) | 7.465 [7.427 to 7.471] | 10.206 [10.192 to 10.231] | 11.214 [11.186 to 11.226] | 5 | 144 | 31,851 | 531,725 |
| Frozen cross-source: zero-history-trained TCN | 7.461 [7.448 to 7.489] | 10.218 [10.202 to 10.250] | 11.226 [11.215 to 11.262] | 5 | 144 | 31,851 | 531,725 |
| Frozen cross-source: GRU | 7.527 [7.501 to 7.547] | 10.245 [10.207 to 10.258] | 11.260 [11.218 to 11.260] | 3 | 144 | 31,851 | 531,725 |
| Frozen cross-source: Point TCN | 7.505 [7.452 to 7.536] | 10.183 [10.174 to 10.229] | 11.217 [11.215 to 11.277] | 3 | 144 | 31,851 | 531,725 |
| Frozen cross-source: XGBoost | 7.654 | 10.437 | 11.556 | 1 (20260722) | 144 | 31,851 | 531,725 |
| Frozen cross-source: Transformer | 7.493 | 10.241 | 11.266 | 1 (20260722) | 144 | 31,851 | 531,725 |
| Frozen cross-source: Persistence | 7.929 | 11.282 | 12.569 | Deterministic | 144 | 31,851 | 531,725 |
| Frozen cross-source: EWMA | 8.670 | 10.911 | 12.011 | Deterministic | 144 | 31,851 | 531,725 |
| Frozen cross-source: Linear trend | 10.897 | 17.400 | 23.012 | Deterministic | 144 | 31,851 | 531,725 |

## Table S4. Raw and conformalized interval performance

### Table S4a. Raw and post-CQR point estimates

PICP is prediction-interval coverage probability; width and conformal adjustment are in bpm. CQR denotes conformalized quantile regression. Entries are medians [seed range] across five seeds; seed ranges are not confidence intervals. Strict-temporal and unseen-user rows use history-informed inference, whereas GoldenCheetah rows use history-masked inference and apply Endomondo-derived adjustments unchanged. Figure 4b instead uses the explicitly matched unseen-user history-masked mode.

| Regime | Mode | Nominal | Intervals | 1 min PICP | 1 min width | 1 min CQR adj. | 3 min PICP | 3 min width | 3 min CQR adj. | 5 min PICP | 5 min width | 5 min CQR adj. | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Strict temporal | History-informed | 50% | Raw quantiles | 0.486 [0.479 to 0.492] | 9.24 [9.17 to 9.33] | 0.000 [0.000 to 0.000] | 0.492 [0.485 to 0.495] | 11.55 [11.47 to 11.59] | 0.000 [0.000 to 0.000] | 0.490 [0.487 to 0.492] | 12.55 [12.37 to 12.61] | 0.000 [0.000 to 0.000] | 5 |
| Strict temporal | History-informed | 50% | CQR | 0.494 [0.491 to 0.494] | 9.34 [9.29 to 9.36] | 0.049 [0.016 to 0.091] | 0.492 [0.487 to 0.496] | 11.58 [11.49 to 11.62] | 0.012 [0.000 to 0.024] | 0.490 [0.488 to 0.492] | 12.57 [12.40 to 12.61] | 0.010 [0.000 to 0.021] | 5 |
| Strict temporal | History-informed | 80% | Raw quantiles | 0.789 [0.786 to 0.792] | 18.33 [18.25 to 18.37] | 0.000 [0.000 to 0.000] | 0.791 [0.788 to 0.795] | 22.97 [22.94 to 23.04] | 0.000 [0.000 to 0.000] | 0.789 [0.789 to 0.792] | 24.95 [24.88 to 25.00] | 0.000 [0.000 to 0.000] | 5 |
| Strict temporal | History-informed | 80% | CQR | 0.796 [0.794 to 0.799] | 18.54 [18.51 to 18.65] | 0.094 [0.086 to 0.196] | 0.794 [0.791 to 0.796] | 23.07 [23.02 to 23.16] | 0.039 [0.016 to 0.076] | 0.791 [0.789 to 0.792] | 24.98 [24.89 to 25.04] | 0.005 [0.000 to 0.052] | 5 |
| Strict temporal | History-informed | 90% | Raw quantiles | 0.893 [0.891 to 0.895] | 24.57 [24.48 to 24.62] | 0.000 [0.000 to 0.000] | 0.895 [0.893 to 0.895] | 30.83 [30.77 to 30.90] | 0.000 [0.000 to 0.000] | 0.892 [0.891 to 0.893] | 33.56 [33.42 to 33.63] | 0.000 [0.000 to 0.000] | 5 |
| Strict temporal | History-informed | 90% | CQR | 0.896 [0.895 to 0.899] | 24.77 [24.74 to 24.92] | 0.102 [0.059 to 0.220] | 0.897 [0.895 to 0.897] | 30.99 [30.88 to 31.13] | 0.070 [0.059 to 0.115] | 0.892 [0.892 to 0.894] | 33.64 [33.53 to 33.77] | 0.031 [0.000 to 0.109] | 5 |
| Unseen user | History-informed | 50% | Raw quantiles | 0.489 [0.478 to 0.498] | 8.99 [8.81 to 9.07] | 0.000 [0.000 to 0.000] | 0.479 [0.470 to 0.495] | 11.12 [10.92 to 11.46] | 0.000 [0.000 to 0.000] | 0.481 [0.476 to 0.491] | 12.02 [11.93 to 12.38] | 0.000 [0.000 to 0.000] | 5 |
| Unseen user | History-informed | 50% | CQR | 0.489 [0.479 to 0.498] | 8.99 [8.85 to 9.07] | 0.003 [0.000 to 0.021] | 0.479 [0.473 to 0.495] | 11.12 [10.97 to 11.46] | 0.000 [0.000 to 0.029] | 0.481 [0.476 to 0.491] | 12.02 [11.93 to 12.38] | 0.000 [0.000 to 0.000] | 5 |
| Unseen user | History-informed | 80% | Raw quantiles | 0.792 [0.786 to 0.806] | 17.76 [17.51 to 18.27] | 0.000 [0.000 to 0.000] | 0.787 [0.781 to 0.800] | 22.08 [21.76 to 22.71] | 0.000 [0.000 to 0.000] | 0.784 [0.769 to 0.786] | 23.97 [23.64 to 24.40] | 0.000 [0.000 to 0.000] | 5 |
| Unseen user | History-informed | 80% | CQR | 0.792 [0.787 to 0.806] | 17.76 [17.55 to 18.27] | 0.000 [0.000 to 0.020] | 0.787 [0.782 to 0.800] | 22.08 [21.85 to 22.71] | 0.000 [0.000 to 0.047] | 0.784 [0.769 to 0.786] | 23.97 [23.64 to 24.40] | 0.000 [0.000 to 0.000] | 5 |
| Unseen user | History-informed | 90% | Raw quantiles | 0.893 [0.889 to 0.902] | 23.75 [23.44 to 24.36] | 0.000 [0.000 to 0.000] | 0.891 [0.887 to 0.897] | 29.75 [29.39 to 30.43] | 0.000 [0.000 to 0.000] | 0.883 [0.876 to 0.885] | 32.24 [31.70 to 32.81] | 0.000 [0.000 to 0.000] | 5 |
| Unseen user | History-informed | 90% | CQR | 0.895 [0.890 to 0.902] | 23.83 [23.67 to 24.36] | 0.021 [0.000 to 0.128] | 0.891 [0.888 to 0.897] | 29.75 [29.52 to 30.43] | 0.000 [0.000 to 0.062] | 0.883 [0.876 to 0.885] | 32.24 [31.98 to 32.81] | 0.000 [0.000 to 0.141] | 5 |
| Frozen cross-source | History-masked | 50% | Raw quantiles | 0.486 [0.477 to 0.500] | 11.12 [10.94 to 11.40] | 0.000 [0.000 to 0.000] | 0.459 [0.455 to 0.475] | 14.24 [14.13 to 14.70] | 0.000 [0.000 to 0.000] | 0.459 [0.450 to 0.465] | 15.57 [15.25 to 15.71] | 0.000 [0.000 to 0.000] | 5 |
| Frozen cross-source | History-masked | 50% | CQR | 0.488 [0.483 to 0.500] | 11.18 [11.05 to 11.40] | 0.021 [0.000 to 0.068] | 0.461 [0.455 to 0.475] | 14.24 [14.15 to 14.70] | 0.000 [0.000 to 0.024] | 0.459 [0.450 to 0.465] | 15.57 [15.25 to 15.71] | 0.000 [0.000 to 0.000] | 5 |
| Frozen cross-source | History-masked | 80% | Raw quantiles | 0.778 [0.770 to 0.795] | 22.10 [21.68 to 22.88] | 0.000 [0.000 to 0.000] | 0.749 [0.743 to 0.762] | 28.14 [27.57 to 28.70] | 0.000 [0.000 to 0.000] | 0.742 [0.736 to 0.749] | 30.70 [29.91 to 31.02] | 0.000 [0.000 to 0.000] | 5 |
| Frozen cross-source | History-masked | 80% | CQR | 0.782 [0.773 to 0.795] | 22.13 [21.83 to 22.88] | 0.000 [0.000 to 0.078] | 0.750 [0.744 to 0.762] | 28.14 [27.64 to 28.70] | 0.000 [0.000 to 0.050] | 0.742 [0.736 to 0.749] | 30.70 [29.91 to 31.02] | 0.000 [0.000 to 0.000] | 5 |
| Frozen cross-source | History-masked | 90% | Raw quantiles | 0.878 [0.873 to 0.888] | 29.45 [28.79 to 30.30] | 0.000 [0.000 to 0.000] | 0.859 [0.852 to 0.863] | 37.50 [36.60 to 37.88] | 0.000 [0.000 to 0.000] | 0.850 [0.846 to 0.854] | 40.56 [39.81 to 40.73] | 0.000 [0.000 to 0.000] | 5 |
| Frozen cross-source | History-masked | 90% | CQR | 0.880 [0.877 to 0.888] | 29.55 [29.27 to 30.30] | 0.040 [0.000 to 0.240] | 0.859 [0.853 to 0.863] | 37.50 [36.76 to 37.88] | 0.062 [0.000 to 0.135] | 0.850 [0.846 to 0.854] | 40.56 [39.81 to 40.73] | 0.000 [0.000 to 0.116] | 5 |

### Table S4b. User-bootstrap uncertainty for primary regime modes

All rows use post-CQR 90% intervals from the frozen reference seed 20260722. The Mode column makes the information state explicit: unseen-user rows here are history-informed, while the matched comparison in Figure 4b is history-masked. Brackets are 95% confidence intervals from 10,000 user bootstrap replicates. Width-error Spearman is first calculated within each eligible user and then averaged across users. The intervals quantify user-sampling variation conditional on that fitted checkpoint and should not be interpreted as multiseed intervals.

| Regime | Mode | Horizon | 90% PICP [95% CI] | 90% width [95% CI], bpm | WIS [95% CI] | Width-error Spearman [95% CI] | Users |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Strict temporal | History-informed | 1 min | 0.897 [0.892, 0.902] | 24.88 [24.25, 25.51] | 3.57 [3.46, 3.68] | 0.327 [0.314, 0.340] | 948 |
| Strict temporal | History-informed | 3 min | 0.896 [0.891, 0.901] | 31.00 [30.27, 31.72] | 4.50 [4.36, 4.63] | 0.285 [0.270, 0.299] | 948 |
| Strict temporal | History-informed | 5 min | 0.892 [0.885, 0.897] | 33.56 [32.81, 34.34] | 4.91 [4.77, 5.06] | 0.263 [0.248, 0.278] | 948 |
| Unseen user | History-informed | 1 min | 0.897 [0.890, 0.903] | 23.83 [22.17, 25.56] | 3.44 [3.20, 3.70] | 0.359 [0.335, 0.382] | 105 |
| Unseen user | History-informed | 3 min | 0.891 [0.882, 0.898] | 29.75 [27.82, 31.73] | 4.39 [4.10, 4.69] | 0.327 [0.302, 0.352] | 105 |
| Unseen user | History-informed | 5 min | 0.883 [0.862, 0.896] | 32.24 [30.27, 34.36] | 4.89 [4.51, 5.36] | 0.289 [0.267, 0.311] | 105 |
| Frozen cross-source | History-masked | 1 min | 0.877 [0.871, 0.883] | 29.49 [28.61, 30.37] | 4.48 [4.30, 4.66] | 0.361 [0.349, 0.374] | 144 |
| Frozen cross-source | History-masked | 3 min | 0.859 [0.851, 0.866] | 37.50 [36.59, 38.41] | 6.10 [5.86, 6.33] | 0.315 [0.301, 0.329] | 144 |
| Frozen cross-source | History-masked | 5 min | 0.850 [0.842, 0.859] | 40.56 [39.61, 41.45] | 6.72 [6.45, 6.99] | 0.272 [0.258, 0.286] | 144 |

## Table S5. Leave-one-sport-family-out and joint-shift results

### Table S5a. Point performance and support

History-informed hierarchical MAE is reported as the median [seed range] across three seeds. Seed ranges are descriptive and are not confidence intervals. Joint intersections below 25 users are explicitly cautionary.

| Regime | Held sport family | 1-min MAE [seed range] | 3-min MAE [seed range] | 5-min MAE [seed range] | Seeds | Users | Sessions | Origins | Support interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Joint user–sport | indoor virtual cycling | 5.919 [5.863 to 5.993] | 8.173 [7.944 to 8.254] | 11.099 [10.967 to 11.360] | 3 | 18 | 234 | 1,523 | Caution (<25 users) |
| Same-user unseen sport | indoor virtual cycling | 6.210 [6.114 to 6.273] | 8.570 [8.494 to 8.690] | 9.060 [8.960 to 9.065] | 3 | 118 | 1,571 | 11,591 | Supported descriptive |
| Joint user–sport | outdoor cycling | 8.133 [8.127 to 8.149] | 9.948 [9.900 to 10.005] | 10.879 [10.861 to 10.902] | 3 | 77 | 6,097 | 37,856 | Supported descriptive |
| Same-user unseen sport | outdoor cycling | 8.146 [8.055 to 8.147] | 9.762 [9.687 to 9.800] | 10.492 [10.399 to 10.540] | 3 | 452 | 40,943 | 236,748 | Supported descriptive |
| Joint user–sport | running | 5.085 [5.064 to 5.155] | 6.583 [6.483 to 6.631] | 7.337 [7.219 to 7.435] | 3 | 88 | 8,431 | 60,542 | Supported descriptive |
| Same-user unseen sport | running | 5.261 [5.251 to 5.264] | 6.869 [6.841 to 6.957] | 7.617 [7.585 to 7.729] | 3 | 448 | 44,652 | 308,648 | Supported descriptive |
| Joint user–sport | strength cross training | 9.395 [9.296 to 9.497] | 11.964 [11.820 to 12.068] | 11.797 [11.719 to 11.978] | 3 | 20 | 112 | 503 | Caution (<25 users) |
| Same-user unseen sport | strength cross training | 9.293 [9.244 to 9.411] | 10.609 [10.494 to 10.722] | 12.012 [12.007 to 12.035] | 3 | 106 | 587 | 3,189 | Supported descriptive |
| Joint user–sport | walking hiking | 6.501 [6.416 to 6.520] | 7.252 [7.117 to 7.423] | 8.842 [8.701 to 8.977] | 3 | 19 | 96 | 428 | Caution (<25 users) |
| Same-user unseen sport | walking hiking | 5.602 [5.599 to 5.612] | 7.731 [7.725 to 7.779] | 8.688 [8.544 to 8.863] | 3 | 144 | 2,063 | 10,149 | Supported descriptive |

### Table S5b. Empirical post-CQR 90% interval performance

PICP and interval width are aggregated within session and then user and reported as medians [seed range] across three seeds. Low-support joint intersections remain cautionary.

| Regime | Held sport family | 1 min PICP [seed range] | 1 min width [seed range] | 3 min PICP [seed range] | 3 min width [seed range] | 5 min PICP [seed range] | 5 min width [seed range] | Seeds | Users | Support interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Joint user–sport | indoor virtual cycling | 0.911 [0.908 to 0.925] | 28.66 [28.36 to 28.83] | 0.865 [0.865 to 0.866] | 37.62 [36.25 to 39.01] | 0.885 [0.820 to 0.908] | 41.27 [39.62 to 43.36] | 3 | 18 | Caution (<25 users) |
| Same-user unseen sport | indoor virtual cycling | 0.920 [0.905 to 0.921] | 27.53 [26.80 to 28.14] | 0.911 [0.902 to 0.911] | 37.16 [35.47 to 37.23] | 0.922 [0.912 to 0.924] | 40.99 [38.97 to 41.71] | 3 | 118 | Supported descriptive |
| Joint user–sport | outdoor cycling | 0.862 [0.859 to 0.878] | 30.03 [29.79 to 31.50] | 0.873 [0.867 to 0.878] | 38.19 [37.25 to 39.13] | 0.859 [0.855 to 0.866] | 41.49 [40.32 to 42.11] | 3 | 77 | Supported descriptive |
| Same-user unseen sport | outdoor cycling | 0.859 [0.852 to 0.875] | 30.50 [29.53 to 31.75] | 0.876 [0.871 to 0.885] | 38.86 [37.25 to 39.54] | 0.879 [0.874 to 0.886] | 42.27 [40.41 to 42.57] | 3 | 452 | Supported descriptive |
| Joint user–sport | running | 0.939 [0.928 to 0.957] | 26.08 [25.19 to 28.34] | 0.931 [0.919 to 0.950] | 34.69 [32.92 to 37.06] | 0.919 [0.915 to 0.941] | 37.95 [35.83 to 40.63] | 3 | 88 | Supported descriptive |
| Same-user unseen sport | running | 0.934 [0.923 to 0.949] | 26.59 [25.24 to 28.63] | 0.926 [0.918 to 0.942] | 35.43 [32.96 to 37.48] | 0.910 [0.905 to 0.932] | 38.74 [35.88 to 41.09] | 3 | 448 | Supported descriptive |
| Joint user–sport | strength cross training | 0.915 [0.914 to 0.938] | 34.72 [33.95 to 35.48] | 0.870 [0.865 to 0.917] | 44.08 [42.22 to 45.13] | 0.862 [0.853 to 0.869] | 48.72 [45.91 to 49.32] | 3 | 20 | Caution (<25 users) |
| Same-user unseen sport | strength cross training | 0.886 [0.873 to 0.898] | 35.78 [35.35 to 36.72] | 0.886 [0.869 to 0.898] | 44.66 [43.83 to 46.31] | 0.883 [0.880 to 0.891] | 49.06 [47.39 to 50.41] | 3 | 106 | Supported descriptive |
| Joint user–sport | walking hiking | 0.918 [0.917 to 0.938] | 30.36 [29.51 to 30.52] | 0.925 [0.918 to 0.930] | 39.86 [38.68 to 40.09] | 0.940 [0.937 to 0.941] | 45.05 [43.73 to 45.54] | 3 | 19 | Caution (<25 users) |
| Same-user unseen sport | walking hiking | 0.936 [0.928 to 0.946] | 31.09 [29.56 to 31.11] | 0.937 [0.918 to 0.944] | 41.12 [39.14 to 41.62] | 0.935 [0.925 to 0.950] | 46.94 [44.41 to 47.01] | 3 | 144 | Supported descriptive |

### Table S5c. Three-seed paired-user main-versus-EWMA effects

Within each seed, MAE differences were aggregated within session and then user; each user's effect was averaged across the three matched seeds before 10,000 user-bootstrap resamples. Negative values favour the history-informed main model. Joint user--sport rows are exploratory regardless of user count, and rows below 25 users receive an additional caution flag.

| Regime | Held sport family | Horizon | Main minus aligned EWMA MAE [95% CI], bpm | Users | Matched seeds | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Joint user--sport (exploratory) | indoor virtual cycling | 1 min | -1.879 [-3.430, -0.526] | 18 | 3 | Caution (<25 users) |
| Joint user--sport (exploratory) | indoor virtual cycling | 3 min | -1.534 [-2.954, -0.399] | 18 | 3 | Caution (<25 users) |
| Joint user--sport (exploratory) | indoor virtual cycling | 5 min | 0.136 [-1.409, 1.955] | 18 | 3 | Caution (<25 users) |
| Joint user--sport (exploratory) | outdoor cycling | 1 min | -0.654 [-0.936, -0.368] | 77 | 3 | Exploratory |
| Joint user--sport (exploratory) | outdoor cycling | 3 min | -0.085 [-0.282, 0.108] | 77 | 3 | Exploratory |
| Joint user--sport (exploratory) | outdoor cycling | 5 min | -0.303 [-0.726, -0.008] | 77 | 3 | Exploratory |
| Joint user--sport (exploratory) | running | 1 min | -0.405 [-0.605, -0.106] | 88 | 3 | Exploratory |
| Joint user--sport (exploratory) | running | 3 min | 0.206 [-0.113, 0.675] | 88 | 3 | Exploratory |
| Joint user--sport (exploratory) | running | 5 min | 0.334 [0.001, 0.802] | 88 | 3 | Exploratory |
| Joint user--sport (exploratory) | strength cross training | 1 min | 0.192 [-0.631, 1.098] | 20 | 3 | Caution (<25 users) |
| Joint user--sport (exploratory) | strength cross training | 3 min | 0.264 [-0.708, 1.419] | 20 | 3 | Caution (<25 users) |
| Joint user--sport (exploratory) | strength cross training | 5 min | 0.857 [-0.039, 1.990] | 20 | 3 | Caution (<25 users) |
| Joint user--sport (exploratory) | walking hiking | 1 min | -0.086 [-0.754, 0.540] | 19 | 3 | Caution (<25 users) |
| Joint user--sport (exploratory) | walking hiking | 3 min | 0.313 [-0.477, 1.045] | 19 | 3 | Caution (<25 users) |
| Joint user--sport (exploratory) | walking hiking | 5 min | 1.063 [0.439, 1.713] | 19 | 3 | Caution (<25 users) |
| Same-user held sport | indoor virtual cycling | 1 min | -0.875 [-1.195, -0.552] | 118 | 3 | Supported descriptive |
| Same-user held sport | indoor virtual cycling | 3 min | -0.447 [-0.694, -0.210] | 118 | 3 | Supported descriptive |
| Same-user held sport | indoor virtual cycling | 5 min | -0.413 [-0.698, -0.139] | 118 | 3 | Supported descriptive |
| Same-user held sport | outdoor cycling | 1 min | -0.435 [-0.545, -0.317] | 452 | 3 | Supported descriptive |
| Same-user held sport | outdoor cycling | 3 min | -0.107 [-0.212, -0.003] | 452 | 3 | Supported descriptive |
| Same-user held sport | outdoor cycling | 5 min | -0.137 [-0.250, -0.023] | 452 | 3 | Supported descriptive |
| Same-user held sport | running | 1 min | -0.609 [-0.676, -0.545] | 448 | 3 | Supported descriptive |
| Same-user held sport | running | 3 min | -0.064 [-0.118, -0.009] | 448 | 3 | Supported descriptive |
| Same-user held sport | running | 5 min | 0.103 [0.045, 0.162] | 448 | 3 | Supported descriptive |
| Same-user held sport | strength cross training | 1 min | -0.458 [-0.987, 0.117] | 106 | 3 | Supported descriptive |
| Same-user held sport | strength cross training | 3 min | -0.122 [-0.568, 0.338] | 106 | 3 | Supported descriptive |
| Same-user held sport | strength cross training | 5 min | 0.014 [-0.406, 0.483] | 106 | 3 | Supported descriptive |
| Same-user held sport | walking hiking | 1 min | -0.286 [-0.602, 0.074] | 144 | 3 | Supported descriptive |
| Same-user held sport | walking hiking | 3 min | 0.465 [0.145, 0.801] | 144 | 3 | Supported descriptive |
| Same-user held sport | walking hiking | 5 min | 0.965 [0.653, 1.286] | 144 | 3 | Supported descriptive |

## Table S6. Model and ablation effects

### Table S6a. Seed-paired main-versus-comparator effects

Delta is MAE(left strategy) minus MAE(right strategy); negative values favour the left strategy. History contrasts use matched history-capable checkpoints, whereas comparator contrasts use history-masked inference. Entries are medians [seed range] over matched seeds and are descriptive rather than inferential confidence intervals.

| Comparison (left minus right) | Regime | Horizon | Median ΔMAE [seed range], bpm | Matched seeds |
| --- | --- | --- | --- | --- |
| History-informed − history-masked | Strict temporal | 1 min | -0.048 [-0.058 to -0.041] | 5 |
| History-informed − history-masked | Strict temporal | 3 min | -0.090 [-0.108 to -0.088] | 5 |
| History-informed − history-masked | Strict temporal | 5 min | -0.111 [-0.121 to -0.103] | 5 |
| History-informed − history-masked | Unseen user | 1 min | -0.041 [-0.044 to -0.034] | 5 |
| History-informed − history-masked | Unseen user | 3 min | -0.088 [-0.094 to -0.081] | 5 |
| History-informed − history-masked | Unseen user | 5 min | -0.124 [-0.147 to -0.119] | 5 |
| History-masked − GRU | Strict temporal | 1 min | 0.022 [-0.026 to 0.031] | 3 |
| History-masked − GRU | Strict temporal | 3 min | 0.069 [0.068 to 0.070] | 3 |
| History-masked − GRU | Strict temporal | 5 min | 0.066 [0.052 to 0.070] | 3 |
| History-masked − TCN | Strict temporal | 1 min | -0.022 [-0.071 to -0.020] | 3 |
| History-masked − TCN | Strict temporal | 3 min | 0.033 [0.024 to 0.043] | 3 |
| History-masked − TCN | Strict temporal | 5 min | 0.027 [0.026 to 0.042] | 3 |
| History-masked − GRU | Frozen cross-source | 1 min | -0.082 [-0.101 to -0.046] | 3 |
| History-masked − GRU | Frozen cross-source | 3 min | -0.048 [-0.052 to 0.005] | 3 |
| History-masked − GRU | Frozen cross-source | 5 min | -0.047 [-0.069 to -0.004] | 3 |
| History-masked − GRU | Unseen user | 1 min | 0.008 [-0.017 to 0.009] | 3 |
| History-masked − GRU | Unseen user | 3 min | 0.072 [0.005 to 0.091] | 3 |
| History-masked − GRU | Unseen user | 5 min | 0.048 [0.035 to 0.126] | 3 |
| History-masked − TCN | Frozen cross-source | 1 min | -0.071 [-0.078 to 0.002] | 3 |
| History-masked − TCN | Frozen cross-source | 3 min | 0.029 [-0.031 to 0.032] | 3 |
| History-masked − TCN | Frozen cross-source | 5 min | -0.003 [-0.085 to -0.001] | 3 |
| History-masked − TCN | Unseen user | 1 min | 0.002 [-0.021 to 0.013] | 3 |
| History-masked − TCN | Unseen user | 3 min | 0.035 [-0.003 to 0.060] | 3 |
| History-masked − TCN | Unseen user | 5 min | 0.021 [-0.001 to 0.080] | 3 |

### Table S6b. Zero-history-trained strategy contrasts

Delta is MAE(strategy A) minus MAE(strategy B); negative values favour strategy A. Paired per-user differences were averaged over five matched seeds before 10,000 user-bootstrap resamples. The resulting 95% intervals condition on the declared seed set and do not resample seeds.

| Strategy contrast (left minus right) | Regime | Horizon | Seed-averaged user ΔMAE [95% CI], bpm | Users | Matched seeds/user | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| History-informed − zero-history-trained | Strict temporal | 1 min | -0.044 [-0.059, -0.029] | 948 | 5 | CI excludes 0 |
| History-informed − history-masked | Strict temporal | 1 min | -0.048 [-0.063, -0.034] | 948 | 5 | CI excludes 0 |
| History-masked − zero-history-trained | Strict temporal | 1 min | 0.004 [-0.008, 0.015] | 948 | 5 | CI includes 0 |
| History-informed − zero-history-trained | Strict temporal | 3 min | -0.063 [-0.091, -0.034] | 948 | 5 | CI excludes 0 |
| History-informed − history-masked | Strict temporal | 3 min | -0.094 [-0.121, -0.068] | 948 | 5 | CI excludes 0 |
| History-masked − zero-history-trained | Strict temporal | 3 min | 0.031 [0.015, 0.047] | 948 | 5 | CI excludes 0 |
| History-informed − zero-history-trained | Strict temporal | 5 min | -0.079 [-0.113, -0.046] | 948 | 5 | CI excludes 0 |
| History-informed − history-masked | Strict temporal | 5 min | -0.113 [-0.148, -0.078] | 948 | 5 | CI excludes 0 |
| History-masked − zero-history-trained | Strict temporal | 5 min | 0.034 [0.017, 0.051] | 948 | 5 | CI excludes 0 |
| History-masked − zero-history-trained | Frozen cross-source | 1 min | -0.006 [-0.015, 0.003] | 144 | 5 | CI includes 0 |
| History-masked − zero-history-trained | Frozen cross-source | 3 min | -0.014 [-0.029, 0.002] | 144 | 5 | CI includes 0 |
| History-masked − zero-history-trained | Frozen cross-source | 5 min | -0.024 [-0.041, -0.008] | 144 | 5 | CI excludes 0 |
| History-informed − zero-history-trained | Unseen user | 1 min | -0.029 [-0.056, 0.006] | 105 | 5 | CI includes 0 |
| History-informed − history-masked | Unseen user | 1 min | -0.040 [-0.066, -0.013] | 105 | 5 | CI excludes 0 |
| History-masked − zero-history-trained | Unseen user | 1 min | 0.011 [-0.006, 0.029] | 105 | 5 | CI includes 0 |
| History-informed − zero-history-trained | Unseen user | 3 min | -0.076 [-0.124, -0.036] | 105 | 5 | CI excludes 0 |
| History-informed − history-masked | Unseen user | 3 min | -0.088 [-0.133, -0.048] | 105 | 5 | CI excludes 0 |
| History-masked − zero-history-trained | Unseen user | 3 min | 0.012 [-0.032, 0.042] | 105 | 5 | CI includes 0 |
| History-informed − zero-history-trained | Unseen user | 5 min | -0.112 [-0.187, -0.048] | 105 | 5 | CI excludes 0 |
| History-informed − history-masked | Unseen user | 5 min | -0.130 [-0.223, -0.059] | 105 | 5 | CI excludes 0 |
| History-masked − zero-history-trained | Unseen user | 5 min | 0.018 [-0.031, 0.062] | 105 | 5 | CI includes 0 |

### Table S6c. Three-seed paired-user main-versus-GRU/TCN effects

The history-capable main model is evaluated with prior-workout input masked to match the established architecture-comparison estimand. Per-user effects were averaged across three matched seeds before 10,000 user-bootstrap resamples. Negative values favour the main model; these post hoc intervals condition on the declared seeds.

| Regime | Main-model mode | Comparator | Horizon | Main MAE, bpm | Comparator MAE, bpm | Main minus comparator MAE [95% CI], bpm | Users | Matched seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen cross-source | History-masked | GRU | 1 min | 7.449 | 7.525 | -0.076 [-0.095, -0.059] | 144 | 3 |
| Frozen cross-source | History-masked | GRU | 3 min | 10.205 | 10.237 | -0.032 [-0.061, -0.003] | 144 | 3 |
| Frozen cross-source | History-masked | GRU | 5 min | 11.206 | 11.246 | -0.040 [-0.070, -0.010] | 144 | 3 |
| Frozen cross-source | History-masked | TCN | 1 min | 7.449 | 7.498 | -0.049 [-0.067, -0.033] | 144 | 3 |
| Frozen cross-source | History-masked | TCN | 3 min | 10.205 | 10.195 | 0.010 [-0.009, 0.028] | 144 | 3 |
| Frozen cross-source | History-masked | TCN | 5 min | 11.206 | 11.236 | -0.030 [-0.050, -0.011] | 144 | 3 |
| Strict temporal | History-masked | GRU | 1 min | 6.070 | 6.061 | 0.009 [-0.017, 0.035] | 948 | 3 |
| Strict temporal | History-masked | GRU | 3 min | 7.652 | 7.583 | 0.069 [0.038, 0.100] | 948 | 3 |
| Strict temporal | History-masked | GRU | 5 min | 8.363 | 8.300 | 0.063 [0.030, 0.095] | 948 | 3 |
| Strict temporal | History-masked | TCN | 1 min | 6.070 | 6.108 | -0.037 [-0.056, -0.019] | 948 | 3 |
| Strict temporal | History-masked | TCN | 3 min | 7.652 | 7.618 | 0.033 [0.011, 0.056] | 948 | 3 |
| Strict temporal | History-masked | TCN | 5 min | 8.363 | 8.331 | 0.032 [0.005, 0.058] | 948 | 3 |
| Unseen user | History-masked | GRU | 1 min | 5.892 | 5.892 | -0.000 [-0.034, 0.034] | 105 | 3 |
| Unseen user | History-masked | GRU | 3 min | 7.491 | 7.435 | 0.056 [-0.014, 0.122] | 105 | 3 |
| Unseen user | History-masked | GRU | 5 min | 8.297 | 8.227 | 0.070 [-0.002, 0.151] | 105 | 3 |
| Unseen user | History-masked | TCN | 1 min | 5.892 | 5.894 | -0.002 [-0.027, 0.023] | 105 | 3 |
| Unseen user | History-masked | TCN | 3 min | 7.491 | 7.460 | 0.031 [-0.007, 0.067] | 105 | 3 |
| Unseen user | History-masked | TCN | 5 min | 8.297 | 8.264 | 0.033 [-0.012, 0.091] | 105 | 3 |

### Table S6d. Reference-seed signal-ablation effects

These paired user-level multimodal-minus-HR-only effects use the frozen reference seed 20260722. Confidence intervals, rather than rank-test p-values alone, govern mean-effect claims.

| Comparison family | Horizon | ΔMAE [95% CI], bpm | Holm-adjusted Wilcoxon p | Users | Reference-seed interpretation |
| --- | --- | --- | --- | --- | --- |
| unseen_history_multimodal_vs_hr_only | 1 min | -0.002 [-0.057, 0.080] | 0.006 | 105 | CI includes 0 |
| unseen_history_multimodal_vs_hr_only | 3 min | -0.025 [-0.090, 0.065] | 0.002 | 105 | CI includes 0 |
| unseen_history_multimodal_vs_hr_only | 5 min | -0.071 [-0.143, 0.023] | 8.14e-06 | 105 | CI includes 0 |
| unseen_zero_multimodal_vs_hr_only | 1 min | -0.008 [-0.063, 0.072] | 0.001 | 105 | CI includes 0 |
| unseen_zero_multimodal_vs_hr_only | 3 min | -0.029 [-0.093, 0.058] | 4.70e-04 | 105 | CI includes 0 |
| unseen_zero_multimodal_vs_hr_only | 5 min | -0.088 [-0.165, 0.005] | 6.75e-06 | 105 | CI includes 0 |
| external_zero_multimodal_vs_hr_only | 1 min | 0.006 [-0.016, 0.029] | 0.938 | 144 | CI includes 0 |
| external_zero_multimodal_vs_hr_only | 3 min | -0.021 [-0.062, 0.019] | 0.778 | 144 | CI includes 0 |
| external_zero_multimodal_vs_hr_only | 5 min | -0.058 [-0.099, -0.019] | 0.026 | 144 | CI excludes 0 |

## Table S7. Evaluation-origin stride sensitivity

All predictions use the frozen unseen-user reference checkpoint (seed 20260722). Values are hierarchical MAE in bpm.

| Mode | Horizon | 300-s-origin MAE | 60-s-origin MAE | Dense minus standard |
| --- | --- | --- | --- | --- |
| History-masked | 1 min | 5.868 | 5.938 | 0.070 |
| History-masked | 3 min | 7.484 | 7.446 | -0.037 |
| History-masked | 5 min | 8.251 | 8.145 | -0.106 |
| History-informed | 1 min | 5.832 | 5.906 | 0.074 |
| History-informed | 3 min | 7.397 | 7.338 | -0.059 |
| History-informed | 5 min | 8.127 | 8.016 | -0.111 |

## Table S8. Recorded-gender descriptive contrasts

These frozen-reference-seed (20260722) results are unadjusted platform-recorded subgroup descriptions, not biological, causal, fairness, or clinical comparisons. Every confidence interval includes zero; the unseen-user recorded-female subgroup contains only 10 users.

| Regime | Mode | Horizon | Female minus male MAE [95% CI], bpm | User support | Status |
| --- | --- | --- | --- | --- | --- |
| Unseen user | History-informed | 1 min | 1.289 [-0.164, 2.779] | 10 recorded female / 91 recorded male | exploratory small subgroup |
| Unseen user | History-informed | 3 min | 1.708 [-0.300, 3.768] | 10 recorded female / 91 recorded male | exploratory small subgroup |
| Unseen user | History-informed | 5 min | 1.426 [-0.770, 3.774] | 10 recorded female / 91 recorded male | exploratory small subgroup |
| Unseen user | History-masked | 1 min | 1.329 [-0.121, 2.801] | 10 recorded female / 91 recorded male | exploratory small subgroup |
| Unseen user | History-masked | 3 min | 1.704 [-0.281, 3.793] | 10 recorded female / 91 recorded male | exploratory small subgroup |
| Unseen user | History-masked | 5 min | 1.417 [-0.886, 3.810] | 10 recorded female / 91 recorded male | exploratory small subgroup |
| Strict temporal | History-informed | 1 min | 0.084 [-0.691, 0.957] | 79 recorded female / 857 recorded male | supported descriptive |
| Strict temporal | History-informed | 3 min | 0.113 [-0.884, 1.201] | 79 recorded female / 857 recorded male | supported descriptive |
| Strict temporal | History-informed | 5 min | -0.331 [-1.331, 0.706] | 79 recorded female / 857 recorded male | supported descriptive |
| Strict temporal | History-masked | 1 min | 0.102 [-0.687, 0.984] | 79 recorded female / 857 recorded male | supported descriptive |
| Strict temporal | History-masked | 3 min | 0.034 [-0.984, 1.132] | 79 recorded female / 857 recorded male | supported descriptive |
| Strict temporal | History-masked | 5 min | -0.422 [-1.414, 0.594] | 79 recorded female / 857 recorded male | supported descriptive |

## Table S9. Targeted evidence comparison with direct HR-modelling studies

The comparison uses the frozen 41-record project bibliography originally reconciled to the author-approved Zotero collection, together with a documented targeted update conducted on 23 July 2026; it is not a systematic review and cannot support a global first/only claim. Later unrelated Zotero collection-membership drift is audited separately and did not alter this bibliography. `NR` means not reported or not established from the inspected primary evidence; it does not mean no. Current-window HR estimation, past-only future forecasting, and forecasting with known future route information are kept distinct. Study names refer to the numbered references in the main manuscript.

| Study | Task | Prediction range/target | User boundary | Individual history | Held-sport evaluation | External-data design | Predictive intervals |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Present study | Future recorded exercise-HR forecasting | +1/+3/+5 min | Strict temporal + unseen-user | Completed prior workouts | Five held families + joint user-sport shift | Frozen Endomondo-to-GoldenCheetah cross-source evaluation | 50/80/90% empirical post-CQR intervals |
| Qiu et al., 2021 | Personalized mountain-biking HR forecasting | Future HR along one course; physical lead NR | No; one cyclist, chronological first-80/last-20 split | Personalized course/ride data; completed-workout encoder no | No; mountain biking on one course | No; one rider and course | No calibrated PI documented |
| Gilbert et al., 2022 | Biking HR forecasting with future gradient values | Up to 10 min; future course gradient is an input | Participant-independent boundary NR | Current ride plus future route information | No; biking only | No frozen source transfer documented | No calibrated PI documented |
| Fedorin et al., 2021 | HIIT HR-trend forecasting from consumer wearables | Future trend; exact physical lead NR from inspected record | Participant split NR | Completed-workout history NR | No held-sport protocol documented | No frozen source transfer documented | Predictive interval/coverage NR |
| Ni et al., 2019 | Personalized Endomondo speed/HR modelling | Full profile; next 10-s sample | Within-user chronological; not unseen-user | User representation + most recent workout | No held-sport test documented | No independent source | No documented interval |
| Nazaret et al., 2023 | Future-run HR profile modelling | Whole run, up to 2 h | Within-person temporal; not unseen-user | Prior-workout history encoder | No; outdoor running only | No independent source documented | No; fixed +/-5-bpm band is not a PI |
| Hallgrímsson et al., 2018 | Minute-level free-living HR modelling | Future lead time NR | Same-person 2017-to-2018; not unseen-user | Longitudinal participant signature | No held-sport test documented | No frozen HR-transfer dataset | NR |
| Pacheco et al., 2024 | Exercise HR estimation from accelerometry/demographics | Several minutes; exact lead definition NR | NR | Online within-workout adaptation | NR | Five datasets; frozen source transfer NR | NR |
| Reiss et al., 2019 | PPG/accelerometer HR estimation | Current 8-s window; not future forecasting | Leave-one-session-out; subject-independent | No personalization | No held-activity transfer | Datasets analysed separately; no frozen transfer | No |
| Kayange et al., 2024 | Personalized whole-workout HR modelling on FitRec | Whole profile; fixed +1/+3/+5-min leads NR | User-grouped 80/20; no independent final test | Completed recent/past workouts | NR; no held-sport test reported | No; FitRec only | No calibrated PI; fixed +/-5-bpm band |
| Namazi et al., 2025 | Multivariate sports-HR prediction from HR/BR/RR | Future epochs; physical lead time NR | 80/20 split; split unit NR | Completed-session/user history NR | NR; no held-sport protocol reported | No; one Sport Database source | NR |
| De Sabbata & Simonini, 2025 | Per-user ARIMA/random-walk HR forecasting | Next 1 min; 15-to-150-min rolling inputs | Per-user chronological; no unseen-user test | Per-user fitting + recent same-user window | No | No; two datasets fitted separately per user | No |
| Mateescu et al., 2025 | Activity-conditioned Transformer/diffusion forecasting | L input steps to L future steps; L/time unit NR | 65/15/20 split; split unit NR | No completed-session history; current context only | No; activity-specific, not held-out | No; one private Fitbit cohort | No; sample median only, no coverage |
| Zhang et al., 2026 | HR(t) estimation from contemporaneous VO2(t) | 0 s; not future forecasting | Participant-wise 80/20; no unseen-user test | Participant-specific fit; no workout-history encoder | No | No; both sources used in development | No |
| Namazi, 2022 | Univariate running-HR forecasting | Past 1,500 s to next 30 s | Participant-disjoint split NR | Same-record past 1,500 s; not completed workouts | No; running only | No; one source | No; copula draws averaged to a point |
| Zhu et al., 2022 | HR + wrist-inertial exercise-HR forecasting | +5/+7/+10/+15/+20/+25 s | Yes; nine-fold participant split | No; current 5.12-s sensor window only | No; separate model per activity | NR; TicWatch test not established as frozen | No |

## Table S10. Past-only completed-workout history availability

Counts are evaluated over unique 300-s-reporting test sessions. Prior-session counts include only workouts that ended no later than the current workout start. Q1 and Q3 denote session-level quartiles. A history-informed checkpoint still uses its explicit no-history state when no completed workout is available.

| Regime | Users | Sessions | Sessions with history | Prior sessions, median [Q1, Q3] | 0 prior | 1-4 prior | 5-9 prior | 10+ prior | Users with any history | Users with history in all test sessions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Strict temporal test | 948 | 16,012 | 100.0% | 269.0 [148.0, 438.0] | 0 | 5 | 20 | 15,987 | 948 | 948 |
| Unseen-user test | 105 | 15,026 | 99.3% | 98.0 [41.0, 196.0] | 101 | 377 | 477 | 14,071 | 103 | 5 |

## Table S11. Calibration-estimand and clustered-calibration sensitivity

### Table S11a. Five-seed origin-pooled versus equal-user/session calibration

This five-seed sensitivity gives each calibration user equal influence by weighting sessions equally within user and origins equally within session. Values are medians [seed ranges]. The resulting threshold is a weighted empirical estimand-matching analysis, not a distribution-free guarantee. GoldenCheetah rows reuse the corresponding Endomondo history-masked thresholds without target-data recalibration. Strict-temporal calibration predictions were not persisted and were not reconstructed from test targets.

| Evaluation regime | Mode | Horizon | Calibration analysis | Adjustment, median [seed range], bpm | 90% PICP, median [seed range] | Absolute coverage error, median [seed range] | Width, median [seed range], bpm | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen cross-source | History-masked | 1 min | Origin-pooled CQR | 0.040 [0.000 to 0.240] | 0.880 [0.877 to 0.888] | 0.020 [0.012 to 0.023] | 29.55 [29.27 to 30.30] | 5 |
| Frozen cross-source | History-masked | 1 min | Equal-user/session empirical | 0.586 [0.215 to 0.633] | 0.892 [0.886 to 0.893] | 0.008 [0.007 to 0.014] | 30.69 [29.89 to 30.73] | 5 |
| Frozen cross-source | History-masked | 3 min | Origin-pooled CQR | 0.062 [0.000 to 0.135] | 0.859 [0.853 to 0.863] | 0.041 [0.037 to 0.047] | 37.50 [36.76 to 37.88] | 5 |
| Frozen cross-source | History-masked | 3 min | Equal-user/session empirical | 0.633 [0.370 to 0.812] | 0.870 [0.865 to 0.875] | 0.030 [0.025 to 0.035] | 38.62 [37.87 to 39.22] | 5 |
| Frozen cross-source | History-masked | 5 min | Origin-pooled CQR | 0.000 [0.000 to 0.116] | 0.850 [0.846 to 0.854] | 0.050 [0.046 to 0.054] | 40.56 [39.81 to 40.73] | 5 |
| Frozen cross-source | History-masked | 5 min | Equal-user/session empirical | 0.992 [0.565 to 1.068] | 0.866 [0.864 to 0.869] | 0.034 [0.031 to 0.036] | 42.12 [41.85 to 42.71] | 5 |
| Unseen user | History-informed | 1 min | Origin-pooled CQR | 0.021 [0.000 to 0.128] | 0.895 [0.890 to 0.902] | 0.005 [0.002 to 0.010] | 23.83 [23.67 to 24.36] | 5 |
| Unseen user | History-informed | 1 min | Equal-user/session empirical | 0.304 [0.141 to 0.320] | 0.901 [0.898 to 0.906] | 0.002 [0.000 to 0.006] | 24.24 [24.08 to 24.65] | 5 |
| Unseen user | History-informed | 3 min | Origin-pooled CQR | 0.000 [0.000 to 0.062] | 0.891 [0.888 to 0.897] | 0.009 [0.003 to 0.012] | 29.75 [29.52 to 30.43] | 5 |
| Unseen user | History-informed | 3 min | Equal-user/session empirical | 0.492 [0.205 to 0.625] | 0.902 [0.899 to 0.904] | 0.002 [0.001 to 0.004] | 30.64 [30.46 to 31.27] | 5 |
| Unseen user | History-informed | 5 min | Origin-pooled CQR | 0.000 [0.000 to 0.141] | 0.883 [0.876 to 0.885] | 0.017 [0.015 to 0.024] | 32.24 [31.98 to 32.81] | 5 |
| Unseen user | History-informed | 5 min | Equal-user/session empirical | 0.555 [0.434 to 0.844] | 0.893 [0.891 to 0.895] | 0.007 [0.005 to 0.009] | 33.67 [33.13 to 33.75] | 5 |
| Unseen user | History-masked | 1 min | Origin-pooled CQR | 0.040 [0.000 to 0.240] | 0.892 [0.889 to 0.901] | 0.008 [0.001 to 0.011] | 23.57 [23.42 to 24.14] | 5 |
| Unseen user | History-masked | 1 min | Equal-user/session empirical | 0.586 [0.215 to 0.633] | 0.906 [0.902 to 0.907] | 0.006 [0.002 to 0.007] | 24.66 [24.03 to 24.68] | 5 |
| Unseen user | History-masked | 3 min | Origin-pooled CQR | 0.062 [0.000 to 0.135] | 0.887 [0.884 to 0.893] | 0.013 [0.007 to 0.016] | 29.52 [29.35 to 30.31] | 5 |
| Unseen user | History-masked | 3 min | Equal-user/session empirical | 0.633 [0.370 to 0.812] | 0.900 [0.897 to 0.902] | 0.001 [0.000 to 0.003] | 30.75 [30.49 to 31.23] | 5 |
| Unseen user | History-masked | 5 min | Origin-pooled CQR | 0.000 [0.000 to 0.116] | 0.877 [0.873 to 0.882] | 0.023 [0.018 to 0.027] | 32.21 [31.93 to 32.81] | 5 |
| Unseen user | History-masked | 5 min | Equal-user/session empirical | 0.992 [0.565 to 1.068] | 0.893 [0.892 to 0.894] | 0.007 [0.006 to 0.008] | 34.02 [33.71 to 34.31] | 5 |

### Table S11b. Five-seed paired user-bootstrap calibration differences

Differences are equal-user/equal-session minus origin-pooled results after each user's metrics were averaged across the five fixed seeds. Positive PICP and width differences indicate larger values under equal-user/equal-session calibration; negative absolute-coverage-error differences favour that sensitivity. Confidence intervals use 10,000 paired user resamples, condition on both estimated thresholds and the declared seeds, and do not include calibration-sample or optimization-seed uncertainty.

| Evaluation regime | Inference mode | Horizon | Delta PICP [95% CI] | Delta absolute coverage error [95% CI] | Delta width [95% CI], bpm | Users | Bootstrap replicates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen cross-source | History-masked | 1 min | 0.010 [0.009, 0.010] | -0.010 [-0.010, -0.009] | 0.86 [0.86, 0.86] | 144 | 10,000 |
| Frozen cross-source | History-masked | 3 min | 0.011 [0.011, 0.012] | -0.011 [-0.012, -0.011] | 1.16 [1.16, 1.16] | 144 | 10,000 |
| Frozen cross-source | History-masked | 5 min | 0.015 [0.014, 0.016] | -0.015 [-0.016, -0.014] | 1.77 [1.77, 1.77] | 144 | 10,000 |
| Unseen user | History-informed | 1 min | 0.006 [0.006, 0.007] | -0.003 [-0.007, 0.006] | 0.41 [0.41, 0.41] | 105 | 10,000 |
| Unseen user | History-informed | 3 min | 0.010 [0.009, 0.011] | -0.007 [-0.011, 0.007] | 0.87 [0.87, 0.88] | 105 | 10,000 |
| Unseen user | History-informed | 5 min | 0.012 [0.011, 0.013] | -0.012 [-0.013, 0.000] | 1.16 [1.15, 1.17] | 105 | 10,000 |
| Unseen user | History-masked | 1 min | 0.012 [0.011, 0.013] | -0.002 [-0.013, 0.011] | 0.85 [0.85, 0.86] | 105 | 10,000 |
| Unseen user | History-masked | 3 min | 0.012 [0.011, 0.013] | -0.012 [-0.013, 0.005] | 1.16 [1.15, 1.16] | 105 | 10,000 |
| Unseen user | History-masked | 5 min | 0.016 [0.015, 0.017] | -0.016 [-0.017, 0.000] | 1.76 [1.75, 1.77] | 105 | 10,000 |

### Table S11c. Calibration-user bootstrap sensitivity at the reference seed

Using the reference seed 20260722, each replicate resamples calibration users and one session and origin within each sampled user before estimating the threshold. Entries are medians and 2.5th--97.5th percentiles over 10,000 replicates. Wide threshold intervals reflect limited calibration-user support and clustered time-series dependence; no formal finite-sample coverage guarantee is claimed.

| Evaluation regime | Mode | Horizon | Median adjustment [95% interval], bpm | Median evaluation PICP [95% interval] | Calibration users | Evaluation users |
| --- | --- | --- | --- | --- | --- | --- |
| Unseen user | History-informed | 1 min | 0.805 [0.000, 3.565] | 0.916 [0.891, 0.959] | 97 | 105 |
| Unseen user | History-informed | 3 min | 0.945 [0.000, 5.396] | 0.911 [0.885, 0.960] | 97 | 105 |
| Unseen user | History-informed | 5 min | 1.047 [0.000, 5.410] | 0.903 [0.870, 0.950] | 97 | 105 |
| Unseen user | History-masked | 1 min | 0.875 [0.000, 3.766] | 0.913 [0.883, 0.957] | 97 | 105 |
| Frozen cross-source | History-masked | 1 min | 0.875 [0.000, 3.766] | 0.897 [0.873, 0.937] | 97 | 144 |
| Unseen user | History-masked | 3 min | 1.348 [0.000, 6.065] | 0.913 [0.881, 0.960] | 97 | 105 |
| Frozen cross-source | History-masked | 3 min | 1.348 [0.000, 6.065] | 0.882 [0.854, 0.937] | 97 | 144 |
| Unseen user | History-masked | 5 min | 1.500 [0.000, 6.558] | 0.902 [0.865, 0.951] | 97 | 105 |
| Frozen cross-source | History-masked | 5 min | 1.500 [0.000, 6.558] | 0.875 [0.845, 0.930] | 97 | 144 |

## Table S12. Cross-source sport-composition standardization

### Table S12a. Point-error standardization

All comparisons use history-masked predictions from the same frozen reference checkpoint (seed 20260722). Differences are GoldenCheetah minus Endomondo hierarchical MAE in bpm. Sport-matched and standardized estimates are descriptive contrasts; they do not identify causal platform or device effects. The shared-family analyses are restricted to outdoor cycling, indoor/virtual cycling, and running.

| Comparison/standardization | Sport family or target mix | 1-min external minus internal MAE [95% CI] | 3-min external minus internal MAE [95% CI] | 5-min external minus internal MAE [95% CI] |
| --- | --- | --- | --- | --- |
| Reported natural mix | all internal test vs external shared three | 1.586 [1.024, 2.133] | 2.722 [2.025, 3.383] | 2.963 [2.092, 3.761] |
| Shared-three-family natural mix | all three supported families | 1.615 [1.046, 2.186] | 2.781 [2.090, 3.480] | 3.014 [2.145, 3.830] |
| Sport matched | outdoor cycling | -0.169 [-0.806, 0.450] | 1.015 [0.144, 1.804] | 1.018 [0.009, 1.938] |
| Sport matched | indoor virtual cycling | 0.672 [-1.251, 2.379] | 1.613 [-0.748, 3.879] | -0.180 [-4.121, 3.201] |
| Sport matched | running | 0.996 [0.220, 1.862] | 1.686 [0.639, 2.817] | 2.065 [0.871, 3.337] |
| Equal-family standardized | macro average three families | 0.500 [-0.317, 1.273] | 1.438 [0.425, 2.417] | 0.968 [-0.551, 2.350] |
| Standardized to Endomondo session mix | three families weighted to internal sessions | 0.510 [-0.057, 1.110] | 1.408 [0.655, 2.177] | 1.597 [0.727, 2.481] |
| Standardized to GoldenCheetah session mix | three families weighted to external sessions | 0.031 [-0.558, 0.592] | 1.138 [0.353, 1.869] | 1.058 [0.126, 1.913] |

### Table S12b. Interval-metric standardization under matched history masking

All comparisons use the same frozen reference-seed history-capable checkpoint with prior-workout input masked in Endomondo and GoldenCheetah. I denotes Endomondo, E denotes GoldenCheetah, and delta is E minus I. PICP, 90% width, and WIS are aggregated within session and then user; WIS is better when lower. Confidence intervals independently resample users within each source 10,000 times. The estimates are descriptive and do not identify causal platform or device effects or furnish user-level conformal guarantees.

| Comparison/standardization | Sport family or target mix | Metric | 1 min: internal; external; delta [95% CI] | 3 min: internal; external; delta [95% CI] | 5 min: internal; external; delta [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Shared-three-family natural mix | all three supported families | 90% PICP | I 0.890; E 0.877; delta -0.013 [-0.022, -0.003] | I 0.889; E 0.859; delta -0.030 [-0.041, -0.019] | I 0.879; E 0.850; delta -0.029 [-0.047, -0.006] |
| Shared-three-family natural mix | all three supported families | 90% interval width | I 23.46; E 29.49; delta 6.03 [4.25, 7.75] | I 29.50; E 37.50; delta 8.00 [6.08, 9.87] | I 32.18; E 40.56; delta 8.38 [6.46, 10.26] |
| Shared-three-family natural mix | all three supported families | WIS (lower is better) | I 3.464; E 4.479; delta 1.015 [0.694, 1.326] | I 4.416; E 6.097; delta 1.681 [1.298, 2.058] | I 4.952; E 6.723; delta 1.771 [1.246, 2.251] |
| Sport matched | outdoor cycling | 90% PICP | I 0.898; E 0.883; delta -0.015 [-0.031, 0.003] | I 0.901; E 0.862; delta -0.039 [-0.054, -0.022] | I 0.884; E 0.852; delta -0.032 [-0.056, -0.001] |
| Sport matched | outdoor cycling | 90% interval width | I 32.17; E 30.86; delta -1.31 [-2.89, 0.28] | I 39.19; E 39.09; delta -0.10 [-1.68, 1.45] | I 42.17; E 42.22; delta 0.05 [-1.51, 1.57] |
| Sport matched | outdoor cycling | WIS (lower is better) | I 4.545; E 4.609; delta 0.064 [-0.284, 0.399] | I 5.533; E 6.269; delta 0.735 [0.255, 1.181] | I 6.200; E 6.909; delta 0.709 [0.102, 1.241] |
| Sport matched | indoor virtual cycling | 90% PICP | I 0.879; E 0.810; delta -0.069 [-0.138, 0.004] | I 0.852; E 0.796; delta -0.056 [-0.148, 0.073] | I 0.782; E 0.766; delta -0.016 [-0.135, 0.128] |
| Sport matched | indoor virtual cycling | 90% interval width | I 23.96; E 20.31; delta -3.66 [-8.83, 1.06] | I 32.45; E 28.59; delta -3.86 [-9.27, 1.08] | I 35.97; E 32.00; delta -3.97 [-9.40, 0.94] |
| Sport matched | indoor virtual cycling | WIS (lower is better) | I 3.563; E 4.108; delta 0.546 [-0.717, 1.698] | I 5.025; E 6.071; delta 1.045 [-0.350, 2.463] | I 6.872; E 7.386; delta 0.514 [-2.124, 2.797] |
| Sport matched | running | 90% PICP | I 0.888; E 0.856; delta -0.032 [-0.081, 0.005] | I 0.890; E 0.860; delta -0.030 [-0.064, -0.000] | I 0.890; E 0.854; delta -0.037 [-0.072, -0.005] |
| Sport matched | running | 90% interval width | I 18.60; E 21.10; delta 2.50 [0.34, 4.72] | I 24.09; E 27.75; delta 3.66 [1.17, 6.21] | I 26.66; E 30.52; delta 3.86 [1.37, 6.42] |
| Sport matched | running | WIS (lower is better) | I 2.840; E 3.665; delta 0.825 [0.269, 1.453] | I 3.699; E 4.870; delta 1.171 [0.476, 1.942] | I 4.084; E 5.506; delta 1.421 [0.623, 2.308] |
| Equal-family standardized | macro average three families | 90% PICP | I 0.888; E 0.850; delta -0.039 [-0.070, -0.009] | I 0.881; E 0.839; delta -0.041 [-0.076, 0.003] | I 0.852; E 0.824; delta -0.028 [-0.072, 0.022] |
| Equal-family standardized | macro average three families | 90% interval width | I 24.91; E 24.09; delta -0.82 [-2.97, 1.15] | I 31.91; E 31.81; delta -0.10 [-2.37, 2.04] | I 34.93; E 34.91; delta -0.02 [-2.25, 2.09] |
| Equal-family standardized | macro average three families | WIS (lower is better) | I 3.649; E 4.128; delta 0.479 [-0.043, 1.009] | I 4.752; E 5.736; delta 0.984 [0.382, 1.601] | I 5.719; E 6.600; delta 0.882 [-0.112, 1.800] |
| Standardized to Endomondo session mix | three families weighted to endomondo sessions | 90% PICP | I 0.892; E 0.867; delta -0.025 [-0.055, -0.002] | I 0.894; E 0.860; delta -0.034 [-0.055, -0.015] | I 0.886; E 0.852; delta -0.034 [-0.058, -0.012] |
| Standardized to Endomondo session mix | three families weighted to endomondo sessions | 90% interval width | I 24.29; E 25.12; delta 0.83 [-0.60, 2.33] | I 30.46; E 32.45; delta 1.98 [0.39, 3.65] | I 33.21; E 35.37; delta 2.16 [0.56, 3.84] |
| Standardized to Endomondo session mix | three families weighted to endomondo sessions | WIS (lower is better) | I 3.555; E 4.062; delta 0.507 [0.131, 0.915] | I 4.477; E 5.466; delta 0.989 [0.519, 1.495] | I 5.002; E 6.115; delta 1.113 [0.558, 1.698] |
| Standardized to GoldenCheetah session mix | three families weighted to goldencheetah sessions | 90% PICP | I 0.895; E 0.875; delta -0.021 [-0.037, -0.004] | I 0.896; E 0.857; delta -0.039 [-0.055, -0.022] | I 0.878; E 0.846; delta -0.031 [-0.055, -0.003] |
| Standardized to GoldenCheetah session mix | three families weighted to goldencheetah sessions | 90% interval width | I 29.96; E 28.94; delta -1.02 [-2.46, 0.39] | I 36.90; E 36.98; delta 0.08 [-1.38, 1.51] | I 39.87; E 40.09; delta 0.22 [-1.23, 1.64] |
| Standardized to GoldenCheetah session mix | three families weighted to goldencheetah sessions | WIS (lower is better) | I 4.270; E 4.460; delta 0.190 [-0.132, 0.509] | I 5.277; E 6.087; delta 0.810 [0.377, 1.227] | I 5.994; E 6.775; delta 0.781 [0.222, 1.299] |

## Table S13. Sport-specific frozen cross-source interval performance

All rows use the frozen reference seed 20260722 and apply Endomondo-derived CQR thresholds unchanged. Brackets are 95% intervals from 10,000 user bootstrap replicates and quantify user-sampling variation conditional on that fitted checkpoint.

| External sport family | Horizon | 90% PICP [95% CI] | 90% width [95% CI], bpm | WIS [95% CI] | Users |
| --- | --- | --- | --- | --- | --- |
| outdoor cycling | 1 min | 0.883 [0.877, 0.889] | 30.86 [30.06, 31.70] | 4.61 [4.43, 4.79] | 143 |
| outdoor cycling | 3 min | 0.862 [0.854, 0.870] | 39.09 [38.29, 39.89] | 6.27 [6.04, 6.50] | 143 |
| outdoor cycling | 5 min | 0.852 [0.843, 0.861] | 42.22 [41.46, 42.98] | 6.91 [6.66, 7.17] | 143 |
| indoor virtual cycling | 1 min | 0.810 [0.771, 0.845] | 20.31 [19.37, 21.29] | 4.11 [3.49, 4.86] | 41 |
| indoor virtual cycling | 3 min | 0.796 [0.761, 0.831] | 28.59 [27.50, 29.67] | 6.07 [5.24, 6.97] | 41 |
| indoor virtual cycling | 5 min | 0.766 [0.713, 0.810] | 32.00 [30.88, 33.15] | 7.39 [6.25, 8.69] | 41 |
| running | 1 min | 0.856 [0.809, 0.890] | 21.10 [19.59, 22.81] | 3.66 [3.19, 4.24] | 50 |
| running | 3 min | 0.860 [0.827, 0.886] | 27.75 [25.99, 29.74] | 4.87 [4.24, 5.58] | 50 |
| running | 5 min | 0.854 [0.821, 0.882] | 30.52 [28.77, 32.53] | 5.51 [4.79, 6.32] | 50 |

## Table S14. Descriptive source-shift characterization

Internal and cross-source inputs are the unseen-user history-masked Endomondo test set and history-masked GoldenCheetah set, respectively, from reference seed 20260722. Metrics are averaged within session and then equally across users; confidence intervals resample users independently within each source. These contrasts jointly reflect users, sports, devices, sampling, session structure, and platform processing and are not causal source or device effects.

| Category | Metric | Internal | External | External minus internal [95% CI] | Unit |
| --- | --- | --- | --- | --- | --- |
| heart rate | context hr bpm | 145.440 | 136.623 | -8.816 [-11.994, -5.575] | bpm |
| heart rate | target hr 60 bpm | 146.558 | 138.273 | -8.285 [-11.416, -5.203] | bpm |
| heart rate | target hr 180 bpm | 146.816 | 138.746 | -8.069 [-11.390, -5.093] | bpm |
| heart rate | target hr 300 bpm | 146.997 | 138.847 | -8.150 [-11.165, -4.966] | bpm |
| context missingness | context hr missing percent | 6.721 | 0.594 | -6.128 [-6.763, -5.486] | percent |
| context missingness | context speed missing percent | 6.699 | 0.603 | -6.095 [-6.727, -5.441] | percent |
| context missingness | context altitude missing percent | 6.694 | 8.363 | 1.669 [-0.901, 4.488] | percent |
| session support | session duration minutes | 70.308 | 114.808 | 44.500 [34.078, 54.298] | minutes |
| raw support | raw valid hr coverage percent | 99.906 | 99.690 | -0.216 [-0.363, -0.037] | percent |
| raw support | raw median sampling gap seconds | 7.218 | 1.038 | -6.180 [-6.480, -5.871] | seconds |
| raw support | raw max sampling gap seconds | 519.620 | 363.710 | -155.910 [-616.286, 149.249] | seconds |
| causal grid support | causal grid hr observed percent | 86.453 | 94.632 | 8.179 [6.133, 10.349] | percent |
| causal grid support | causal grid speed observed percent | 86.296 | 94.524 | 8.227 [6.165, 10.326] | percent |
| causal grid support | causal grid altitude observed percent | 86.506 | 87.099 | 0.593 [-2.527, 3.787] | percent |
| evaluation support | evaluation origins per session | 6.669 | 17.469 | 10.801 [9.758, 11.865] | origins/session |
| deployment | model user history available | 0.000 | 0.000 | 0.000 [not estimable: both sources constant] | percent |

## Table S15. Cross-source normalized-signal duplicate audit

The exact stage joins cryptographic fingerprints of complete processed 10-s HR, speed, and altitude values and masks after removing identifiers, source labels, absolute timestamps, and sport labels. The approximate stage combines deterministic quantized profiles with random-hyperplane locality-sensitive hashing and continuous verification. A zero verified count reduces contamination concern but cannot prove absence of every duplicate affected by cropping, drift, long gaps, smoothing, or transformations outside the declared search.

| Audit stage | Cross-source pairs | Result |
| --- | --- | --- |
| Processed modelling sessions | Not applicable | 165,660 Endomondo; 32,444 GoldenCheetah |
| Exact full normalized-signal fingerprint | 0 | No confirmed cross-source exact match |
| Exact HR-only fingerprint screen | 0 | No HR-only exact candidate |
| Quantized deterministic profile screen | 0 | No signature candidate |
| Random-hyperplane LSH after duration filter | 962,017 | 70 passed the HR prefilter |
| Continuous verification | 70 | Zero verified HR or HR-plus-auxiliary near-duplicate pairs |

## Table S16. Independent symmetric split-conformal persistence baseline

Persistence used the most recent observed context HR as its deterministic point forecast. For each horizon and nominal level, the absolute-residual radius was the finite-sample higher order statistic from the dedicated strict-temporal or unseen-user calibration partition. GoldenCheetah reused the unseen-user Endomondo radii unchanged. Bounds were clipped to 30--240 bpm, and all metrics were averaged within session and then user. This independently calibrated baseline has no learned parameters; it is not a second calibration of the quantile TCN. Because calibration pooled correlated origins, the intervals do not furnish a finite-sample guarantee for session-then-user PICP.

| Regime | Horizon | MAE, bpm | Calibration radius 50/80/90%, bpm | PICP 50/80/90% | Width 50/80/90%, bpm | WIS | Users / sessions / origins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Strict temporal | 1 min | 6.726 | 3.89 / 10.33 / 16.75 | 0.499 / 0.799 / 0.899 | 7.78 / 20.66 / 33.48 | 4.542 | 948 / 16,012 / 104,144 |
| Strict temporal | 3 min | 8.757 | 5.00 / 13.62 / 21.30 | 0.507 / 0.798 / 0.894 | 10.00 / 27.24 / 42.57 | 5.847 | 948 / 16,012 / 104,144 |
| Strict temporal | 5 min | 9.587 | 5.62 / 14.75 / 22.75 | 0.486 / 0.793 / 0.892 | 11.23 / 29.49 / 45.47 | 6.323 | 948 / 16,012 / 104,144 |
| Unseen user | 1 min | 6.559 | 3.67 / 9.43 / 15.00 | 0.489 / 0.783 / 0.883 | 7.33 / 18.84 / 29.96 | 4.433 | 105 / 15,026 / 101,184 |
| Unseen user | 3 min | 8.627 | 5.00 / 12.50 / 19.75 | 0.517 / 0.780 / 0.880 | 10.00 / 24.97 / 39.43 | 5.772 | 105 / 15,026 / 101,184 |
| Unseen user | 5 min | 9.599 | 5.20 / 13.50 / 21.00 | 0.481 / 0.772 / 0.879 | 10.40 / 26.97 / 41.93 | 6.387 | 105 / 15,026 / 101,184 |
| Frozen cross-source | 1 min | 7.929 | 3.67 / 9.43 / 15.00 | 0.411 / 0.714 / 0.847 | 7.33 / 18.86 / 30.00 | 5.319 | 144 / 31,851 / 531,725 |
| Frozen cross-source | 3 min | 11.282 | 5.00 / 12.50 / 19.75 | 0.415 / 0.683 / 0.815 | 10.00 / 25.00 / 39.49 | 7.508 | 144 / 31,851 / 531,725 |
| Frozen cross-source | 5 min | 12.569 | 5.20 / 13.50 / 21.00 | 0.374 / 0.665 / 0.811 | 10.40 / 27.00 / 41.99 | 8.323 | 144 / 31,851 / 531,725 |

## Table S17. Post hoc matched-origin sport-availability sensitivity

For each held family and each of the three shared seeds, history-masked predictions from the full-sport unseen-user model and held-family model were aligned by the same global origin index within the joint unseen-user/sport test. Absolute errors were averaged within session and user, then each user's effect was averaged across seeds before 10,000 user-bootstrap resamples. Positive differences indicate higher error when the sport family was unavailable during fitting and represented by code 0. This is an operational sport-availability contrast, not a causal sport effect: it also captures the held-family model's locked token exposure, sport-excluded fitting data, and training budget. Rows with fewer than 25 users remain cautionary.

| Held sport family | Horizon | Full-sport / held-sport MAE, bpm | Held minus full MAE [95% CI], bpm | Users with higher held error | Users / sessions / origins |
| --- | --- | --- | --- | --- | --- |
| Outdoor cycling | 1 min | 7.853 / 8.125 | 0.271 [0.104, 0.416] | 83.1% | 77 / 6,097 / 37,856 |
| Outdoor cycling | 3 min | 9.549 / 9.913 | 0.365 [0.181, 0.548] | 79.2% | 77 / 6,097 / 37,856 |
| Outdoor cycling | 5 min | 10.572 / 10.922 | 0.350 [0.162, 0.524] | 81.8% | 77 / 6,097 / 37,856 |
| Indoor/virtual cycling | 1 min | 5.622 / 5.892 | 0.271 [-0.101, 0.592] | 72.2% | 18 / 234 / 1,523 |
| Indoor/virtual cycling | 3 min | 8.005 / 8.049 | 0.044 [-0.512, 0.513] | 50.0% | 18 / 234 / 1,523 |
| Indoor/virtual cycling | 5 min | 11.794 / 11.502 | -0.292 [-0.946, 0.259] | 61.1% | 18 / 234 / 1,523 |
| Running | 1 min | 4.732 / 5.085 | 0.353 [0.222, 0.559] | 89.8% | 88 / 8,431 / 60,542 |
| Running | 3 min | 6.019 / 6.498 | 0.479 [0.276, 0.794] | 88.6% | 88 / 8,431 / 60,542 |
| Running | 5 min | 6.650 / 7.240 | 0.590 [0.419, 0.838] | 88.6% | 88 / 8,431 / 60,542 |
| Walking/hiking | 1 min | 6.481 / 6.511 | 0.030 [-0.320, 0.365] | 57.9% | 19 / 96 / 428 |
| Walking/hiking | 3 min | 7.022 / 7.477 | 0.455 [0.043, 0.857] | 73.7% | 19 / 96 / 428 |
| Walking/hiking | 5 min | 8.012 / 9.151 | 1.139 [0.479, 1.783] | 73.7% | 19 / 96 / 428 |
| Strength/cross-training | 1 min | 9.482 / 9.660 | 0.178 [-0.159, 0.537] | 70.0% | 20 / 112 / 503 |
| Strength/cross-training | 3 min | 12.189 / 12.522 | 0.333 [-0.116, 0.834] | 60.0% | 20 / 112 / 503 |
| Strength/cross-training | 5 min | 11.683 / 12.268 | 0.585 [0.103, 1.152] | 60.0% | 20 / 112 / 503 |

## Table S18. Deliberately contaminated same-session-window negative control

This retrospective negative control kept the 104,144 strict-temporal test origins unchanged but deliberately assigned 290,245, 62,146, and 62,463 other 60-s origins from those test sessions to fitting, validation, and calibration by a model-seed-independent SHA-256 rule. Exact test rows remained disjoint. Nevertheless, 15,839 of 16,012 test sessions entered fitting, 98.9% of test origins had a contaminated fitting origin within 300 s, and 95.0% shared at least one target timestamp with a contaminated fitting origin. Three freshly initialized zero-history-trained models used the formal budget and were compared with the corresponding clean v0.23 predictions on the identical row order. Per-user differences were averaged over the three matched seeds before 10,000 user-bootstrap resamples. This deliberately invalid pipeline is not eligible for a model leaderboard, and its observed effect is specific to this contamination design rather than a general estimate of leakage bias.

### Table S18a. Point-error contrast

Negative contaminated-minus-clean MAE denotes apparent optimism. Relative optimism is 100 * (clean - contaminated) / clean; negative values denote deterioration. Seed ranges are descriptive, whereas the difference confidence interval is a paired user bootstrap conditional on the three seeds.

| Horizon | Clean MAE median [seed range], bpm | Contaminated MAE median [seed range], bpm | Contaminated minus clean MAE [95% CI], bpm | Apparent MAE optimism median [seed range], % | Users / matched seeds |
| --- | --- | --- | --- | --- | --- |
| 1 min | 6.067 [6.067 to 6.078] | 6.059 [6.048 to 6.071] | -0.011 [-0.025, 0.002] | 0.143 [0.109 to 0.306] | 948 / 3 |
| 3 min | 7.632 [7.612 to 7.632] | 7.610 [7.603 to 7.619] | -0.014 [-0.027, -0.001] | 0.171 [0.022 to 0.373] | 948 / 3 |
| 5 min | 8.332 [8.321 to 8.335] | 8.329 [8.324 to 8.332] | -0.001 [-0.016, 0.014] | 0.070 [-0.132 to 0.095] | 948 / 3 |

### Table S18b. Empirical 90% interval contrast

Intervals use each pipeline's own invalidly contaminated or clean calibration partition as applicable. The CQR guarantee is not valid for the contaminated design. Differences are contaminated minus clean after identical session--user aggregation and three-seed user pairing.

| Horizon | Clean / contaminated 90% PICP median | PICP difference [95% CI] | Clean / contaminated width median, bpm | Width difference [95% CI], bpm |
| --- | --- | --- | --- | --- |
| 1 min | 0.895 / 0.897 | 0.001 [-0.001, 0.002] | 24.79 / 24.66 | -0.085 [-0.112, -0.057] |
| 3 min | 0.893 / 0.893 | 0.000 [-0.001, 0.002] | 30.92 / 30.94 | 0.040 [0.006, 0.073] |
| 5 min | 0.891 / 0.891 | 0.000 [-0.001, 0.002] | 33.60 / 33.74 | 0.084 [0.048, 0.120] |

## Table S19. Horizon-specific target-availability sensitivity

The primary complete-three-target rule conditions every horizon on availability of all 1-, 3-, and 5-min targets. Both post hoc diagnostics held the original users and evaluation sessions fixed, rebuilt the same 300-s reporting grid from the raw heart-rate streams, and required only the target for the horizon being evaluated. Positive expanded-minus-common differences indicate higher error on the enlarged cohort.

### Table S19a. Parameter-free persistence diagnostic

Persistence used the most recent observed 10-s context-bin HR. Errors were averaged within session and then user; confidence intervals used 10,000 paired user resamples. All common-cohort user, session, and origin counts were reproduced exactly, and persistence MAE differed from the authoritative artifacts by less than 0.000001 bpm.

| Regime | Horizon | Common-cohort origins | Horizon-specific origins | Added origins (%) | Common-cohort MAE, bpm | Horizon-specific MAE, bpm | MAE difference [95% CI], bpm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Strict temporal | 1 min | 104,144 | 118,631 | 14,487 (13.9%) | 6.726 | 6.701 | -0.024 [-0.067, 0.019] |
| Strict temporal | 3 min | 104,144 | 112,710 | 8,566 (8.2%) | 8.757 | 8.877 | 0.120 [0.069, 0.175] |
| Strict temporal | 5 min | 104,144 | 107,326 | 3,182 (3.1%) | 9.587 | 9.699 | 0.112 [0.058, 0.173] |
| Unseen user | 1 min | 101,184 | 114,803 | 13,619 (13.5%) | 6.559 | 6.507 | -0.052 [-0.152, 0.044] |
| Unseen user | 3 min | 101,184 | 109,252 | 8,068 (8.0%) | 8.627 | 8.724 | 0.097 [0.028, 0.159] |
| Unseen user | 5 min | 101,184 | 104,019 | 2,835 (2.8%) | 9.599 | 9.685 | 0.085 [0.052, 0.125] |
| Frozen cross-source | 1 min | 531,725 | 576,694 | 44,969 (8.5%) | 7.929 | 7.990 | 0.060 [0.021, 0.097] |
| Frozen cross-source | 3 min | 531,725 | 560,135 | 28,410 (5.3%) | 11.282 | 11.380 | 0.098 [0.064, 0.134] |
| Frozen cross-source | 5 min | 531,725 | 546,076 | 14,351 (2.7%) | 12.569 | 12.660 | 0.091 [0.063, 0.122] |

### Table S19b. Five-seed frozen-main-model sensitivity

The original saved predictions were retained for common-cohort rows, and only the additional horizon-eligible rows were newly inferred. The strict-temporal and unseen-user tests used their frozen history-informed checkpoints; GoldenCheetah used frozen history-masked inference. No model was retrained or selected, normalizers were not refitted, no interval calibration was performed, and GoldenCheetah supplied no adaptation signal. Seed ranges are descriptive. The paired-user effect first averages each user's expanded-minus-common difference across the five matched seeds and then uses 10,000 user-bootstrap resamples; seeds are not resampled.

| Regime and inference mode | Horizon | Common MAE median [seed range], bpm | Expanded MAE median [seed range], bpm | Expanded - common median [seed range], bpm | Paired-user difference [95% CI], bpm | Added origins |
| --- | --- | --- | --- | --- | --- | --- |
| Strict temporal (history-informed) | 1 min | 6.012 [6.004 to 6.057] | 6.015 [6.003 to 6.053] | 0.002 [-0.005 to 0.004] | 0.001 [-0.037, 0.036] | 14,487 |
| Strict temporal (history-informed) | 3 min | 7.557 [7.548 to 7.563] | 7.701 [7.692 to 7.706] | 0.144 [0.138 to 0.144] | 0.143 [0.102, 0.185] | 8,566 |
| Strict temporal (history-informed) | 5 min | 8.245 [8.239 to 8.267] | 8.355 [8.347 to 8.379] | 0.110 [0.102 to 0.114] | 0.109 [0.058, 0.170] | 3,182 |
| Unseen user (history-informed) | 1 min | 5.849 [5.821 to 5.867] | 5.836 [5.808 to 5.853] | -0.013 [-0.021 to 0.004] | -0.010 [-0.113, 0.080] | 13,619 |
| Unseen user (history-informed) | 3 min | 7.407 [7.394 to 7.420] | 7.552 [7.536 to 7.567] | 0.144 [0.142 to 0.152] | 0.145 [0.089, 0.207] | 8,068 |
| Unseen user (history-informed) | 5 min | 8.157 [8.127 to 8.212] | 8.238 [8.205 to 8.284] | 0.078 [0.072 to 0.081] | 0.077 [0.037, 0.130] | 2,835 |
| Frozen cross-source (history-masked) | 1 min | 7.465 [7.427 to 7.471] | 7.562 [7.522 to 7.574] | 0.098 [0.095 to 0.104] | 0.099 [0.069, 0.130] | 44,969 |
| Frozen cross-source (history-masked) | 3 min | 10.206 [10.192 to 10.231] | 10.366 [10.344 to 10.382] | 0.155 [0.146 to 0.161] | 0.154 [0.123, 0.186] | 28,410 |
| Frozen cross-source (history-masked) | 5 min | 11.214 [11.186 to 11.226] | 11.287 [11.259 to 11.297] | 0.073 [0.068 to 0.079] | 0.074 [0.052, 0.096] | 14,351 |

## Supplementary figure caption

**Supplementary Fig. 1. Ablation, stride sensitivity, and subgroup boundaries.** (a) Reference-seed paired multimodal-minus-HR-only MAE differences. (b) Reference-seed change in MAE when the frozen unseen-user model is evaluated every 60 s rather than every 300 s. (c) Five-seed history-informed-minus-zero-history-trained effects, with paired per-user differences averaged over seeds before user bootstrap. (d) Reference-seed recorded-female-minus-recorded-male descriptive MAE differences; the unseen-user recorded-female subgroup contains only 10 users.

## Supplementary provenance

- Forecast-origin flow: `outputs/audit/forecast_origins_full_v0_3_1.json`.
- Split support: `outputs/audit/split_manifest_v0_2_0_summary.json`.
- Multiseed main/comparator and interval summaries: `outputs/q1_multiseed_v0_21_0/aggregation/seed_variability_summary_v0_22_0.csv`, `outputs/q1_multiseed_v0_21_0/aggregation/main_history_difference_summary_v0_22_0.csv`, and `outputs/q1_multiseed_v0_21_0/aggregation/main_vs_comparator_summary_v0_22_0.csv`.
- Zero-history-trained strategy summaries: `outputs/independent_zero_history_v0_23_0/aggregation/strategy_contrast_seed_summary_v0_23_0.csv`, `strategy_contrast_user_bootstrap_v0_23_0.csv`, and `strategy_contrasts_per_seed_v0_23_0.csv`.
- Strict temporal metrics: `outputs/results/temporal_uncertainty_point_v0_13_0.csv` and `temporal_aligned_baselines_v0_13_0.csv`.
- User/cross-source metrics: `uncertainty_point_metrics_v0_11_0.csv`, neural/XGBoost comparator files, and `naive_baseline_metrics_v0_5_0.csv`.
- Interval metrics: `temporal_uncertainty_interval_v0_13_0.csv`, `uncertainty_interval_metrics_v0_11_0.csv`, and `figure3_uncertainty_bootstrap_v0_18_0.csv`.
- Sport shift: `sport_shift_point_v0_12_0.csv`, `sport_shift_mae_bootstrap_v0_19_0.csv`, and `sport_shift_uncertainty_bootstrap_v0_17_0.csv`.
- Three-seed paired-user comparator and sport effects: `outputs/results/multiseed_paired_model_comparisons_v0_25_0.csv`, `multiseed_paired_sport_shift_v0_25_0.csv`, and `outputs/audit/multiseed_paired_user_bootstrap_v0_25_0.audit.json`.
- Reference-seed paired effects and sensitivities: `temporal_paired_comparisons_v0_13_0.csv`, `paired_model_comparisons_v0_11_0.csv`, and `signal_ablation_paired_v0_14_0.csv`.
- Stride sensitivity and recorded-gender contrasts: version 0.15.0 and 0.16.0 result artifacts.
- Prior-work coding: `references/PRIOR_WORK_COMPARISON.md` and the documented targeted update `references/TARGETED_LITERATURE_UPDATE_2026-07-23.md`; this is not a systematic review.
- Completed-history availability: `outputs/results/history_availability_v0_19_0.csv`.
- Calibration sensitivity: `outputs/results/multiseed_balanced_calibration_summary_v0_24_0.csv`, `multiseed_balanced_calibration_differences_v0_24_0.csv`, `outputs/audit/multiseed_balanced_calibration_v0_24_0.json`, and `clustered_calibration_bootstrap_v0_20_0.csv`.
- Cross-source sport composition and interval heterogeneity: `external_sport_standardization_v0_20_1.csv`, `external_sport_uncertainty_standardization_v0_24_0.csv`, `outputs/audit/external_sport_uncertainty_standardization_v0_24_0.json`, and `external_sport_uncertainty_bootstrap_v0_21_0.csv`.
- Source-shift characterization: `source_shift_characterization_v0_21_0.csv`, `source_shift_sport_composition_v0_21_0.csv`, and `source_shift_session_distributions_v0_21_0.csv`.
- Cross-source duplicate audit: `outputs/audit/cross_source_signal_duplicate_audit_v0_20_0.json` and its two result CSV files.
- Independent probabilistic baseline: `outputs/results/persistence_conformal_baseline_v0_26_0.csv` and `outputs/audit/persistence_conformal_baseline_v0_26_0.json`.
- Matched-origin sport availability: `outputs/results/matched_sport_availability_v0_27_0.csv` and `outputs/audit/matched_sport_availability_v0_27_0.json`.
- Deliberately contaminated same-session-window negative control: `outputs/deliberately_leaky_negative_control_v0_28_0/aggregation/paired_metrics_seed_summary_v0_28_0.csv`, `paired_user_bootstrap_v0_28_0.csv`, `interval_diagnostics_per_seed_v0_28_0.csv`, and `audit.json`; the per-user bootstrap input remains private.
- Horizon-specific target eligibility: `outputs/results/horizon_specific_eligibility_v0_29_0.csv` and `outputs/audit/horizon_specific_eligibility_v0_29_0.json`.
- Frozen-model horizon-specific target eligibility: `outputs/results/horizon_specific_frozen_model_per_seed_v0_30_0.csv`, `horizon_specific_frozen_model_summary_v0_30_0.csv`, and `outputs/audit/horizon_specific_frozen_models_v0_30_0.json`.
