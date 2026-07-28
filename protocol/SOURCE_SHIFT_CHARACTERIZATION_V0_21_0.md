# Frozen internal--external source-shift characterization (v0.21.0)

## Estimands and support

This audit compares Endomondo unseen-user **history-masked inference** with the frozen GoldenCheetah **history-masked cross-source evaluation** from the same history-capable checkpoint. It characterizes the evaluated data distributions; it does not attribute differences causally to platform, device, sport, or population.

| Evaluation | Users | Sessions | 300-s origins |
|---|---:|---:|---:|
| Endomondo unseen-user history-masked inference | 105 | 15,026 | 101,184 |
| GoldenCheetah frozen cross-source history-masked inference | 144 | 31,851 | 531,725 |

Origin-based numeric comparisons average origins within sessions, sessions within users, and then weight users equally. Session-based metrics average selected sessions within users and then weight users equally. Confidence intervals are percentile intervals from 2,000 independent user-cluster bootstrap replicates within each source. The separate session-distribution file is descriptive and deliberately makes no session-level independence claim.

## Main shifts

- Mean selected-session duration was 70.3 min internally and 114.8 min externally (external minus internal 44.5 min; 95% CI 34.1 to 54.3).
- The raw median positive sampling gap averaged 7.2 s internally and 1.0 s externally. These are source-format support descriptors, not model inputs.
- Mean context HR was 145.4 versus 136.6 bpm. Target HR differences (external minus internal) were -8.3 bpm at +1 min, -8.1 bpm at +3 min, and -8.1 bpm at +5 min.
- Context missingness differed most for speed (6.7% internal; 0.6% external) and altitude (6.7% internal; 8.4% external). Missingness is retained as an observed mask and was not repaired using future samples.
- The model-level completed-workout input was masked in both comparisons. This protocol setting does not assert that earlier raw workouts were absent for every user and is distinct from a zero-history-trained model.

## Sport composition

Natural source composition is reported at both the selected-session and overlapping-origin levels. Origin shares are descriptive only.

| Sport family | Internal sessions (%) | External sessions (%) | Internal origins (%) | External origins (%) |
|---|---:|---:|---:|---:|
| outdoor_cycling | 40.6 | 80.9 | 37.4 | 88.5 |
| running | 56.1 | 12.0 | 59.8 | 6.5 |
| indoor_virtual_cycling | 1.6 | 7.1 | 1.5 | 4.9 |
| walking_hiking | 0.6 | 0.0 | 0.4 | 0.0 |
| skiing | 0.4 | 0.0 | 0.3 | 0.0 |
| strength_cross_training | 0.7 | 0.0 | 0.5 | 0.0 |

The outcome-blind external sport-family eligibility rule was fixed to the three families supported by the primary cross-source scope before the corresponding model inference; the GoldenCheetah outcomes themselves were not prospectively sequestered. The internal unseen-user test also contains small walking/hiking, skiing, and strength/cross-training components. Consequently, natural-mix source differences combine source, sensor, user, session, and sport-composition shifts. The separate sport-standardization analysis should be used when asking how much broad three-family composition explains performance differences.

## Interpretation limits

- Users are the inferential resampling unit; repeated forecast origins are not treated as independent.
- Confidence intervals quantify finite-sample user heterogeneity within these two selected cohorts, not population-representative sampling uncertainty.
- Raw gap and coverage fields come from frozen quality manifests; 10-s grid support and context missingness come from frozen model arrays.
- This characterization is descriptive. It cannot isolate device effects, platform effects, physiology, training status, or demographic selection.
