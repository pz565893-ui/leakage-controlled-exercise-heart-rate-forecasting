# Data feasibility decision

**Decision date:** 2026-07-22  
**Status:** approved for study construction; final analytic counts remain pending forecast-origin construction, duplicate checks, and a locked sport ontology.

## Decision

The available Endomondo and GoldenCheetah data are sufficient to begin the proposed leakage-controlled, multi-horizon heart-rate forecasting study. Endomondo will be the development dataset and GoldenCheetah will be reserved as a frozen external-validation dataset. This decision does **not** imply that every downloaded activity is eligible or that the final model results are known.

The primary external-validation scope will initially be restricted to:

1. outdoor cycling;
2. running;
3. indoor or virtual cycling.

Other sport families may enter secondary analyses only after the full census confirms adequate users, sessions, heart-rate coverage, and target-window support. Swimming is not a primary candidate because only 21.1% of its GoldenCheetah metadata records indicate heart-rate availability in the current audit.

## Evidence used

The audit used deterministic, evenly spaced sampling rather than the beginning of either source file.

| Dataset | Audit scope | Users observed | Provisionally eligible sessions | Eligibility rate |
|---|---:|---:|---:|---:|
| Endomondo | 5,000 records | 922 | 4,767 | 95.3% |
| GoldenCheetah | 3,000 CSV files sampled from 51,470 | 150 total users | 2,232 | 74.4% |

Provisional eligibility requires enough time span and usable heart-rate coverage for a 5-minute context and forecasts up to 5 minutes. It is a feasibility measure, not the final analytic cohort.

### Endomondo findings

- Median session span: 3,715 s.
- Median observation interval: 8 s.
- Median valid heart-rate coverage: 100% among sampled records.
- Timestamp, heart rate, altitude, latitude, and longitude appeared in all 5,000 sampled records.
- A direct speed field appeared in only 1,103 records (22.1%).
- Running and cycling dominate the sampled activities, with smaller support for mountain biking, transport cycling, walking, indoor cycling, orienteering, skiing, and other activities.

### GoldenCheetah findings

- 150 user directories and 51,470 activity CSV files are present.
- 148 of 150 user metadata JSON files parse successfully; two malformed metadata files are retained and must be flagged rather than silently repaired.
- The audit observed 347 raw sport labels, so a versioned mapping table is mandatory.
- Among 3,000 deterministically sampled CSV files, the median duration was 4,262 s and 74.4% were provisionally eligible.
- The standard CSV header exposes elapsed time, distance, power, heart rate, cadence, and altitude; field presence in a header must not be confused with valid row-level measurements.
- Metadata heart-rate availability was 76.8% for outdoor cycling, 70.7% for running, and 93.0% for indoor or virtual cycling.
- Metadata indicate 144 men and 4 women among the 148 valid user records. External sex-stratified results therefore cannot support balanced inferential claims.
- Strict within-user timestamp linkage uniquely connects 50,002 of 51,470 CSV files (97.15%) to ride metadata and its sport label. The 1,468 non-unique or unavailable links receive explicit status codes and are not force-matched.

## Protocol consequences

### Full session-level census

After the bounded audit, all 253,020 Endomondo records and all 51,470 GoldenCheetah CSV files were scanned at session level. Screening required a 10-minute to 24-hour duration, at least 80% valid heart-rate observations, monotonic timestamps, and an adjudicated sport family for model eligibility. This produced:

| Dataset | Total sessions | Provisionally model-eligible | Rate |
|---|---:|---:|---:|
| Endomondo | 253,020 | 207,943 | 82.2% |
| GoldenCheetah | 51,470 | 32,648 | 63.4% |

These remain session-level counts. Final analytic counts will be lower after local context-gap checks, causal resampling, target alignment, duplicate detection, and forecast-origin construction.

Before duplicate control, the GoldenCheetah primary external candidates contain 26,024 outdoor-cycling sessions, 3,836 running sessions, and 2,279 indoor/virtual-cycling sessions. Together they contribute 32,139 provisionally eligible sessions; the locked split manifest reports the post-duplicate total.

### Temporal representation

The primary grid remains 10 s. The observed Endomondo median interval is 8 s, so 10 s is a conservative shared grid that limits interpolation and keeps 30 context steps for a 5-minute history. Sensitivity analyses at 5 s and 30 s remain prespecified.

### Shared feature set

Direct speed is not a universally available core feature. The shared causal input will use:

- heart rate;
- elapsed time;
- altitude and causal altitude change;
- distance and/or speed derived only from measurements at or before the forecast origin;
- canonical sport family;
- missingness and interpolation indicators.

Power and cadence remain restricted to a secondary sensor-rich analysis because they are not consistently shared across datasets.

### External validation

GoldenCheetah must remain untouched during Endomondo model selection. The Endomondo-trained preprocessing, model weights, sport mapping, and primary calibration rule will be frozen before primary external evaluation. Any GoldenCheetah recalibration or adaptation will be labeled secondary and will use user-disjoint calibration and test subsets.

### Data quality gates before modelling

The final full-data manifest must report, for every dataset and sport family:

- unique users and sessions;
- session duration and sampling-gap distributions;
- valid heart-rate coverage and physiologic-range exclusions;
- availability of every candidate feature;
- eligible forecast origins at 1, 3, and 5 minutes;
- duplicate or near-duplicate sessions;
- user-by-sport support;
- excluded records with mutually exclusive reason codes.

No training windows may be generated before users and sessions have been assigned to their locked split.

## Limitations of this decision

This is a bounded feasibility audit, not a complete cohort census. The reported percentages may change after full parsing, row-level validation, duplicate detection, sport-label adjudication, and forecast-origin construction. GoldenCheetah's severe sex imbalance and device heterogeneity constrain external subgroup and sensor-generalization claims.

## Reproducible artifacts

- `notebooks/01_data_feasibility_audit.ipynb`
- `outputs/audit/bounded_feasibility_audit.json`
- `src/data_audit.py`
- `outputs/manifests/goldencheetah_session_linkage_v0_2_0.csv`
- `configs/sport_ontology_v0_2_0.csv`
- `outputs/manifests/endomondo_session_quality_v0_2_0.csv`
- `outputs/manifests/goldencheetah_session_quality_v0_2_0.csv`
