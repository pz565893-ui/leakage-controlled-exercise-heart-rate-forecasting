# Forecast-origin specification and full-build audit

**Window-index version:** 0.3.1  
**Built from:** ontology/split version 0.2.0  
**Status:** complete and locked for primary 10 s-grid experiments.

## Causal task definition

Each forecast origin uses only measurements at or before the origin. The context covers the preceding 300 s and is represented by thirty right-closed 10 s bins, `(t-300, t-290], ..., (t-10, t]`. Candidate origins occur every 60 s. A lower-overlap subset aligned every 300 s is marked for primary validation, calibration, and test reporting.

The direct multi-horizon targets are heart rate at +60, +180, and +300 s. An exact valid measurement is used when available. Otherwise, the target is linearly interpolated between two valid future heart-rate samples only when their separation is at most 30 s. Every sample supporting a target must be strictly later than the forecast origin.

## Origin eligibility

An origin is retained only when:

- at least 24 of 30 context bins contain a valid 30–240 bpm heart-rate value;
- no boundary or internal valid-heart-rate gap in the context exceeds 60 s;
- all three future targets are available under the 30 s interpolation-span rule;
- the context remains within one preassigned session;
- the session passed ontology, signal-quality, duplicate, and split controls.

No full feature tensor is copied into the index. The database stores source session keys, origin times, target values, construction diagnostics, and inherited partitions. Model datasets will reconstruct causal features from the raw session and this immutable index.

## Full-build counts

| Dataset | Sessions processed | Candidate origins | Accepted origins | Acceptance | Primary evaluation origins |
|---|---:|---:|---:|---:|---:|
| Endomondo | 201,823 | 17,192,690 | 5,008,341 | 29.1% | 1,001,128 |
| GoldenCheetah | 32,587 | 3,213,126 | 2,626,835 | 81.8% | 537,672 |
| **Total** | **234,410** | **20,405,816** | **7,635,176** | **37.4%** | **1,538,800** |

Accepted Endomondo origins occur in 164,589 sessions from 1,085 users. Accepted GoldenCheetah origins occur in 32,443 sessions from 144 users. Endomondo's lower acceptance reflects irregular observations and the requirement that all three horizons be locally supported; it confirms that session-level eligibility cannot be reported as the analytic window count.

## Sport-family origin support

Primary internal sport-shift families retain the following Endomondo origin counts:

- running: 2,587,826;
- outdoor cycling: 2,231,354;
- indoor/virtual cycling: 82,371;
- walking/hiking: 61,723;
- strength/cross-training: 20,719.

The frozen GoldenCheetah primary external families retain:

- outdoor cycling: 2,306,980 origins from 143 users;
- running: 165,353 origins from 50 users;
- indoor/virtual cycling: 125,791 origins from 41 users.

Primary reported metrics use the 300 s evaluation subset and user aggregation, so these highly overlapping training-origin counts are not treated as independent statistical observations.

## Automated assertions

The full database passed all of the following with zero failures:

- session processing errors;
- input timestamps after the origin;
- target-support timestamps at or before the origin;
- context coverage below 80%;
- context gaps above 60 s;
- target heart rate outside 30–240 bpm;
- target interpolation spans above 30 s;
- primary-evaluation stride violations;
- duplicate dataset/session/origin keys.

SQLite integrity check returned `ok` after the complete build.

## Reproducible artifacts

- `outputs/origins/forecast_origins_v0_3_1.sqlite`
- `outputs/audit/forecast_origins_full_v0_3_1.json`
- `notebooks/02_forecast_origin_audit.ipynb`
- `src/build_forecast_origins.py`
- `tests/test_forecast_origins.py`
