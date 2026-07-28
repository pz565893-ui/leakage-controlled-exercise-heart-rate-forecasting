# Literature search and verification report

**Evidence freeze:** 2026-07-23  
**Workflow:** targeted multi-source verification, BibTeX reconciliation, and reference-management audit  
**Scope:** exercise heart-rate modelling plus conformal/time-series uncertainty methods; not a systematic review

The 23 July reproducible targeted update added four verified records (`qiu2021`, `gilbert2022`, `fedorin2021`, and `kasl2024`). Its exact query strings, source route, OpenAlex rate-limit event, primary verification, item keys, and revised positioning boundary are recorded in [`TARGETED_LITERATURE_UPDATE_2026-07-23.md`](TARGETED_LITERATURE_UPDATE_2026-07-23.md).

## 1. Reference-state reconciliation

| Artifact or collection | Verified state | Interpretation |
|---|---:|---|
| `references/references.bib` | **41 unique citation keys** | Consolidated bibliography after the 22 July reconciliation, AdamW correction, and four-record targeted update |
| `references/recent_verified_v0_21_0.bib` | **12 unique citation keys** | Seven direct HR-modelling studies and five conformal/time-series methods papers; all 12 keys are already present in `references.bib` |
| Zotero collection snapshot inspected for the search | **41 intended top-level records** | Point-in-time post-import reconciliation matched the 41-record local bibliography; later membership drift is separately audited |
| Recent records imported into Zotero | **12 of 12** | Imported into collection `YJKPV56Q` on 22 July 2026; ten DOI records and two JMLR URL records each matched exactly once |
| Targeted-update records imported into Zotero | **4 of 4** | Imported into collection `YJKPV56Q` on 23 July 2026; item keys are recorded in the formal update log |

The verified 12-record BibTeX file was imported only after the selected connector target was confirmed as collection `YJKPV56Q`. A separate EndoMondo dataset record was subsequently imported as Zotero item type `dataset` from the official UC San Diego FitRec page; AdamW and the four targeted-update records were then added on 23 July. Point-in-time post-import reconciliation found 41 intended top-level records and zero duplicated DOI/URL identifier groups among them. No existing record was edited, moved, or deleted. A later read-only audit found unrelated membership drift; the frozen 41-entry project bibliography remains the search authority.

## 2. Search concepts and evidence hierarchy

The targeted search covered:

- future heart-rate forecasting from wearable or exercise time series;
- personalized HR response modelling and use of prior individual records;
- current-window HR estimation versus explicitly future HR prediction;
- user-disjoint, temporal, held-activity, and cross-dataset evaluation;
- prediction intervals, conformal calibration, temporal dependence, and distribution shift;
- leakage caused by user/session/window overlap and post-origin information.

Substantive coding was based on publisher pages, DOI landing pages, open full text, proceedings pages, JMLR article pages, or author-hosted primary manuscripts. Bibliographic identity was checked against DOI or official journal metadata. Search-result snippets were not treated as sufficient evidence for a positive design classification when the primary record did not establish it.

Coding rules used in the companion comparison:

- **Yes** and **No** are used only when the inspected primary evidence supports the classification.
- **NR** means not reported or not established from the inspected evidence; it is not equivalent to No.
- A chronological split within each known participant is not an unseen-user split.
- Several datasets evaluated separately do not establish frozen source-to-target external validation.
- A probabilistic model, residual distribution, confidence interval around an aggregate metric, or fixed error band is not automatically an individual predictive interval.

## 3. Verified 2021–2026 direct HR records

| Citation key | Primary record | Why it enters the comparison |
|---|---|---|
| `qiu2021` | [SciTePress / DOI](https://doi.org/10.5220/0010630600003059) | Personalized mountain-biking HR forecasting for one cyclist on one course with a chronological 80/20 split; no unseen-user, held-sport, independent-source, or calibrated-interval evaluation |
| `gilbert2022` | [SciTePress / DOI](https://doi.org/10.5220/0011541800003321) | Biking HR forecasts up to 10 min that use future course-gradient values, an intentionally different information boundary from the present past-only task |
| `fedorin2021` | [ACM / DOI](https://doi.org/10.1145/3447993.3482870) | Consumer-wearable HR-trend forecasting during HIIT; split, transfer, and predictive-interval fields not established by accessible primary metadata remain `NR` |
| `kayange2024` | [Electronics / DOI](https://doi.org/10.3390/electronics13193888) | Hybrid physiological/DBN and neural modelling of personalized HR response using FitRec workout data and recent workout history |
| `namazi2025` | [Sports / DOI](https://doi.org/10.3390/sports13030087) | Wearable sports HR prediction using SSA-augmented LSTM/CNN/PINN/RNN models with HR, breathing rate, and RR inputs |
| `desabbata2025` | [Journal of Healthcare Informatics Research / DOI](https://doi.org/10.1007/s41666-025-00191-y) | Explicit real-time, one-step-ahead HR forecasting with rolling autoregressive models and chronological out-of-sample evaluation |
| `mateescu2025` | [Machine Learning with Applications / DOI](https://doi.org/10.1016/j.mlwa.2025.100746) | Transformer forecasting of HR in daily-activity context using a confidential longitudinal patient dataset |
| `zhang2026` | [Computer Methods and Programs in Biomedicine / DOI](https://doi.org/10.1016/j.cmpb.2026.109240) | Physiological-model-based neural estimation of the metabolic–HR relationship across physical activities; relevant but not an explicit fixed-lead forecasting comparator |
| `namazi2022` | [PeerJ / DOI](https://doi.org/10.7717/peerj.14601) | One-step HR prediction combining singular spectrum analysis and copula-based analysis |
| `zhu2022` | [Methods / DOI](https://doi.org/10.1016/j.ymeth.2022.06.006) | Smartwatch training system with activity-specific LSTM models and approximately 5-s-ahead HR prediction |

These papers broaden the direct comparison materially. They prevent claims that future HR prediction, personalized exercise-HR modelling, activity-conditioned HR modelling, minute-scale cycling-HR forecasting, or consumer-wearable HIIT trend forecasting are absent from prior work.

## 4. Verified conformal and dependent-time-series records

| Citation key | Primary record | Methodological role |
|---|---|---|
| `oliveira2024` | [JMLR article](https://jmlr.org/papers/v25/23-1553.html) | Split conformal prediction when observations are non-exchangeable; supports explicit discussion of dependence-sensitive coverage |
| `sousa2024` | [Neurocomputing / DOI](https://doi.org/10.1016/j.neucom.2024.128434) | Adaptive heteroscedastic conformal forecasting for multi-step time series |
| `schlembach2025` | [Machine Learning / DOI](https://doi.org/10.1007/s10994-024-06722-9) | Conformal multistep-ahead multivariate time-series forecasting |
| `gibbs2024` | [JMLR article](https://jmlr.org/papers/v25/22-1218.html) | Online conformal inference under arbitrary distribution shifts |
| `xu2023` | [IEEE TPAMI / DOI](https://doi.org/10.1109/TPAMI.2023.3272339) | Conformal prediction designed for time-series dependence |

These methods do not retrospectively create a coverage guarantee for densely overlapping exercise origins nested within sessions and users. They instead establish that dependence, online adaptation, horizon structure, and shift-aware calibration are active methodological alternatives that the manuscript must acknowledge when interpreting fixed split-CQR results.

## 5. Adjacent cross-dataset wearable evidence

| Citation key | Primary record | Methodological role |
|---|---|---|
| `kasl2024` | [PMLR proceedings](https://proceedings.mlr.press/v248/kasl24a.html) | Cross-study analysis of wearable datasets and generalizability of acute-illness monitoring models; adjacent rather than direct exercise-HR forecasting evidence, but directly relevant to separating within-source performance from cross-dataset transport |

Kasl et al. support the need for an explicit cross-dataset boundary in wearable modelling. Their acute-illness task does not establish exercise-HR forecast accuracy or predictive-interval transport in the present setting.

## 6. Updated positioning boundary

The structured targeted search found prior studies addressing personalized exercise-HR forecasting, future covariate-assisted biking forecasts, consumer-wearable HIIT trend forecasting, participant-independent evaluation, multi-dataset wearable modelling, and conformal uncertainty as separate strands. Direct evidence combining these strands under one boundary-aligned exercise-HR protocol remains limited:

> Prior work separately establishes short- and longer-horizon HR prediction, personalized physiological modelling, participant-independent wearable evaluation, activity-conditioned models, cross-dataset wearable generalization, and conformal methods for dependent or shifted time series. We integrate these strands in a fixed-origin 1/3/5-min exercise-HR study with explicit temporal, user, sport, joint-shift, and frozen cross-source boundaries. The contribution is methodological integration and auditable transportability evidence rather than architectural novelty.

This is a targeted evidence audit. It does not support the words *first*, *only*, or *no previous study* without a separately registered, reproducible systematic search.

## 7. Reference-manager verification

The 12 records in `recent_verified_v0_21_0.bib`, one separately typed EndoMondo dataset record, AdamW, and the four targeted-update records were imported into the author-approved Zotero collection after explicit target verification. The reconciled point-in-time state was **41 unique bibliography keys; 41 intended Zotero top-level records; 12 of 12 recent research/method records imported; one of one EndoMondo dataset addition imported; four of four targeted-update records imported; zero duplicate DOI/URL identifier groups among the intended records**. The targeted item keys and primary verification route are recorded in `TARGETED_LITERATURE_UPDATE_2026-07-23.md`; current live-state drift is recorded in `outputs/audit/ZOTERO_REFERENCE_AUDIT_2026-07-23.md`.
