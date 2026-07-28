# Targeted literature update for editorial positioning

**Search date:** 23 July 2026  
**Purpose:** strengthen the direct-comparator and cross-source context before BSPC submission  
**Scope:** targeted update, not a systematic review and not evidence for a global first/only claim

## Reproducible search route

The planned OpenAlex fallback search was attempted with the local `nature-academic-search` helper but returned HTTP 429. The search therefore continued through indexed web discovery, followed by primary-record verification on SciTePress, ACM SIGMOBILE, PMLR, and the Crossref API. Search-result snippets were not used to assign positive design features when the primary paper did not establish them.

Queries used verbatim were:

1. `heart rate forecasting wearable exercise deep learning`
2. `heart rate prediction activity generalization wearable`
3. `heart rate prediction cross dataset validation wearable`
4. `prediction interval heart rate forecasting conformal wearable`
5. `"Biomedical Signal Processing and Control" "heart rate" prediction wearable 2021 2022 2023 2024 2025`
6. `"Analyzing Machine Learning Models that Provide Personalized Heart Rate Forecasting for Elite Cyclists" DOI`
7. `"Heart rate trend forecasting during high-intensity interval training using consumer wearable devices"`

Records were eligible for the direct comparison if they forecast future exercise HR or explicitly model personalized future HR in an exercise setting. One adjacent record was retained because it directly evaluates cross-study generalization of wearable-derived HR models. DOI was the primary deduplication key; title plus first author was used when a DOI was absent.

## Added verified records

| Citation key | Record | Primary evidence and coding relevance |
|---|---|---|
| `qiu2021` | Qiu, White and Schmidt, *A Study of Machine Learning Models for Personalized Heart Rate Forecasting in Mountain Biking* | SciTePress full text and Crossref DOI `10.5220/0010630600003059`; one cyclist, one course, chronological 80/20 split, course-specific inputs, no unseen-user, held-sport, external-source, or calibrated-interval evaluation. |
| `gilbert2022` | Gilbert et al., *Using LSTM Networks and Future Gradient Values to Forecast Heart Rate in Biking* | SciTePress full text and Crossref DOI `10.5220/0011541800003321`; up to 10-min forecasts use future course-gradient values, making its information boundary intentionally different from the present past-only task. |
| `fedorin2021` | Fedorin et al., *Heart Rate Trend Forecasting During High-Intensity Interval Training Using Consumer Wearable Devices* | ACM/Crossref DOI `10.1145/3447993.3482870`; a short MobiCom paper establishing consumer-wearable HIIT trend forecasting. Design fields not established from the accessible primary metadata remain `NR`. |
| `kasl2024` | Kasl et al., *A Cross-Study Analysis of Wearable Datasets and the Generalizability of Acute Illness Monitoring Models* | Official PMLR record; adjacent rather than direct exercise-HR forecasting evidence, but directly supports the need to separate cross-dataset wearable generalization from within-source performance. |

The four entries were deduplicated against the 37-entry local bibliography, added to `references/references.bib`, and imported into Zotero collection `YJKPV56Q`. Zotero item keys are `LXU3I2B4`, `4V7MM8SF`, `G63ZRF4I`, and `M8MX4DS2`, respectively. The project bibliography and target collection therefore contained 41 top-level records immediately after this update. A later live-state membership drift is documented separately in `outputs/audit/ZOTERO_REFERENCE_AUDIT_2026-07-23.md` and does not alter this import log.

## Positioning consequence

The new papers further rule out broad claims that personalized cycling-HR forecasting, minute-scale exercise-HR forecasting, or cross-study wearable generalization are themselves novel. The defensible contribution is the integrated, auditable comparison of temporal, unseen-user, held-sport-family, joint user--sport, and retrospectively frozen cross-source boundaries together with empirical predictive-interval transport.

The manuscript should therefore say that prior studies address these components separately and that evidence combining them under one boundary-aligned evaluation remains limited. It should not make a global first/only claim, and the leakage controls should be presented as methodological validity conditions rather than as a new algorithm.

## Limitations of this update

- This was a targeted search designed to challenge and refine the manuscript's positioning, not a registered systematic search.
- OpenAlex was rate-limited during this update; the failure is recorded rather than silently treated as a completed database search.
- Several short conference papers do not expose all split, horizon, or interval details in their metadata. Unverified fields remain `NR`.
- A systematic-review protocol would be required before making an exhaustive or global-priority claim.
