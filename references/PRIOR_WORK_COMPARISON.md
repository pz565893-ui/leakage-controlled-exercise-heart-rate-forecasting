# Prior-work comparison for exercise heart-rate forecasting

**Evidence freeze:** 2026-07-23  
**Reference state:** 41 unique records in `references/references.bib`  
**Reference-manager snapshot used for this audit:** 41 intended records reconciled in Zotero collection `YJKPV56Q` on 23 July 2026; a later read-only audit found unrelated collection-membership drift, documented in `outputs/audit/ZOTERO_REFERENCE_AUDIT_2026-07-23.md`

## 1. Scope and evidence rules

This is a **targeted positioning audit**, not a systematic review or a claim of complete global coverage. The authoritative 41-record project bibliography combines the original evidence set, the 12 DOI- or journal-verified records in `references/recent_verified_v0_21_0.bib`, the separately typed EndoMondo dataset record, the AdamW optimizer method record, and four records added by the 23 July targeted update. Those 41 intended records were reconciled to Zotero collection `YJKPV56Q`; later unrelated collection-membership drift does not change the frozen bibliography used here.

The formal query strings, source route, OpenAlex rate-limit event, primary-record checks, and four added item keys are recorded in [`TARGETED_LITERATURE_UPDATE_2026-07-23.md`](TARGETED_LITERATURE_UPDATE_2026-07-23.md). That reproducible update was designed to challenge the manuscript's positioning; it was not registered as a systematic search.

Inclusion in the task-level comparison required relevance to at least one of the following: future HR forecasting, exercise-HR estimation, personalization, user-independent evaluation, held-activity evaluation, external validation, leakage control, temporal dependence, distribution shift, or predictive intervals. Generic architecture and statistical-test references remain in the bibliography but are not treated as direct HR prior work.

Coding rules:

- **Yes** or **No** is used only when the inspected primary source supports the classification.
- **NR** means not reported or not established from the inspected evidence; it does not mean No.
- **Unseen user** requires participant-disjoint evaluation. A later segment or workout from a participant represented in training is a temporal test, not an unseen-user test.
- **Individual history** distinguishes prior completed workouts or participant-specific fitting from the ordinary lagged samples inside the current input window.
- **Held sport** requires that an activity or sport family be absent from model fitting and then evaluated. Training a separate model for every observed activity is not held-sport transfer.
- **External data** requires an independently sourced evaluation with a frozen source-trained pipeline. Analysing multiple datasets separately or pooling all sources during development is not frozen external validation.
- **Predictive interval** means an interval for an individual future HR outcome. A probabilistic architecture, Monte Carlo sample mean, residual histogram, aggregate confidence interval, or fixed measurement-error band is not sufficient.

## 2. Reference design: the present study

| Task and information set | Prediction horizons | User boundaries | Individual history | Sport boundaries | External boundary | Uncertainty output |
|---|---|---|---|---|---|---|
| Recorded exercise-HR forecasting from the 300 s available before each fixed origin; Endomondo development and GoldenCheetah frozen evaluation | +60, +180, and +300 s | Strict within-user temporal test plus a separate unseen-user test; calibration and evaluation users separated where required | Completed prior workouts only; matched history-informed and zero-history inference from one checkpoint | Leave-one-sport-family-out models and a joint unseen-user–sport regime | Frozen Endomondo model, normalizers, ontology, and calibration thresholds applied without GoldenCheetah fine-tuning or recalibration | Seven ordered quantiles, split CQR, central 50%, 80%, and 90% intervals, coverage, width, pinball loss, and WIS |

The contribution is defined by this integrated deployment-boundary protocol. It is not premised on a wholly new neural architecture.

## 3. Earlier direct HR comparators in the evidence set

| Study | Task / data | Prediction range | User split | Individual history | Held sport | External data | Predictive interval | Main distinction |
|---|---|---|---|---|---|---|---|---|
| **Ni et al., 2019 (FitRec)** [@ni2019] | Personalized Endomondo speed/HR profile modelling | Full-workout profile plus next 10-s sample | Per-user chronological 80/10/10; not unseen-user | Yes; user representation and most recent workout | No documented held-sport test | No independent source | No documented interval | Closest Endomondo predecessor, but not the same 1/3/5-min user/sport/source-shift protocol |
| **Nazaret et al., 2023** [@nazaret2023] | Personalized HR response for outdoor runs in the Apple Heart and Movement Study | Future workout HR trend, up to 2 h | First 80% of each participant's workouts train; final 20% test | Yes; prior-workout history encoder | No; outdoor running cohort | No independent source documented | No; the plotted ±5 bpm band is an assumed measurement SD | Uses the target workout intensity sequence and same-user history; does not establish unseen-user or frozen source transfer |
| **Hallgrímsson et al., 2018** [@hallgrimsson2018] | Minute-level free-living HR modelling at population scale | Explicit future lead time NR | Same individual's 2017 signature assessed in 2018; not unseen-user | Yes; longitudinal participant signature | No held-sport test documented | No frozen HR-transfer dataset | NR | Demonstrates stable person-specific structure, not the present short-horizon shift design |
| **Pacheco et al., 2024** [@pacheco2024] | Exercise HR estimation from accelerometry/demographics when PPG is unavailable or corrupted | Described as several minutes; exact lead definition NR | NR | Online within-workout parameter adaptation | NR | Five datasets evaluated; frozen source-to-target transfer NR | NR | Multi-dataset, adaptive sensor-pipeline estimation rather than a no-adaptation external forecast protocol |
| **Reiss et al., 2019** [@reiss2019] | PPG/accelerometer same-window HR estimation under motion artefact | Current 8-s window, not a future response forecast | Leave-one-session-out; subject-independent when one session represented one subject | No personalization mechanism | No held-activity transfer | Several datasets analysed separately; no frozen cross-source transfer documented | No | Supports subject-independent wearable evaluation but addresses current-window reconstruction |

## 4. Verified 2021–2026 direct HR comparators

| Study | Task / data | Prediction range | User split | Individual history | Held sport | External data | Predictive interval | Conservative interpretation |
|---|---|---|---|---|---|---|---|---|
| **Qiu et al., 2021** [@qiu2021] | Personalized mountain-biking HR forecasting for one cyclist on one 8.71-mile course | Future HR along the course; an exact fixed physical lead is **NR** | **No** unseen-user test; chronological first 80% train and final 20% test from the same cyclist/course | Completed-workout history **NR** | **No** held-sport test; mountain biking on one course | **No** independent source | **No** calibrated predictive interval documented | Direct personalized cycling-HR comparator, but its single-rider, course-specific temporal split does not test transfer to new users, sports, or sources |
| **Gilbert et al., 2022** [@gilbert2022] | LSTM forecasting of HR in biking with course-gradient covariates | Up to **10 min** | Participant-disjoint evaluation **NR** from the inspected primary evidence | Completed-workout history **NR** | No held-sport protocol documented | No frozen independent-source evaluation documented | No calibrated predictive interval documented | Uses **future course-gradient values** as inputs; this is an intentionally different information boundary from a past-only forecast and should not be treated as a like-for-like comparator |
| **Fedorin et al., 2021** [@fedorin2021] | Consumer-wearable HR-trend forecasting during high-intensity interval training | Exact physical forecast lead **NR** in the accessible primary metadata | **NR** | **NR** | **NR** | **NR** | **NR** | Establishes a wearable HIIT trend-forecasting task, but the short proceedings record does not establish the split, transfer, or interval fields needed for positive coding |
| **Kayange et al., 2024** [@kayange2024] | Hybrid physiological DBN/LSTM modelling on 38,323 FitRec workouts from 665 individuals | Future whole-workout HR profile; retained workouts last 10 min–2 h 20 min, but fixed +1/+3/+5-min leads are **NR** | **Yes**, user-grouped 80/20 train/validation: one user's workouts do not cross subsets; no independent final test is described | **Yes**; completed recent/past workouts inform the personalized representation | **NR**; no leave-one-sport evaluation is reported | **No**; FitRec only | **No** calibrated predictive interval; Figure 6 uses a fixed ±5-bpm band without nominal coverage or calibration | Strong completed-history and user-isolation comparator, but with a different horizon, validation, and uncertainty estimand |
| **Namazi et al., 2025** [@namazi2025] | SSA-augmented LSTM/CNN/PINN/RNN HR prediction from HR, breathing rate, and RR intervals; 126 recordings from 81 participants across 10 sports | Future epochs are discussed, but the input window and physical time lead are **NR** | 80/20 data split; whether the split unit is a point, recording, or participant is **NR** | Completed-session/user history **NR**; the reported inputs are physiological series within recordings | **NR**; no held-sport protocol is reported | **No**; one Sport Database source | **NR**; no predictive interval or coverage analysis is reported | Demonstrates multivariate wearable sports-HR modelling, but user and sport transfer boundaries cannot be inferred from an unspecified 80/20 split |
| **De Sabbata & Simonini, 2025** [@desabbata2025] | Per-user univariate ARIMA/random-walk forecasting on MMASH (22 users, Polar H7) and RRITS (147 users, Holter), each containing 24-h daily-life HR | One step ahead at 1-min granularity; 15/30/45/90/150-min rolling windows and 220 chronological forecasts per user | **No** unseen-user split; each user is fitted and evaluated independently with a later out-of-sample segment | **Yes**, the same user's recent 15–150 min; not completed-workout history | **No** sport holdout | **No** frozen external transfer; both datasets are analysed by separate per-user fits | **No** predictive interval; point forecasts, MAE, and residual diagnostics | Important explicit real-time baseline showing that simple random-walk/ARIMA models can be competitive at a one-minute horizon |
| **Mateescu et al., 2025** [@mateescu2025] | Activity-conditioned Transformer with Laplace diffusion; private Fitbit data from 29 people over 4 months, approximately 20,000 HR records and 294 activity sessions | Non-overlapping sequence-to-sequence **L input steps to L future steps**; L and its physical time unit are **NR** | 65/15/20 train/validation/test split; whether splitting is by person, session, or window is **NR** | **No** completed-session/person-history module is documented; inputs are the preceding L HR/context steps and derived/activity features | **No**; activity-specific encoders and results are used without a held-activity test | **No**; one private cohort | **No** reported interval; repeated diffusion forecasts are summarized by the pointwise median without coverage analysis | Relevant activity-context and longitudinal forecasting comparator, but user separation and physical horizon remain unresolved |
| **Zhang et al., 2026** [@zhang2026] | Physiological-model-based neural estimation of HR from contemporaneous VO₂ using WEEE and ACTES (25 participants total) | **0 s; HR(t) estimation**, not future HR forecasting | **No** unseen-user test; each participant's activity/power segments were split 80/20 and fitted participant-wise | Participant-specific training and identifiable parameters, but no completed-workout history encoder | **No**; activities/power segments enter fitting, and ACTES contains cycling only | **No**; both sources enter development/participant-wise fitting rather than frozen transfer | **No**; R²/RMSE/MAE point estimates and VO₂-noise robustness | A physiological personalization comparator that must not be represented as future forecasting |
| **Namazi, 2022** [@namazi2022] | Univariate HR forecasting on running records from 10 healthy participants in the Sport Database | Past 1,500 s used to forecast the next **30 s** | Participant-disjoint split **NR**; a within-series historical segment and future segment are described | **Yes**, same-record past 1,500 s; not cross-session completed-workout history | **No**; running only | **No**; one source | **No**; copula samples are averaged to a point forecast and MAE is reported | Establishes a 30-s future HR task, but not unseen-user, sport-shift, external, or interval evaluation |
| **Zhu et al., 2022** [@zhu2022] | Activity-specific LSTMs using HR and wrist inertial signals for walking, running, and rope jumping; PAMAP2 plus a TicWatch experiment | Explicitly evaluates **5, 7, 10, 15, 20, and 25 s**; main result emphasizes 5 s ahead | **Yes**; nine-fold participant split with eight users for training and one for testing | **No** completed-user-history encoder; current 5.12-s sensor window only | **No**; a separate model is fitted for each observed activity | **NR**; the TicWatch experiment is not documented as a frozen independent source transfer | **No**; point prediction metrics only | Provides a genuine unseen-user, short-lead exercise-HR comparator, but no unseen-activity or calibrated-interval design |

### Direct-literature interpretation

The structured targeted search changes the positioning in three important ways.

First, future HR prediction is not absent from prior work: Qiu et al. (2021), Gilbert et al. (2022), Fedorin et al. (2021), Namazi (2022), Zhu et al. (2022), De Sabbata and Simonini (2025), and other included studies forecast future HR or HR trends at horizons ranging from seconds to minutes or longer workout trajectories. Second, user-independent evaluation is not absent: Zhu et al. report participant-fold evaluation, and Reiss et al. use session/subject-independent HR estimation. Third, activity-conditioned and personalized models are established by Qiu et al., Gilbert et al., Kayange et al., Mateescu et al., Nazaret et al., and Zhang et al.

The defensible gap is narrower: the structured targeted search found prior studies addressing these components separately, while direct evidence combining the same past-only 1/3/5-min information set with temporal, unseen-user, held-sport, joint user–sport, and frozen cross-source evaluations plus calibrated central prediction intervals remains limited.

## 5. Measurement and external-source context

| Record | Evidence supplied | Boundary on interpretation |
|---|---|---|
| **Shcherbina et al., 2017** [@shcherbina2017] | Device- and activity-dependent wrist-wearable HR error against telemetry | Supports measurement heterogeneity, not external validation of the present forecasting model |
| **Bent et al., 2020** [@bent2020] | Wearable-versus-ECG accuracy across devices and conditions | Supports source/device shift concerns; aggregate accuracy uncertainty is not forecast uncertainty |
| **GoldenCheetah OpenData** [@goldencheetah2018] | Independent athlete-contributed workout repository | Establishes source independence, not a synchronized ECG or controlled device-validation cohort |
| **Kasl et al., 2024** [@kasl2024] | Cross-study analysis of wearable datasets and generalizability of acute-illness monitoring models | Adjacent rather than direct exercise-HR forecasting evidence; shows why within-study performance should not be equated with cross-dataset generalization |

GoldenCheetah therefore supports a platform/source transportability test. It does not identify whether any performance change is caused by sensor brand, placement, firmware, participant selection, activity mix, preprocessing, or another source characteristic.

Kasl et al. independently reinforce the broader wearable-methods rationale for an explicit cross-dataset boundary. Their illness-monitoring task does not supply evidence about exercise-HR forecast accuracy or interval calibration in the present setting.

## 6. Leakage, dependence, shift, and interval methods

| Study | Methodological evidence | Relevance and limitation for the present study |
|---|---|---|
| **Dehghani et al., 2019** [@dehghani2019] | Overlapping windows and subject-dependent versus subject-independent validation in wearable activity recognition | Motivates splitting people/sessions before windowing; not direct HR forecasting evidence |
| **Kapoor & Narayanan, 2023** [@kapoor2023] | Taxonomy of leakage in ML-based science | Supports explicit control of user, session, history, sport, calibration, and external boundaries |
| **Koenker & Bassett, 1978** [@koenker1978] | Regression quantiles | Basis for conditional quantile outputs; no conformal or wearable-shift guarantee |
| **Romano et al., 2019** [@romano2019] | Conformalized quantile regression | Basis for split CQR under its assumptions; does not guarantee transport across shifted wearable sources |
| **Angelopoulos & Bates, 2023** [@angelopoulos2023] | Conformal prediction principles and extensions | Supports terminology and assumption disclosure, not exchangeability of clustered exercise origins |
| **Ovadia et al., 2019** [@ovadia2019] | Empirical uncertainty behaviour under dataset shift | Motivates measuring external calibration instead of assuming it |
| **Bracher et al., 2021** [@bracher2021] | Interval score and WIS | Supplies proper interval evaluation; not an HR comparator |
| **Oliveira et al., 2024** [@oliveira2024] | Split conformal prediction for non-exchangeable data | Directly strengthens the dependence caveat; fixed origin-pooled calibration is only one estimand and does not automatically equalize users or sessions |
| **Sousa et al., 2024** [@sousa2024] | Adaptive heteroscedastic conformal inference for multi-step time-series forecasts | Shows that horizon-specific, changing uncertainty can be updated online; such adaptation would change a strictly frozen external-validation estimand |
| **Schlembach et al., 2025** [@schlembach2025] | Conformal multistep-ahead multivariate time-series forecasting | Supports joint consideration of horizon and multivariate dependence; not a plug-in guarantee for nested workout origins |
| **Gibbs & Candès, 2024** [@gibbs2024] | Online conformal inference under arbitrary distribution shifts | Provides an adaptive-shift alternative; it requires a sequential feedback setting and is distinct from no-recalibration external testing |
| **Xu & Xie, 2023** [@xu2023] | Conformal prediction for time series | Establishes that temporal dependence needs dedicated treatment; it does not resolve participant/session clustering without an aligned calibration design |

## 7. Dependence and shift implications

Dense forecast origins from the same workout share most of their 300-s context and are nested inside sessions and participants. Treating those origins as exchangeable independent calibration units can allow long sessions or high-volume users to dominate a threshold. The non-exchangeable and time-series conformal literature therefore supports reporting the calibration unit and sensitivity to user/session weighting. It does not justify claiming a finite-sample user-level guarantee from an origin-level split-CQR calculation.

Multi-step methods also clarify that +60, +180, and +300 s are different uncertainty problems. Separate horizon thresholds and horizon-specific coverage are appropriate, while simultaneous path coverage would be a stronger and different estimand.

Finally, online/adaptive conformal methods provide possible responses to continuing shift. They should not be conflated with the current frozen external experiment: updating thresholds on GoldenCheetah outcomes would answer an adaptation question rather than pure transportability. The present design should therefore report external under- or over-coverage as an empirical finding, not claim distribution-free protection under arbitrary shift.

## 8. Evidence-supported positioning

### Targeted-search positioning statement

> Prior work separately establishes personalized exercise-HR forecasting, short- and longer-horizon HR prediction, participant-independent wearable evaluation, activity-conditioned models, cross-dataset wearable generalization, and conformal methods for dependent or shifted time series. Evidence combining all of these elements under one past-only, boundary-aligned exercise-HR evaluation remains limited. We therefore position the contribution as an integrated leakage-controlled evaluation and transportability framework rather than a new forecasting task, a globally first study, or a wholly new architecture.

Suggested concise English:

> Prior work separately establishes short-lead and workout-level HR prediction, personalized physiological modelling, participant-independent wearable evaluation, activity-conditioned models, and conformal methods for dependent or shifted time series. We integrate these strands in a fixed-origin 1/3/5-min exercise-HR study with explicit temporal, user, sport, joint-shift, and frozen external boundaries. The contribution is methodological integration and auditable transportability evidence rather than architectural novelty.

Clean Chinese interpretation:

> 现有研究已经分别证明了短时距心率预测、整段训练心率建模、个体化生理建模、未见受试者评估、运动情境建模以及依赖或分布偏移条件下的保形预测方法。本研究可辩护的增量，是将这些要素整合到一个仅使用预测时点之前信息的 1、3、5 分钟运动心率框架中，并明确区分时间、用户、运动项目、用户–项目联合偏移和冻结外部数据边界。创新重点应表述为可审计的泄漏控制与可迁移性证据，而不是“首次预测心率”或“提出全新网络”。

### Claims not supported by the targeted evidence

- “This is the first exercise-HR forecasting study” or any global first/only claim.
- “No previous study predicted future HR.” Several included studies did so at seconds-, minute-, or workout-level horizons.
- “No previous study evaluated unseen users.” Zhu et al. and subject-independent estimation work provide counterexamples, although for different designs.
- “No previous study used multiple datasets.” Multiple included studies use more than one dataset; the narrower issue is frozen source-to-target transfer.
- “Multiple sports imply unseen-sport evaluation.” A sport must be excluded from fitting to qualify as held sport.
- “A probabilistic model produces calibrated prediction intervals.” Calibration, nominal levels, coverage, and interval construction must be reported.
- “CQR guarantees coverage under arbitrary source shift.” Fixed split-CQR guarantees do not automatically survive dependence or changed source distributions.
- “GoldenCheetah proves device-level validity.” It is an independent platform source, not a controlled synchronized ECG cohort.
- Equating current-window HR estimation with a future HR response forecast.
- Equating later observations from known users with unseen-user transfer.

## 9. Remaining evidence gaps

1. **Coverage gap:** the targeted evidence audit is adequate for cautious manuscript positioning but insufficient to establish global priority; a registered systematic search would be required for first/only language.
2. **Horizon-definition gap:** several papers use “prediction” without a fully specified physical lead time, input cutoff, or direct versus recursive horizon.
3. **User-boundary gap:** participant grouping is NR in several direct papers; random sample/window splits cannot be assumed participant-disjoint.
4. **Joint-shift gap:** no inspected direct record reports the same unseen-user × held-sport intersection plus frozen platform transfer.
5. **Uncertainty-transport gap:** dependent and adaptive conformal methods exist, but direct evidence on cross-source transport of calibrated multi-level exercise-HR intervals remains limited.
6. **Device-validation gap:** GoldenCheetah lacks controlled device metadata and synchronized reference ECG.
7. **Deployment gap:** retrospective forecasting does not establish prospective latency, safety, exercise-prescription benefit, or clinical utility.
8. **Calibration-unit gap:** origin-, session-, and user-weighted coverage answer different questions and should not be silently interchanged.

## 10. Primary evidence links

### Direct HR and measurement records

- Ni et al. (2019): [DOI](https://doi.org/10.1145/3308558.3313643); [author-hosted paper](https://cseweb.ucsd.edu/~jmcauley/pdfs/www19.pdf)
- Nazaret et al. (2023): [npj Digital Medicine](https://www.nature.com/articles/s41746-023-00926-4)
- Hallgrímsson et al. (2018): [arXiv](https://arxiv.org/abs/1812.01696)
- Pacheco et al. (2024): [PubMed](https://pubmed.ncbi.nlm.nih.gov/37028018/); [DOI](https://doi.org/10.1109/JBHI.2023.3251742)
- Reiss et al. (2019): [Sensors](https://www.mdpi.com/1424-8220/19/14/3079)
- Kayange et al. (2024): [Electronics](https://www.mdpi.com/2079-9292/13/19/3888); [DOI](https://doi.org/10.3390/electronics13193888)
- Namazi et al. (2025): [Sports](https://www.mdpi.com/2075-4663/13/3/87); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11946376/); [DOI](https://doi.org/10.3390/sports13030087)
- De Sabbata and Simonini (2025): [Springer article](https://link.springer.com/article/10.1007/s41666-025-00191-y); [DOI](https://doi.org/10.1007/s41666-025-00191-y)
- Mateescu et al. (2025): [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S266682702500129X); [author manuscript](https://arxiv.org/abs/2508.16655); [DOI](https://doi.org/10.1016/j.mlwa.2025.100746)
- Zhang et al. (2026): [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0169260726000088); [DOI](https://doi.org/10.1016/j.cmpb.2026.109240)
- Namazi (2022): [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9774013/); [PeerJ](https://peerj.com/articles/14601/)
- Zhu et al. (2022): [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1046202322001463); [DOI](https://doi.org/10.1016/j.ymeth.2022.06.006)
- Shcherbina et al. (2017): [DOI](https://doi.org/10.3390/jpm7020003)
- Bent et al. (2020): [npj Digital Medicine](https://www.nature.com/articles/s41746-020-0226-6)
- GoldenCheetah OpenData: [OSF](https://osf.io/6hfpz/); [DOI](https://doi.org/10.17605/OSF.IO/6HFPZ)
- Qiu et al. (2021): [DOI](https://doi.org/10.5220/0010630600003059)
- Gilbert et al. (2022): [DOI](https://doi.org/10.5220/0011541800003321)
- Fedorin et al. (2021): [DOI](https://doi.org/10.1145/3447993.3482870)
- Kasl et al. (2024): [PMLR](https://proceedings.mlr.press/v248/kasl24a.html)

### Leakage and uncertainty methods

- Dehghani et al. (2019): [Sensors](https://www.mdpi.com/1424-8220/19/22/5026)
- Kapoor and Narayanan (2023): [Patterns](https://doi.org/10.1016/j.patter.2023.100804)
- Koenker and Bassett (1978): [DOI](https://doi.org/10.2307/1913643)
- Romano et al. (2019): [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
- Angelopoulos and Bates (2023): [Foundations and Trends in Machine Learning](https://www.nowpublishers.com/article/Details/MAL-101)
- Ovadia et al. (2019): [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)
- Bracher et al. (2021): [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618)
- Oliveira et al. (2024): [JMLR](https://jmlr.org/papers/v25/23-1553.html)
- Sousa et al. (2024): [DOI](https://doi.org/10.1016/j.neucom.2024.128434)
- Schlembach et al. (2025): [DOI](https://doi.org/10.1007/s10994-024-06722-9)
- Gibbs and Candès (2024): [JMLR](https://jmlr.org/papers/v25/22-1218.html)
- Xu and Xie (2023): [DOI](https://doi.org/10.1109/TPAMI.2023.3272339)

The 12 recent verified records and the EndoMondo dataset record were added to the author-approved target collection on 22 July 2026. The AdamW method record and the four targeted-update records (`qiu2021`, `gilbert2022`, `fedorin2021`, and `kasl2024`) were added on 23 July 2026. Existing Zotero records were not edited, moved, or deleted; the point-in-time post-import reconciliation found 41 top-level records and no duplicated DOI/URL identifier groups among them. A later live-state audit identified unrelated membership drift and is the authority for the current Zotero count.
