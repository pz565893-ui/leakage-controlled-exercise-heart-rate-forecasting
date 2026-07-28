# Physiological Measurement submission profile and compliance gates

**Target journal:** *Physiological Measurement* (PM)  
**Article type:** Full paper  
**Checked:** 2026-07-23  
**Authoritative sources:** [journal page](https://www.sciencedirect.com/journal/physiological-measurement) and [Guide for Authors](https://www.sciencedirect.com/journal/physiological-measurement/publish/guide-for-authors)

## Scope fit

The manuscript is framed as an applications-led biomedical signal-processing study of wearable exercise heart-rate streams. The central technical questions are robustness to user and sport distribution shifts, causal multi-horizon forecasting, calibrated predictive uncertainty, and leakage-controlled evaluation. The study does not claim diagnosis, therapy, disease prevention, or causal physiology.

This framing matches PM's stated interest in physiological measurement, wearable data, and practical applications. Scope fit still depends on demonstrating signal-processing value rather than presenting only a generic machine-learning benchmark.

## Mandatory submission constraints

| Requirement | PM instruction | Project gate |
|---|---|---|
| Full-paper length | typically around 5,000 words | current main text is 5,938 words including headings; extensive robustness tables are in the supplement |
| Abstract | typically fewer than 250 words | current abstract is 241 words and contains no citations |
| Keywords | 1-7 English keywords | 6 concise keywords are prepared |
| Highlights | 3-5 bullets, at most 85 characters including spaces | 5 validated bullets are prepared in a separate editable file |
| Source format | editable `.doc/.docx` or `.tex`; Word should be single-column | the rebuilt primary source is a single-column `.docx` |
| Article structure | numbered sections and subsections | numbered Introduction, Materials and methods, Results, Discussion, and Conclusions are used |
| Figures | separate files, captions required, accessible colors | vector PDF/SVG working files and PM-compatible PDF/TIFF/PNG deliverables are prepared |
| Graphical abstract | optional; current AI policy restricts tool choice | omit by default; do not submit the current internal artifact unless a licensed dedicated scientific/professional illustration workflow is documented |
| Research data | data sharing policy is handled in submission form | cite original public datasets; archive code, manifests, split definitions, aggregate figure source data, and non-identifying audits without redistributing raw or row-level source-derived records |
| Data statement | required at submission | the manuscript Data availability section is present; the live submission-system statement remains an upload task |
| CRediT | corresponding author must report contributor roles | leave named-role placeholders until the author list is supplied |
| AI disclosure | generative-AI manuscript assistance must be declared | include the required declaration before references; final author review and responsibility confirmation remain mandatory |
| Sex/gender reporting | address SGBA or state the limitation | describe recorded variable semantics; limit external subgroup inference because GoldenCheetah is strongly male-dominated |

## Scientific quality gates beyond formatting

The paper is not submission-ready until all of the following are supported by executed artifacts:

1. session and user splits are assigned before forecast-window construction;
2. exact-signal duplicates are controlled before all protocols;
3. target construction is causal with respect to the forecast origin;
4. within-user, unseen-user, unseen-sport, joint-shift, and frozen cross-source protocols are reported separately;
5. persistence, statistical/tree, recurrent, convolutional, and transformer baselines share identical inputs and splits;
6. primary point accuracy is user-aggregated and accompanied by uncertainty calibration metrics;
7. calibration data are disjoint from model selection and final tests;
8. paired uncertainty intervals or user-clustered bootstrap intervals accompany model comparisons;
9. executed sensitivities isolate completed-workout history, inference-time history masking versus zero-history training, HR-only versus multimodal input, origin spacing, and calibration estimands; no unexecuted sport-encoding or missing-mask component claim is made;
10. conclusions remain within the observed platforms, users, sensors, sports, and horizons.

## Prepared manuscript package

- main manuscript with 5,938 words including headings and a 241-word abstract;
- separate title page;
- 5 validated highlights;
- omit the optional graphical abstract by default unless a policy-compliant workflow is documented;
- main figures and editable tables;
- supplementary methods, ontology, cohort flow, extra horizons, subgroup results, leakage diagnostics, and Tables S1--S18;
- code/data availability statement;
- placeholder sections/templates for CRediT, funding, competing interests, and acknowledgements, plus a completed AI-use disclosure pending final author confirmation;
- cover letter explaining PM signal-processing relevance and the frozen cross-source evaluation.

## Current status

The scope, data audit, ontology, duplicate control, pre-window split manifests, causal origin construction, sensitivity analyses, figures, Highlights, cover letter, and English manuscript have executable artifacts. The formal v0.22.0 multiseed analysis is complete for all 37 declared jobs, and the zero-history-trained v0.23.0 analysis is complete for all 10 declared jobs. Frozen-prediction v0.24.0 balanced-calibration and sport-composition analyses, v0.25.0 paired-user bootstraps, v0.26.0 persistence-conformal baseline, v0.27.0 matched-sport sensitivity, and v0.28.0 deliberately leaky negative-control analyses were integrated without model retraining or GoldenCheetah recalibration. The reported-number validator passes 473/473 checks, including 47 provenance-path checks, and all 169 unit tests pass. The final document sources state the history-masked cross-source boundary, data-reuse limits, pseudonymized-data status, empirical-interval limitations, and figure-level AI provenance. The rebuilt 18-page main manuscript, 19-page supplement, one-page Highlights file, and two-page caption file pass structure, accessibility, and page-by-page render QA. The release policy excludes linkable free-text ontology labels, subgroups below 10 users, the internal graphical abstract, row-level files, and checkpoints while including the safe raw-source integrity audit. The privacy-conservative working-tree manifest has been regenerated and verified; the final tagged release remains author-controlled.

The technical manuscript package is `PASS / AUTHOR_INPUT_REQUIRED`, not yet upload-ready. Remaining blockers require author or institutional decisions: final authorship and affiliations, corresponding-author details, institutional ethics determination, CRediT roles, funding, competing interests and Declaration Tool file, acknowledgements, scoped software/content licences and release rights, final AI-disclosure confirmation, and a versioned public code release with a persistent DOI. The live submission portal must also be reconfirmed, and the optional graphical abstract should be omitted unless a policy-compliant route is documented.
