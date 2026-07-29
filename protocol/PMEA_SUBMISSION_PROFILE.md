# Physiological Measurement submission profile

**Journal:** *Physiological Measurement* (IOP Publishing for IPEM)  
**Article type:** Research paper  
**Verified:** 28 July 2026 against current official IOP pages  
**Role:** Controlling journal-specific profile for the first submission. Older BSPC files are retained as provenance but are superseded for submission decisions.

## Scope and editorial positioning

The journal publishes work on sensing, assessing, modelling, predicting, and controlling physiological functions, including physiological time-series processing, artificial intelligence and machine learning, benchmarking, mobile health, sports medicine, personal fitness tracking, and wellness monitoring. It explicitly values rigorous large-scale validation of existing methods and asks for titles and abstracts that are understandable across disciplines.

The study fits this scope as a large-scale validation and reliability paper about forecasting a recorded physiological signal. The defensible contribution is not a new neural architecture. It is the combination of past-only forecasting, user/session leakage control, user-level uncertainty evaluation, sport-family transfer, and frozen cross-source testing. The manuscript must continue to distinguish recorded wearable HR from electrocardiographic reference HR and must not make diagnostic, treatment, injury-prevention, or exercise-prescription claims.

Official source: <https://publishingsupport.iopscience.iop.org/journals/physiological-measurement/about-physiological-measurement/>

## Journal-specific requirements and current state

| Requirement | Official rule | Current project state | Gate |
| --- | --- | --- | --- |
| Research-paper length | Normally not more than 8000 words | Automated count: 6941 words from Introduction through the pre-reference declarations | Pass |
| Abstract | Maximum 250 words; headings Objective, Approach, Main results, Significance | Automated count: 244 words; all four required headings are present and no citation is included | Pass |
| Reference style | Harvard alphabetical; article titles required | Complete document contains 35 cited, alphabetized author-year references with titles | Pass |
| Initial submission | One complete PDF for review, with authors/institutes and figures/tables embedded in the text; supplementary files uploaded separately | A complete review PDF and separate supplementary PDF were generated and visually inspected; author metadata and declarations are integrated | Pass |
| Peer-review identity | Single-anonymous or double-anonymous at author choice | The identified review file is complete; authors must choose the review model before the upload package is finalized | Author decision |
| Human group size | For papers reporting measurements on groups of human subjects, each reported group should contain 30+ subjects; exceptions require contact with the journal | Three joint user--sport cells contain only 18--20 users | Outcome estimates masked from the PMEA presentation; support counts retained |
| SAGER | Sex/gender must be considered and imbalance justified | A dedicated Methods statement now explains the metadata limitation, imbalance, and why no underpowered group comparison is reported | Pass pending author review |
| Data availability | The journal has adopted IOP's open-data policy; a data-availability statement is required | Third-party raw-data restrictions are stated; GitHub release `v0.30.0` is archived in Zenodo at https://doi.org/10.5281/zenodo.21649896 | Pass |
| Human-data ethics | IOP ethical policy applies; a suitable ethics statement is required | Secondary pseudonymized-data statement exists, but an institutional determination/reference is missing | Blocking author/institution input |
| Generative AI disclosure | Permitted uses must be disclosed in Acknowledgements with model/version and purpose; authors retain responsibility | Disclosure identifies OpenAI Codex desktop, model `gpt-5.6-sol`, and use only for English-language polishing and Chinese–English translation; authors retain responsibility | Pass |
| Conflicts/funding/authorship | Conflicts, funding, author list, roles, and corresponding-author information must be supplied | Author names/order, affiliations, emails, PANG KEREN's ORCID and corresponding-author role, CRediT roles, no-specific-funding statement, no-competing-interests statement, originality and final approval are confirmed | Pass |

## Required document architecture

The complete review PDF should contain, in this order:

1. Title, authors, affiliations, and corresponding-author details, unless the authors select double-anonymous review.
2. Structured abstract and keywords.
3. Main article with editable table 1 and figures 1--4 embedded near their first substantive discussion.
4. Acknowledgements, including funding and the IOP generative-AI disclosure.
5. Ethics statement, conflict-of-interest statement, data availability, and author contributions in the locations required by the live submission form.
6. Alphabetized Harvard references with article titles.

The supplementary material should be a separate reader-facing file. For PMEA, it must omit outcome comparisons for cells below 30 users and omit the recorded-gender performance comparison; the corresponding immutable internal artifacts remain unchanged.

## Reporting decisions specific to this submission

- The title is finding-led: boundary-dependent reliability, not architectural novelty.
- The 5-min horizon remains primary; 1- and 3-min horizons are secondary.
- Joint user--sport outcome reporting is restricted to outdoor cycling (77 users) and running (88 users). The 18--20-user cells are shown only as support limitations.
- Sex/gender performance effects are not reported because the available strata are incomplete, imbalanced, semantically limited, and below the journal threshold in key regimes.
- GoldenCheetah is described as a frozen cross-source dataset, not an independent-device validation cohort.
- Prediction intervals are described as empirical post-CQR intervals; no user-level finite-sample guarantee is claimed.

## Authoritative official sources

- Journal scope, article types, structured abstract, Harvard references, 30+ group rule, SAGER, review model and data policy: <https://publishingsupport.iopscience.iop.org/journals/physiological-measurement/about-physiological-measurement/>
- Initial-submission files and author responsibilities: <https://publishingsupport.iopscience.iop.org/publishing-support/authors/authoring-for-journals/submit-journal-article/>
- Article structure: <https://publishingsupport.iopscience.iop.org/questions/structure-and-format-of-your-journal-article/>
- Article format and embedded figures/tables: <https://publishingsupport.iopscience.iop.org/questions/article-format/>
- Harvard reference guidance: <https://publishingsupport.iopscience.iop.org/questions/references/>
- Open-data policy: <https://publishingsupport.iopscience.iop.org/iop-publishing-open-data-policy/>
- Ethical and generative-AI policy: <https://publishingsupport.iopscience.iop.org/ethical-policy-journals/>
- Submission checklist: <https://publishingsupport.iopscience.iop.org/submission-checklist/>
