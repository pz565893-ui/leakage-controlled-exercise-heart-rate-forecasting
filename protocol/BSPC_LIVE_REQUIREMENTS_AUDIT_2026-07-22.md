# BSPC live official-requirements audit — 22 July 2026

**Target journal:** *Biomedical Signal Processing and Control* (BSPC; ISSN 1746-8094)  
**Intended article type:** Full paper  
**Audit mode:** read-only comparison of current official requirements with the project package  
**Retrieval date:** 22 July 2026 (Asia/Shanghai)

> **Post-audit completion note (23 July 2026).** This file is preserved as the point-in-time record of the live BSPC requirements check. Since that check, all 37 v0.22 and 10 v0.23 formal jobs, the frozen-prediction v0.24 analyses, and the v0.25 paired-user comparisons have been completed and integrated. The current main text is 5,256 words, the abstract is 246 words, and the reported-number audit passes 330/330 checks, including 39 provenance-path checks. The 18-page main Word file, 19-page supplement, one-page Highlights file, and two-page caption file pass structure, accessibility, and page-render QA; all 148 unit tests and the privacy-conservative release-manifest verification pass. Current status is governed by `BSPC_SUBMISSION_READINESS_CHECKLIST.md`; author/institution metadata, ethics, declarations, licence/release rights, and the persistent DOI remain blocking. The historical status statements below should not be read as the present project state.

## 1. Evidence boundary and superseding finding

The complete current BSPC Guide for Authors was retrieved through Elsevier's official journal/ISSN redirect and inspected on 22 July 2026. It resolves several items that an earlier local audit had left conditional because the direct ScienceDirect URL had returned HTTP 403. The live guide now directly confirms the article length guidance, abstract limit, keyword range, single-anonymized review mode, Highlights rules, graphical-abstract status, Option C research-data policy, editable-source requirements, reference policy, and required declarations.

This report therefore supersedes the statement in `protocol/BSPC_REQUIREMENTS_RECHECK_2026-07-22.md` that those journal-specific fields were inaccessible. It does not modify any manuscript or submission artifact.

Status labels:

- **PASS:** a current official requirement is evidenced by the current package.
- **GAP:** the package does not yet meet the official requirement or contains an unresolved placeholder.
- **CONDITIONAL:** official texts conflict or the answer depends on documented provenance/author action.
- **NOT REQUIRED:** the live guide expressly makes the item optional or unnecessary for this submission route.

## 2. Current official sources

1. [BSPC Guide for Authors](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control/publish/guide-for-authors) — controlling journal-specific requirements. The official Elsevier ISSN URL `https://www.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094/guide-for-authors` redirects to this page.
2. [BSPC description and current scope, Elsevier Shop](https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094) — current scope explicitly includes ML/DL, wearables, physiological modelling, robustness, personalization, and benchmark/dataset papers.
3. [Elsevier generative-AI policies for journals](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals) — explicitly marked **updated June 2026**; controls AI-assisted manuscript, code, data-visualization, explanatory-image, and graphical-abstract use.
4. [Elsevier Highlights instructions](https://www.elsevier.com/researcher/author/tools-and-resources/highlights) — Word file, 3–5 items, no more than 85 characters each, no jargon/acronyms.
5. [Elsevier research-ethics policy](https://www.elsevier.com/about/policies-and-standards/research-ethics) — human-data studies require an ethics statement; an exempt study must explain the exemption and participant/privacy protections.

No third-party journal-summary page was used as authority.

## 3. Executive verdict

The study is **in scope**, and the technical package now meets the confirmed requirements for abstract length, keywords, Highlights, editable source structure, separate artwork, a separate caption file, dataset-reference markers, and table geometry. The package is nevertheless **not upload-ready** because the formal multiseed analyses and final Word rebuild are still in progress and author/institution-supplied declarations remain incomplete.

Current highest-priority gaps are:

1. formal multiseed and independent zero-history analyses must be completed before the abstract, Results, Discussion, tables and figures are finalized;
2. the latest Markdown source contains 5,385 words through Conclusions and is newer than the current main-manuscript Word file, so final Word generation and page-level QA remain pending;
3. author/title-page metadata, ethics determination or exemption, CRediT roles, funding, acknowledgements, all-author competing-interest declaration, and the separate Elsevier declaration-tool Word file are incomplete;
4. BSPC's confirmed Option C policy is not finally satisfied because the manuscript still promises a future repository and persistent identifier;
5. the future software release has no final license, versioned software citation, repository URL or PID;
6. the optional graphical abstract still requires a documented AI-policy decision and should be omitted on the lowest-risk route.

## 4. Requirement-by-requirement comparison

| Topic | Exact current official requirement | Current project evidence | Status / action |
|---|---|---|---|
| **Scope** | BSPC covers biomedical-signal/data processing, ML/DL, wearable and remote monitoring, physiological modelling, robustness, personalization, and benchmark/dataset studies. The journal emphasizes practical/translational relevance. | Title, abstract, Introduction and cover letter frame recorded exercise HR as a wearable physiological signal, with forecasting, uncertainty, robustness, personalization, and frozen external validation. Clinical/device-accuracy claims are explicitly excluded. | **PASS.** Keep the digital-health monitoring/use-case framing prominent so the paper is not read as a generic fitness benchmark. |
| **Article type** | Full papers describe original technical work and a clinical/experimental study demonstrating concept/value. | Two public observational wearable datasets, model development, leakage-controlled experiments and frozen external validation are reported. | **PASS** as a full paper. |
| **Full-paper length** | A full paper “should normally be about 5,000 words.” This is guidance, not an explicit hard maximum. | The latest source validator reports **5,385** words from Introduction through Conclusions, including headings. | **NEAR TARGET / final check pending.** The source is about 8% above the normal target, not above a stated hard maximum. Recheck after final results are rewritten and avoid claiming a formal maximum that the guide does not state. |
| **Abstract** | Concise, factual, stand-alone, no more than 250 words; state purpose, principal results and major conclusions; avoid references and define uncommon abbreviations. | Validator reports **240 words**. It contains no citation, defines HR and bpm, and gives objective, methods, numerical results, limits and conclusion. | **PASS.** |
| **Keywords** | **1–7** English keywords; avoid multi-word phrases where practical and use only established abbreviations. | Six English keywords are present. No abbreviation is used. | **PASS.** Multi-word standard terms are an advisory issue only, not a breach. |
| **Peer-review anonymity** | BSPC uses **single-anonymized** review; reviewer identities are hidden from authors, but author identities are not hidden from reviewers. | The main Word file is not blinded and is designed to carry author details. | **PASS / no blinded manuscript required.** Remove the outdated “verify whether blinded manuscript is required” uncertainty from the submission checklist. |
| **Title page** | Include concise title, definitive author names/order, full affiliation postal addresses including country, corresponding author and current contact details; the submission checklist additionally asks for email, full postal address and phone number. | `title_page.md` and the Word title page are templates only. Author names, affiliations, corresponding-author address/email/phone are placeholders. | **GAP — blocking.** Supply the definitive author list before original submission; the guide warns that later authorship changes are generally not considered. Phone number is confirmed by the live checklist, not merely “if required.” |
| **Highlights** | Required at submission; separate editable file with “highlights” in its name; 3–5 bullets; each no more than 85 characters including spaces. Elsevier's Highlights page specifies a Word document and no jargon/acronyms. | `manuscript/BSPC_Highlights.docx` exists. Five bullets have validator counts **75, 74, 73, 66 and 69** characters and contain no acronym. Render/a11y QA passes. | **PASS.** |
| **Graphical abstract — requirement** | **Encouraged**, not required. If supplied: separate file, minimum 531 x 1328 px (h x w) or proportional equivalent, readable at 5 x 13 cm, preferred TIFF/EPS/PDF/MS Office. | Separate PDF/TIFF/PNG/SVG files exist and exceed the minimum dimensions. | **NOT REQUIRED.** The dimensions/formats pass, but the safest current route is to omit it because of AI provenance below. |
| **Graphical abstract — AI provenance** | The BSPC guide's embedded wording says AI/AI-assisted tools are not permitted for graphical abstracts. The linked June 2026 Elsevier policy says general-purpose generative AI tools must not be used and graphical abstracts should use dedicated scientific/professional illustration tools with tool disclosure. | `src/make_publication_figures.py` creates the graphical abstract; the Methods and AI declaration state that Codex assisted plotting-code development. Codex is a general-purpose generative-AI tool even though no image-generation model was used. | **CONDITIONAL / safest action: omit.** Do not upload the current graphical abstract. If the authors want one, recreate it in a documented human-controlled dedicated illustration workflow and retain licensing/provenance evidence, or obtain written editor confirmation. |
| **Research-data policy** | BSPC explicitly applies **Option C**: deposit research data in a relevant repository; cite and link it; if sharing is impossible, state why. A data-availability statement is **required at submission**. Research data includes software, code, models, algorithms, protocols and methods. | The manuscript has a Data availability section and identifies both source datasets; GoldenCheetah has an OSF DOI. Raw third-party records are not redistributed. However, the statement still contains `[repository and persistent release DOI to be supplied before submission]`, and Code availability is future tense. | **GAP — blocking unless a justified non-sharing statement replaces the promise.** Publish the allowed code, mappings, split definitions, aggregate figure data and audit summaries in a versioned repository; archive it and insert a stable link/PID. Complete the submission-system data statement. |
| **Dataset references** | Dataset references should include creator, title, repository, version where available, year and global persistent identifier. Add `[dataset]` immediately before the reference. | Endomondo and GoldenCheetah have separate `@dataset` BibTeX records; the compiled Word source contains exactly two `[dataset]` markers and gives the UCSD FitRec and OSF records. | **STRUCTURAL PASS / final rebuild check pending.** Reconfirm both markers and persistent links after the final Word rebuild. |
| **Software citation** | Code/software should be cited like other sources, ideally with creator, title, archive, date, PID, version and type. | The public-release manifest and integrity CSV exist locally, but no public version, license, repository URL, DOI/PID, `CITATION.cff`, or software reference exists. | **GAP.** After the author/institution license decision, mint a versioned public release and cite it as software in the manuscript/reference list. |
| **AI declaration — manuscript preparation** | Substantive AI use must be declared in a separate section before the references, naming the tool, purpose and human review/responsibility. | The manuscript uses the recommended heading immediately before References, names OpenAI Codex desktop and its purposes, and says no generative image model was used. The draft now correctly states that final author inspection and responsibility confirmation remain mandatory before submission. | **STRUCTURAL PASS / author confirmation and metadata pending.** Record the exact app/model/version identifier available at submission; “GPT-5-based” may be insufficient if a more exact identifier is available. |
| **AI-assisted research/code** | AI used in code development or research methods must be described reproducibly in Methods, including tool/model/version/developer where applicable; ordinary reproducibility standards still apply. | Methods §2.9 describes Codex-assisted code drafting/review, execution, audits, plotting-code and literature organization; distinguishes Codex from the forecasting models; and identifies the technical verification workflow. It explicitly leaves final author inspection mandatory. | **SUBSTANTIAL PASS / minor metadata GAP.** Add the exact Codex/app/model version if available and retain prompts/logs/review evidence privately in case the editor requests documentation. |
| **AI-assisted data visualizations** | The June 2026 global policy permits AI support only for figures directly derived from underlying data through reproducible methods, with tool/model/version/developer in Methods. | Figs. 2–4 and Supplementary Fig. 1 are deterministic Matplotlib outputs from stored result/source-data tables, and Methods §2.9 discloses the workflow. | **CONDITIONAL PASS.** This meets the updated global policy, but conflicts with the stricter stale-looking excerpt embedded in the live BSPC guide. Retain source tables/scripts and ask BSPC for confirmation if desired. Do not conceal the AI-assisted code role. |
| **AI-assisted explanatory image (Fig. 1)** | The June 2026 global policy permits AI-supported explanatory diagrams only with disclosure in both the figure caption and general AI statement. | Fig. 1 is deterministically rendered from versioned code. Its source caption states that no generative image model was used, and the general Methods/declaration text discloses the Codex-assisted workflow. | **DISCLOSURE ADDED / final author and policy check pending.** Keep the caption disclosure in the rebuilt files; because the BSPC guide excerpt is stricter, editor confirmation remains prudent. |
| **Editable main source** | Supply editable source files for the full submission; Word `.doc/.docx` or LaTeX `.tex`; PDF is not an acceptable source. Word must be single-column. | The current compiled Word file is editable and single-column, but the Markdown source has since changed and now cites 31 keys. | **FORMAT PASS / CONTENT REBUILD PENDING.** Rebuild from the final source, then repeat structural, accessibility and page-level QA. |
| **Figures/artwork sources** | Figures must be separately numbered, cited, captioned and uploaded as separate files. Vector drawings should be EPS/PDF; raster requirements depend on image type. | Four main and one supplementary figure exist separately as vector PDF and 600-dpi TIFF, with additional PNG/SVG; all are cited and have captions. | **PASS for artwork files.** Submit vector PDFs where possible and the compliant TIFFs as alternatives. |
| **Figure-caption file** | The guide expressly says, “Provide captions in a separate file.” | `manuscript/BSPC_Figure_Captions.docx` contains Fig. 1–4 and Supplementary Fig. 1 captions, contains no embedded image, and passed one-page render and accessibility QA. | **STRUCTURAL PASS / final rebuild check pending.** Rebuild once the final captions are frozen and repeat QA. |
| **Tables** | Tables must be editable, cited, numbered and captioned; avoid vertical rules and shading. | The compiled main table and 19 supplementary tables pass the automated three-line-table check: no vertical rules, internal horizontal grid or shading. | **STRUCTURAL PASS / final rebuild check pending.** Repeat the structure audit after final generation. |
| **Reference completeness/style** | At initial submission there is no strict formatting requirement if style is consistent and core metadata are present. The house style uses square-bracket numbers in order of appearance and recommends DOI links; journal names are abbreviated per LTWA. Every in-text citation must appear in the list and vice versa. | The latest Markdown source cites 31 keys; the BibTeX library contains 36 records, including two typed datasets. The existing Word reference list predates the latest source and must not be treated as final. | **SOURCE COHERENT / FINAL WORD PENDING.** Rebuild and rerun citation-key, numbering, DOI/URL and dataset-marker checks. LTWA abbreviation remains optional at initial submission if the style is consistent. |
| **CRediT** | Corresponding authors are required to report co-author contributions using CRediT roles. | Placeholder only. | **GAP — blocking.** Agree and insert roles for every author. |
| **Competing interests** | All authors must complete Elsevier's declarations tool. If nothing applies, select “I have nothing to declare.” Upload the resulting `.doc/.docx`; signatures are not required. | The manuscript contains a placeholder; no declarations-tool Word file exists. | **GAP — blocking.** Obtain all-author confirmation and generate/upload the official Word document. |
| **Funding** | Disclose all support and sponsor roles; if none, the guide recommends the standard no-specific-grant sentence. | Placeholder only. | **GAP — blocking.** |
| **Ethics / exempt human-data study** | Elsevier's policy requires an ethics statement for studies involving human participants or their data. If exempt/not requiring approval, explain why and confirm privacy/participant protections. | The current text says public deidentified secondary data, but leaves `[institutional ethics determination to be supplied]`. | **GAP — blocking.** Obtain an institutional determination/exemption and insert the committee/institution, determination and reference/date where applicable; do not self-declare exemption without institutional support. |
| **Acknowledgements** | Acknowledgements should be a separate section near the reference list and identify assistance. | Dataset maintainers are acknowledged, but an author-supplied placeholder remains. The main-Word builder now omits the source caption section because captions are supplied separately, leaving Acknowledgements followed by the mandatory AI declaration and References. | **STRUCTURE PASS / content GAP.** Authors must finalize the acknowledgement text. |
| **Submission declaration and author approval** | Submission implies originality, no simultaneous consideration, approval by all authors and responsible authorities. | The cover-letter draft contains these confirmations but still labels them for author confirmation. | **GAP — blocking at upload.** Obtain explicit confirmation from every author. |

## 5. The official AI-policy conflict

Two current official pages are not textually aligned:

- the BSPC Guide's embedded artwork paragraph says generative-AI or AI-assisted tools may not create or alter manuscript images, except when the tool is part of the research design/methods, and says AI assistance is not permitted for graphical abstracts;
- Elsevier's linked policy, explicitly updated June 2026, permits AI-supported explanatory images with caption/general disclosure and permits data visualizations faithfully derived from underlying data through reproducible methods with Methods disclosure. It bans general-purpose generative-AI tools for graphical abstracts and instead calls for dedicated scientific/professional illustration tools.

The linked June 2026 policy appears newer, but the journal-specific guide is the submission-facing authority and is worded more strictly. Therefore the current package should not claim unconditional compliance. The lowest-risk route is:

1. omit the optional graphical abstract;
2. retain deterministic scripts and source-data tables for every quantitative figure;
3. retain the added Fig. 1 caption disclosure and complete final direct author review;
4. retain the existing Methods and general AI disclosure, expanded with exact version identifiers;
5. if submitting any AI-assisted explanatory artwork, request written clarification from BSPC and archive the response.

## 6. Required actions before upload, in priority order

### Blocking author/institution actions

1. Finalize author order, affiliations, corresponding-author email/full address/phone, and all-author approval.
2. Obtain and insert the institutional ethics decision or documented exemption.
3. Complete CRediT, funding, acknowledgements and competing-interest wording.
4. Complete Elsevier's declarations tool and upload its generated Word file.
5. Select the software license/rights position; publish and archive the permitted release; insert stable repository/PID links and a software citation.

### Manuscript/package corrections

6. Complete the formal multiseed and independent zero-history analyses, then rewrite all result-bearing manuscript sections and source tables.
7. Recheck the final full-paper word count against the journal's approximately 5,000-word guidance; the current source is 5,385 words.
8. Replace future-tense repository placeholders with the actual release, or give a valid Option C non-sharing reason.
9. Rebuild the main, supplementary, caption and Highlights Word files from final sources; repeat structure, accessibility, citation, dataset-marker, number-linkage and page-render QA.
10. Add exact Codex/app/model version information where available and obtain final author confirmation of all AI-assisted work and Fig. 1 provenance.
11. Omit the current graphical abstract unless a policy-compliant replacement/provenance route is documented.

### Portal checks

12. Enter the mandatory data statement and dataset links in the submission system.
13. Inspect the system-generated single-anonymized review PDF page by page.
14. Confirm that the portal receives the editable main source, Highlights Word file, separate artwork, separate captions, supplementary file and declaration-tool Word file under the correct item types.

## 7. Bottom line

The live official Guide confirms that BSPC is a credible first-submission venue for this study. The abstract, keywords, Highlights, single-anonymized format, editable-source design, separate figure and caption files, dataset markers, and three-line table geometry pass at the structural level. The package is not yet ready to upload because formal experiments and the final Word/QA cycle are unfinished, while the repository release, author metadata, ethics determination, CRediT, funding, competing interests, licence, persistent identifier and optional graphical-abstract decision remain unresolved.

The previous inability to verify BSPC-specific requirements is now resolved; future readiness checks should use this report and the live Guide rather than treating those fields as unknown.
