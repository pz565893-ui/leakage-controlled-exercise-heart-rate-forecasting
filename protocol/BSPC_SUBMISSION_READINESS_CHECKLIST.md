# Physiological Measurement submission-readiness checklist

**Target:** *Physiological Measurement* (PM), full-length original research article  
**Package review date:** 23 July 2026  
**Status:** formal experiments and result integration are complete; final Word/QA artifacts are present; author- and institution-supplied items remain blocking

This checklist separates checks already evidenced by the project from decisions that cannot be inferred by the analysis code. The live PM Guide for Authors was inspected and is summarized in `BSPC_LIVE_REQUIREMENTS_AUDIT_2026-07-22.md`; the live upload form remains controlling and must be confirmed immediately before submission.

## 1. Scientific and technical package

- [x] Title and scope frame the work as biomedical-signal forecasting, personalization, uncertainty, and robustness under distribution shift.
- [x] Past-only user, session, sport, history, calibration, and cross-source-evaluation boundaries are explicitly defined.
- [x] Complete and aggregate the formal v0.22.0 multiseed strict-temporal, unseen-user, five held-sport-family, joint user--sport, and frozen cross-source evaluations; all 37 declared jobs passed.
- [x] Complete the separate zero-history-trained v0.23.0 analysis, including frozen cross-source inference without adaptation or recalibration; all 10 declared jobs passed.
- [x] Complete the frozen-prediction v0.24.0 balanced-calibration and sport-composition analyses, v0.25.0 seed-averaged paired-user comparisons, v0.26.0 persistence-conformal baseline, v0.27.0 matched-sport sensitivity, and v0.28.0 deliberately leaky negative-control evidence (invalid-control interpretation only).
- [x] Replace primary single-seed result text with per-seed median/range summaries and seed-matched, user-level bootstrap comparisons where specified.
- [x] External differences are described as data-source-associated rather than caused by a device or platform.
- [x] Novelty claims are bounded to the inspected corpus and comparator table rather than stated as a global first.
- [x] Clinical, diagnostic, device-accuracy, and physiological-causality claims are explicitly excluded.
- [x] Integrate the audited v0.22.0--v0.28.0 results into the manuscript, supplement, figures, and source-data tables; the reported-number validator passes 473/473 checks, including 47 provenance-path checks.
- [x] Complete final citation, figure-source, executable-test, document-structure, accessibility, and page-render QA gates after result integration.
- [x] Harden the privacy-conservative release policy: block linkable raw-label ontology content, subgroups below 10 users, the internal graphical abstract, participant-level bootstrap input, and checkpoints; include the non-identifying raw-source integrity audit.
- [x] Regenerate and verify the privacy-conservative manifest after the final document rebuild and validation cycle.
- [ ] Regenerate and verify that manifest against the final author-approved tagged release after the licence, release-rights, and DOI decisions are complete.

## 2. Submission files prepared

- [x] Editable main-manuscript Word build pipeline and draft file.
- [x] Editable supplementary-material Word build pipeline and draft file.
- [x] Separate five-item Highlights Word file; each item is no more than 85 characters.
- [x] Separate editable figure-caption Word file containing all main and supplementary captions.
- [x] Title-page template.
- [x] Cover-letter draft.
- [x] Chinese one-time author/institution input form.
- [x] PM upload-file index.
- [x] Four main figures and one supplementary figure in vector PDF and high-resolution TIFF.
- [x] Numbered reference list backed by the project BibTeX library.
- [x] Data- and code-availability statements.
- [x] Methods disclosure for AI-assisted code, audit, plotting-code, literature-organization, and writing support; Fig. 1 caption identifies the tool, available model information, versions, and use.
- [x] Elsevier-style generative-AI declaration immediately before the references.
- [x] Rebuild the main manuscript, supplement, captions, and Highlights from the integrated final sources and complete structure, accessibility, and page-render QA.

## 3. Author or institution must complete before upload

- [ ] Insert final author spelling and order exactly as entered in the portal; insert complete, verifiable affiliations and postal addresses, and map every author correctly.
- [ ] Insert corresponding-author postal address and valid email; add telephone, ORCID, and present-address notes if requested by the live portal.
- [ ] Obtain the institutional ethics determination. For approval, record committee/institution, date, and number; for exemption/no-review-required, record the institutional basis and privacy, pseudonymization, non-reidentification, participant-rights, and consent determination. Do not self-declare exemption.
- [ ] Confirm funding, grant numbers, funded authors, and the funder's role in design, data collection, analysis/interpretation, writing, and submission; otherwise use the journal-compatible no-specific-grant statement.
- [ ] Obtain an all-author competing-interest declaration; have the corresponding author complete the Elsevier Declaration Tool and upload its original `.docx` with wording consistent with the manuscript.
- [ ] Agree and insert CRediT roles for every author.
- [ ] Confirm acknowledgements, originality, no duplicate publication or simultaneous submission, responsible-institution approval, and all-author approval of the submitted version.
- [ ] Select scoped licences separately for author-created software and author-created tables/figures/documents; explicitly exclude third-party source data and keep checkpoints private unless institutionally approved.
- [ ] Create a versioned public repository release, archive it, mint a persistent DOI, and replace repository placeholders in the manuscript. The DOI is this project's higher reproducibility standard, not the only implementation allowed by the journal policy.
- [ ] Confirm the AI disclosure's tool name, purpose, human oversight, available model/version information and developer; use a more precise product identifier only if the interface provides one.
- [ ] Confirm the AI service's privacy, retention, training-rights, and output-publication terms, and privately retain tool/model, prompt/output, and human-review evidence for editorial queries.

## 4. Graphical abstract decision

The local graphical-abstract artifact is a deterministic Python/Matplotlib rendering, but AI assistance was used in plotting-code development. Under current Elsevier policy statements, general-purpose generative AI is restricted for graphical abstracts:

- [ ] **Default safe route:** do not upload the graphical abstract if it is optional in the live PM form; or
- [ ] recreate and finalize it with a licensed dedicated scientific/professional illustration tool and documented direct human authorship; or
- [ ] obtain written confirmation from the journal that the documented workflow is acceptable.

The local graphical-abstract files should be retained as internal design material even if they are omitted from submission.

## 5. Live PM checks at upload

- [x] Verify article type and current length guidance: full paper, typically about 5,000 words; the current main text contains 5,938 words including headings.
- [x] Verify abstract and keyword rules: the abstract remains below 250 words, is self-contained, reports purpose, main results and conclusion, contains no citations, defines uncommon abbreviations, and the manuscript has 6 English keywords.
- [x] Verify review mode and source requirements from the live PM guide.
- [ ] Verify the current research-data policy and complete the submission-system data statement.
- [x] Verify the current initial reference policy and dataset marker; both dataset records are typed and compiled with `[dataset]`.
- [x] Verify accepted editable source-file requirements and mandatory separate Highlights.
- [x] Use vector PDF as the primary figure upload format and retain TIFF only as a fallback where required.
- [x] Inspect the supplementary Word file for Track Changes, privacy, and standalone explanatory text.
- [x] Verify that the graphical abstract is optional; default safe route is omission unless policy-compliant workflow exists.
- [ ] Reconfirm the live portal fields and complete the submission-system data statement at upload.
- [ ] Inspect the portal-generated PDF page by page before approving submission.

## 6. Release gate

The technical manuscript package is `PASS / AUTHOR_INPUT_REQUIRED`: its experiment, result-integration, 169-test suite, Word-build, structure, accessibility, render-QA, 473-check reported-number audit, and privacy-conservative working-tree manifest gates are complete. Upload readiness still requires completion of all author- and institution-supplied items in Section 3, the live-portal checks in Section 5, and a documented graphical-abstract decision. Public archiving additionally requires an author-approved licence and release-rights decision, a tagged release, a persistent DOI, and a matching verified manifest regenerated from that exact tag.
