# Physiological Measurement submission-readiness checklist

**Controlling profile:** `protocol/PMEA_SUBMISSION_PROFILE.md`  
**Status date:** 29 July 2026  
**Meaning of complete:** a clean, visually checked complete-document PDF and its editable source are ready for upload, all scientific claims are linked to verified outputs, and every author/institution field is final.

## Scientific and reporting gates

- [x] Research question is within the journal's physiological prediction and sports/wellness scope.
- [x] Core claim is bounded to recorded wearable HR and boundary-dependent reliability.
- [x] 5-min primary horizon and secondary 1-/3-min horizons are explicit.
- [x] Users and sessions are split before windows; histories are past-only.
- [x] Frozen GoldenCheetah evaluation uses no adaptation or recalibration.
- [x] Main and zero-history-trained models are compared across declared seeds.
- [x] Point effects use session-then-user aggregation and user bootstrap intervals.
- [x] Empirical interval limitations under clustered time series are explicit.
- [x] Structured Objective/Approach/Main results/Significance abstract is present.
- [x] SAGER-oriented sex/gender statement and imbalance justification are present.
- [x] PMEA-facing joint-shift outcomes are limited to cells with at least 30 users.
- [x] Clinical, device-accuracy, causal-physiology and prescription claims are excluded.
- [x] PMEA-specific figure 4 masks outcome estimates for the three cells with 18--20 users.
- [x] PMEA supplementary tables omit those low-support outcome rows while preserving support accounting.
- [x] PMEA supplementary figure removes the underpowered recorded-gender comparison.
- [x] Final claim--evidence and reported-number validators pass after journal adaptation.

## Format and package gates

- [x] Main text is below the 8000-word research-paper guidance.
- [x] Automated abstract count confirms no more than 250 words.
- [x] All in-text citations use Harvard author-year style.
- [x] Reference list is alphabetized, includes article titles, and resolves every cited key.
- [x] Complete review DOCX contains figures and tables embedded near the relevant text.
- [x] Complete review PDF has been generated from the latest DOCX.
- [x] Every PDF page has been visually inspected for clipping, overlap, missing glyphs, broken tables, and orphaned captions.
- [x] DOCX accessibility audit reports no high-severity issue; all scientific figures have meaningful alt text.
- [x] Supplementary material is built and visually checked as a separate upload item.
- [x] Cover letter is retargeted to Physiological Measurement and explains the 30-person reporting overlay.

## Author and institutional blockers

- [x] Author names, order, affiliations and emails supplied; PANG KEREN confirmed as corresponding author.
- [ ] Single- versus double-anonymous review route selected.
- [x] Author contributions, manuscript originality, final approval and agreement to submit confirmed by both authors.
- [x] Author-approved ethics statement finalized: formal review was not sought because this was secondary analysis of pre-existing, publicly available, pseudonymized datasets with no recruitment, intervention, interaction or access to directly identifying information.
- [x] Funding statement finalized as no specific grant.
- [x] Conflict-of-interest statement finalized as no competing interests.
- [x] Acknowledgements finalized; the outstanding placeholder was removed.
- [x] Generative-AI disclosure finalized as OpenAI Codex desktop, model `gpt-5.6-sol`, used only for English-language polishing and Chinese–English translation.
- [x] Both authors accept responsibility for the manuscript and submission; research code was developed by the authors with additional paid human technical assistance and was verified by the authors.

## Data and reproducibility blockers

- [x] Public code/derived-data release scope approved and released under the MIT License.
- [x] Versioned repository `v0.30.0` published and archived at https://doi.org/10.5281/zenodo.21649896.
- [x] Data-availability and code-availability placeholders replaced with stable GitHub and Zenodo links.
- [x] Authors confirmed that use of Endomondo HR/FitRec and GoldenCheetah OpenData follows the applicable source terms; raw individual-level records are not redistributed.
- [x] The public-release integrity and privacy audit was generated for the archived package and deposited with the release materials.
- [x] Final commands, environment lock, split hashes, model hashes and aggregate source tables reproduce the submission numbers; the PMEA validator reports READY.

## Upload-time live checks

- [ ] Reopen the live journal page and ScholarOne form immediately before upload.
- [ ] Confirm current article type, abstract fields, keyword entry, declarations and supplementary item names.
- [ ] Confirm author choice of peer-review identity and upload the matching PDF.
- [ ] Upload the complete review PDF and supplementary material; retain editable source for revision/production.
- [ ] Verify every submission-system metadata field against the final manuscript before approving submission.
