# Physiological Measurement submission-file index

**Prepared:** 23 July 2026  
**Current gate:** technical files validated; author/institution fields and persistent public release remain open

This index maps each local artifact to the likely Physiological Measurement upload item. The live submission portal remains controlling.

| Upload item | Preferred local file | Status | Final action |
|---|---|---|---|
| Main manuscript | `manuscript/BSPC_main_manuscript_draft.docx` | Technically validated; 18 pages | Insert authors, affiliations, corresponding author, ethics, CRediT, funding, interests, acknowledgements and release DOI; rebuild and rerun QA after those author edits. |
| Supplementary material | `manuscript/BSPC_supplementary_material.docx` | Validated; 19 pages | Upload as one editable supplementary file unless the portal requests PDF. |
| Highlights | `manuscript/BSPC_Highlights.docx` | Validated; 5 bullets, each <=85 characters | Upload using the `Highlights` item type. |
| Figure captions | `manuscript/BSPC_Figure_Captions.docx` | Validated; captions only; 2 pages | Upload separately if the portal provides a caption/legend item; otherwise retain as requested editorial file. |
| Figure 1 | `figures/Figure_1_study_design.pdf` | Vector master passes figure QA | Upload PDF and confirm fonts are embedded; use TIFF only if the portal rejects PDF, re-exporting line art at 1,000 dpi if required. |
| Figure 2 | `figures/Figure_2_primary_performance.pdf` | Vector master passes figure QA | Same. |
| Figure 3 | `figures/Figure_3_uncertainty_calibration.pdf` | Vector master passes figure QA | Same. |
| Figure 4 | `figures/Figure_4_sport_shift.pdf` | Vector master passes figure QA | Same. |
| Supplementary Figure 1 | Embedded in supplementary DOCX; separate master `figures/Supplementary_Figure_1_ablation_sensitivity.pdf` | Passes figure QA | Upload separately only if the portal requires supplementary artwork as individual files; inspect the supplement with Track Changes off. |
| Title page | Information template: `manuscript/title_page.md` | Author fields open | Main manuscript already contains a title page; create a separate editable title-page file only if the portal requests one. |
| Cover letter | `manuscript/cover_letter.md` | Scientific argument updated; author confirmations open | Paste into portal or convert to signed/editable file after corresponding-author details are supplied. |
| Declaration of interests | Not yet generated | Blocking | Obtain all-author confirmation, generate the official Elsevier Declaration Tool file, and upload the original `.docx` without conversion. |
| Data availability statement | Present in main manuscript | Repository DOI open | Enter the matching statement in the portal and link both source datasets and final code archive. |
| Code/software record | Release manifest exists locally | Licence, public URL and DOI open | Tag the author-approved repository, archive it, add `CITATION.cff`, and cite the release. |
| Graphical abstract | `figures/Graphical_Abstract.*` retained internally | Optional; default omit | Do not upload unless the author confirms a policy-compliant human-controlled workflow. |

## Upload exclusions

Do not upload raw Endomondo or GoldenCheetah files, extracted athlete archives, row-level origins or predictions, user/session identifiers, exact timestamps, coordinates, model arrays, checkpoints, render caches, or the local integrity inventory unless the editor specifically requests an approved confidential transfer.

## Final portal checks

1. Match every portal author and affiliation to the final title page exactly.
2. Select the full-paper/original-research article type shown by the live portal.
3. Enter funding, competing-interest, ethics, CRediT, data and AI information consistently with the manuscript.
4. Inspect the portal-generated PDF page by page before approval.
5. Regenerate the public-release integrity manifest from the exact tagged commit and verify the archived DOI from outside the author account.
