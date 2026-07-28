# Publication figure quality-control record

**QA date:** 28 July 2026  
**Target:** *Physiological Measurement*  
**Backend:** Python/Matplotlib only

## Deliverables checked

| Figure | Visual inspection | Vector output | 600-dpi TIFF | Source data |
|---|---:|---:|---:|---:|
| Figure 1 - study design | Pass | PDF + SVG | 4389 x 2709 | Protocol-derived diagram |
| Figure 2 - primary performance | Pass | PDF + SVG | 4389 x 3560 | Pass |
| Figure 3 - sport shift | Pass | PDF + SVG | 4388 x 3069 | Pass |
| Figure 4 - uncertainty calibration | Pass | PDF + SVG | 4389 x 3369 | Pass |
| Supplementary Figure 1 - ablations and sensitivity | Pass | PDF + SVG | 4389 x 3489 | Pass |
| Graphical abstract | Pass as an internal optional artifact | PDF + SVG | 6295 x 2573 | Pass |

## Evidence and checks

- All PNG previews were inspected at original resolution for clipping, overlap, unreadable legends, ambiguous encodings, and panel-label placement.
- Figure 2 reports five-seed history-informed, history-masked, and zero-history-trained summaries, three-seed GRU results, deterministic baselines, and seed-averaged paired-user-bootstrap strategy contrasts.
- Figure 3 reports three-seed held-sport medians and marks joint intersections below 30 users as not reported.
- Figure 4 panels a-d report five-seed interval summaries; panels e-f are explicitly labelled as frozen-reference-seed analyses.
- Supplementary Figure 1 uses canonical history-informed, history-masked, and zero-history-trained labels; panel c reports the five-seed zero-history-trained strategy contrast. The graphical abstract uses the final v0.22/v0.23 values.
- Every TIFF reports 600 x 600 dpi. SVG files retain editable text. The graphical-abstract PNG is 2308 x 943 pixels.
- Colour is not the only encoding in principal comparison plots: marker shapes, line styles, direct labels, or numerical annotations provide redundant cues.
- Twelve CSV files in `figures/source_data/` contain the plotted aggregate values; no participant-level source rows are included. Figure 3's paired user-bootstrap confidence intervals remain in Supplementary Table S5c and are not implied by the heatmap annotations.
- No generative image model was used. The graphical abstract remains optional and is subject to the documented Elsevier AI-policy decision.

## Verdict

PASS for manuscript assembly. Use vector PDF/SVG where accepted and the 600-dpi TIFF files when raster artwork is required.
