# Figure contract

## Figure 1 — leakage-controlled study design

- Core conclusion: the study evaluates causal 5-minute-context heart-rate forecasts under progressively harder, explicitly isolated distribution shifts.
- Archetype: schematic-led composite.
- Target/output: Physiological Measurement double-column, 183 mm wide; Python/matplotlib; editable SVG and PDF plus 600-dpi TIFF and PNG preview.
- Panel map: (a) causal context and 1/3/5-minute horizons; (b) current-session and completed-prior-session encoders; (c) temporal, unseen-user, held-sport, joint-shift, and frozen cross-source protocols.
- Evidence hierarchy: protocol logic is primary; cohort/split labels are supporting.
- Statistics: no inferential statistics; exact task windows and isolation rules shown.
- Reviewer risk: a diagram must not imply that future sessions or GoldenCheetah data enter training.

## Figure 2 — primary point-forecast evidence

- Core conclusion: simple signal baselines are clearly worse, whereas differences among the main model, its zero-history-trained counterpart, and neural comparators are small and boundary dependent; cross-source errors are higher.
- Archetype: quantitative grid with a dominant three-regime comparison row and a paired-effect forest panel.
- Target/output: Physiological Measurement double-column, 183 mm wide; Python/matplotlib; SVG/PDF/TIFF/PNG.
- Panel map: (a) strict temporal MAE; (b) unseen-user MAE; (c) frozen cross-source MAE; (d) five-seed strategy contrasts with paired user-bootstrap intervals.
- Evidence hierarchy: temporal and unseen-user comparisons are primary; the cross-source panel tests transport with matched history-masked information.
- Statistics: hierarchical MAE (origins within sessions, sessions within users, equal-user mean); paired per-user differences averaged across matched seeds before 10,000-replicate user bootstrap 95% CIs.
- Reviewer risk: avoid implying that the small MAE difference versus every neural baseline is universally significant.

## Figure 3 — sport and joint distribution shifts

- Core conclusion: leave-one-sport-out generalization is heterogeneous across sport families and becomes least certain where independent user support is sparse.
- Archetype: asymmetric quantitative grid with MAE heatmaps and a support panel.
- Target/output: Physiological Measurement double-column, 183 mm wide; Python/matplotlib; SVG/PDF/TIFF/PNG.
- Panel map: (a) same-user unseen-sport MAE heatmap; (b) joint unseen-user-and-sport MAE heatmap; (c) gain/loss versus EWMA; (d) users and sessions per family/regime.
- Evidence hierarchy: family-specific error is primary; aligned baseline deltas and support counts qualify interpretation.
- Statistics: independently trained leave-one-family-out models over three declared seeds; equal-user hierarchical MAE; no pooled significance claim for under-supported categories. Paired user-bootstrap model–EWMA intervals are reported in Supplementary Table S5c rather than encoded in the heatmap.
- Reviewer risk: joint-shift cells with fewer than 30 users must be marked not reported in the journal-facing figure.

## Figure 4 — uncertainty under distribution shift

- Core conclusion: empirical post-CQR intervals are near nominal internally, whereas matched history-masked cross-source forecasts show greater longer-horizon undercoverage and wider intervals.
- Archetype: quantitative grid.
- Target/output: Physiological Measurement double-column, 183 mm wide; Python/matplotlib; SVG/PDF/TIFF/PNG.
- Panel map: (a–c) calibrated coverage at 1/3/5 minutes for temporal, unseen-user, and frozen cross-source regimes; (d) 90% interval width; (e) weighted interval score; (f) width–absolute-error association.
- Evidence hierarchy: coverage is primary; width, WIS, and Spearman association diagnose sharpness and informativeness.
- Statistics: nominal 50/80/90% intervals; user-session hierarchical averages; mean within-user Spearman correlations.
- Reviewer risk: calibration thresholds were learned on Endomondo only; cross-source coverage is evaluation, not recalibration, and no user-level finite-sample guarantee is claimed.

## Supplementary Figure 1 — ablation and sensitivity

- Core conclusion: point estimates generally favour completed history relative to a zero-history-trained strategy, while extra speed/altitude channels and dense-origin evaluation change little.
- Archetype: forest-plot quantitative grid.
- Target/output: Physiological Measurement supplement double-column; Python/matplotlib; SVG/PDF/TIFF/PNG.
- Panel map: (a) multimodal versus HR-only paired effects under canonical history-informed/history-masked modes; (b) 60-s versus 300-s evaluation-origin sensitivity; (c) history-informed versus zero-history-trained effects.
- Statistics: 10,000-replicate user bootstrap CIs.
- Reviewer risk: the recorded-gender outcome panel is omitted because the groups do not meet the journal-facing 30-user reporting threshold.

## Integrity and source-data notes

- All figures are generated from audited CSV/JSON artifacts by one Python script.
- No raster scientific images, crops, local contrast changes, or pseudo-colour transformations are used.
- SVG text remains editable; PDF fonts use TrueType embedding; TIFF is exported at 600 dpi.
- Method colors remain consistent: history-informed model (dark blue), zero-history (mid blue), reference baselines (grey), external shift (muted red), and uncertainty/support cues (teal/gold).
