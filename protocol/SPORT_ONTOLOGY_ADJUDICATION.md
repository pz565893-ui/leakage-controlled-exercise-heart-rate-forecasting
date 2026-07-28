# Sport ontology adjudication protocol

**Ontology version:** `0.2.0-adjudicated-rules`  
**Status:** locked outcome-blind rule mapping for the reported analyses; unresolved labels remain `other_unknown` and are excluded from sport-shift claims.

## Purpose

Raw activity labels are not treated as statistically independent sport types. They are mapped to broad exercise families that have a defensible shared movement context and enough support for distribution-shift evaluation. Mapping is performed before split generation and without examining model errors or test outcomes.

## Full raw-label census

| Source | Activity records | Users with usable metadata | Raw labels |
|---|---:|---:|---:|
| Endomondo | 253,020 | 1,104 | 49 |
| GoldenCheetah metadata | 53,146 | 148 | 347 |

Endomondo has strong full-census support for outdoor cycling (121,956 records; 860 users), running (119,169; 867), walking/hiking (4,624; 278), indoor/virtual cycling (3,014; 195), strength/cross-training (2,050; 245), and skiing (1,093; 89). These counts precede final heart-rate and forecast-origin eligibility filtering.

GoldenCheetah metadata show support for outdoor cycling (35,849 records; 145 users), running (5,621; 58), and indoor/virtual cycling (2,645; 44). Swimming has 1,113 records across 26 users but only 235 metadata records with a heart-rate channel indicator.

## GoldenCheetah session linkage

CSV filenames encode local timestamps while ride metadata use UTC timestamps. Sessions are linked only within the same user by exact timestamp after testing UTC offsets from −14:00 to +14:00 in 15-minute increments. Ambiguous and duplicate matches are not resolved heuristically.

| Link status | CSV files |
|---|---:|
| Unique linked match | 50,002 |
| Invalid or missing user metadata | 1,335 |
| Ambiguous timestamp match | 111 |
| Unmatched timestamp | 14 |
| Duplicate metadata match | 8 |

Thus, 97.15% of all 51,470 activity CSV files have a unique metadata linkage. The remaining records receive explicit exclusion codes. The linked manifest contains one unique row per CSV and no linked row lacks its timestamp or sport-family fields.

## Mapping rules

1. Normalize Unicode, case, and accents while retaining the original label.
2. Apply high-specificity indoor/virtual-cycling terms before generic cycling terms.
3. Map unambiguous running, cycling, walking/hiking, swimming, skiing, and strength/cross-training terms to the corresponding family.
4. Keep missing, generic, mixed, or semantically ambiguous labels as `other_unknown`.
5. Preserve the mapping rule, source label, record count, user count, and semantic-audit status in the versioned CSV.
6. Never infer a sport from heart-rate targets, model errors, or post-origin sensor values.

Free-text labels such as “Morning Ride” may be mapped only when they contain an unambiguous modality term. Generic labels such as “Training” and “Track” remain unresolved. Missing labels are not treated as a novel sport family.

## Semantic-audit queue and lock interpretation

The current CSV uses `rule_mapped_locked` for labels assigned by the deterministic rules and `retained_unknown_locked` for unresolved labels fixed to `other_unknown`. The summary JSON records the same analysis lock and retains a `manual_review_queue` only as a queue for possible semantic work in a future ontology version. That queue does not make the reported mapping mutable. Earlier generated copies used the legacy values `rule_mapped_pending_review` and `manual_review_required`; those values likewise denoted semantic-audit workflow state rather than permission to alter the reported analysis. Any future reinterpretation requires a new ontology version and new downstream analysis rather than an in-place edit.

The summary also records a SHA-256 fingerprint over the ordered `(source, raw_label, provisional_family)` mapping. Generated timestamps, counts, and review-status metadata are deliberately excluded from that fingerprint, so a status-only migration can be verified without exposing raw labels.

Version 0.2.0 leaves 203 labels unresolved. Most GoldenCheetah `other_unknown` records arise from a missing sport field (6,691 records), not from a new activity type. High-frequency unresolved Endomondo labels now include skating, kayaking, rowing, and team or racquet sports. Core stability and circuit training map to strength/cross-training; orienteering maps to running. Generic labels such as `Workout` remain unresolved. Adjudication was based on label meaning and prespecified family definitions, not downstream performance.

The v0.2 review also removed unsafe substring behavior: `TT` is recognized only as a complete label, so words such as Italian `Corsetta` no longer match cycling accidentally. Generic `road` and `workout` substrings no longer force a sport assignment.

Every override will be added to a versioned table with:

- raw and normalized labels;
- source dataset;
- final family;
- action (`map`, `retain_unknown`, or `exclude`);
- short rationale;
- reviewer and date.

## Support thresholds

After session-level signal-quality filtering:

- an Endomondo family enters leave-one-sport-family-out evaluation only with at least 50 users and 1,000 eligible sessions;
- a GoldenCheetah family enters primary external validation only with at least 20 users and 500 uniquely linked, heart-rate-eligible sessions;
- `other_unknown` never serves as an unseen-sport test family;
- sparse families may be described but are not pooled into a synthetic “novel sport” result.

These thresholds make outdoor cycling, running, and indoor/virtual cycling the current external-validation candidates. Final inclusion is determined from row-level heart-rate eligibility, not metadata masks alone.

After exact-duplicate control, all three external candidates remain supported: outdoor cycling has 25,964 provisionally eligible sessions from 143 users, running has 3,835 from 50 users, and indoor/virtual cycling has 2,279 from 41 users. In Endomondo, outdoor cycling, running, walking/hiking, indoor/virtual cycling, and strength/cross-training exceed the internal leave-one-family-out threshold. Skiing has 959 eligible sessions and therefore remains below the prespecified 1,000-session threshold.

## Reproducible artifacts

- `configs/sport_ontology_v0_2_0.csv`
- `outputs/audit/sport_ontology_v0_2_0_summary.json`
- `outputs/manifests/goldencheetah_session_linkage_v0_2_0.csv`
- `outputs/audit/goldencheetah_session_linkage_v0_2_0_summary.json`
- `src/build_sport_ontology.py`
- `src/link_goldencheetah_sessions.py`
- `outputs/manifests/endomondo_session_quality_v0_2_0.csv`
- `outputs/manifests/goldencheetah_session_quality_v0_2_0.csv`
- `src/build_session_manifests.py`
- `src/build_golden_manifest_fast.py`
