# Cross-source normalized-signal duplicate audit (v0.20.0)

## Purpose and scope

This audit tests whether a processed Endomondo session used by the modeling pipeline may also occur in the GoldenCheetah external-validation source. The audit unit is a session in `session_series_v0_4_0.sqlite`; consequently, it covers sessions that survived the leakage-controlled preprocessing and had at least one accepted forecast origin. It does not claim to audit raw sessions that were excluded before modeling.

## Confirmed exact normalized-signal matches

Each session is represented on the existing 10-second grid. A SHA-256 fingerprint frames and hashes the grid spacing, number of bins, and the decompressed little-endian float32 values plus uint8 observation masks for heart rate, altitude, and speed. Dataset labels, identifiers, sport labels, partitions, absolute timestamps, and `grid_start_bin` are excluded. The fingerprint is therefore invariant to user/session naming and absolute clock translation, but it remains sensitive to any normalized signal or mask difference. A cross-source equality of this full fingerprint is classified as a confirmed exact normalized-signal match.

For added sensitivity, the audit also hashes the valid-boundary-trimmed heart-rate values and mask alone. Equality of this subset is reported only as an HR candidate, not as confirmation that the sessions are duplicates.

## Deterministic near-duplicate candidate screen

The near screen is restricted to informative records: at least 30 minutes between the first and last valid HR bins, at least 80% HR coverage, no internal missing run longer than 60 seconds, and HR standard deviation of at least 5 bpm. HR is linearly interpolated only for constructing a fixed 48-point normalized-duration profile. Profiles are quantized in 4-bpm bins under two bin offsets, with three endpoint crop variants (none, one 10-second bin from the left, or one from the right). Equality joins are exhaustive for these declared deterministic signatures. Candidate pairs must also differ in valid HR span by no more than 5%.

To broaden candidate retrieval beyond identical quantized profiles, a second screen uses 12 deterministic random-hyperplane locality-sensitive-hash tables with 24 bits per table and seed 20260722. The hyperplanes operate on within-session standardized 48-point HR shapes. A bucket collision is retained only when valid-HR spans differ by at most 5%, 48-point HR correlation is at least 0.98, MAE is at most 4 bpm, and the 95th percentile absolute error is at most 8 bpm. The projection matrix hash is recorded in the audit JSON.

Every retained signature or LSH candidate is re-evaluated on 256 continuous-valued points. It passes the HR screen only if MAE is at most 2 bpm, the 95th percentile absolute error is at most 4 bpm, and Pearson correlation is at least 0.99. Speed agreement (MAE at most 1.5 km/h and correlation at least 0.98) and centered-altitude agreement (MAE at most 5 m and correlation at least 0.98) are separately tested when both sources have at least 70% coverage. An HR candidate with either auxiliary agreement is labeled `near_candidate_hr_plus_auxiliary`; no near candidate is promoted to a confirmed duplicate without equality of the full exact fingerprint.

## False-positive and false-negative controls

False positives are limited by minimum duration, coverage and HR-variability rules; duration agreement; continuous-valued verification; and separate auxiliary evidence. The output retains continuously verified candidate pairs, including those failing the final threshold, for auditability. The screen can nevertheless miss sessions with substantial cropping, long gaps, clock drift, different device smoothing, or differences outside the declared crop and quantization variants; LSH also has a non-zero false-negative probability. Therefore, zero verified candidates lowers contamination concern under this specification but does not prove that every possible cross-platform duplicate is absent.

## Reproducible artifacts

The audit script records SHA-256 hashes of the SQLite cache and both split manifests, SQLite integrity and manifest-coverage assertions, all thresholds, scope counts, runtime, and result paths in the JSON audit file. Exact and near-candidate pair tables retain deterministic domain-separated SHA-256 pseudonyms for sessions and users, plus sport families and split labels; raw IDs and GoldenCheetah file names are not serialized. If a confirmed match requires exclusion, its pseudonym can be resolved locally by recomputing the same mapping against the private manifest, without publishing the source key.
