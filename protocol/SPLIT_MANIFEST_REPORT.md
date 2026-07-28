# Leakage-controlled split manifest report

**Split version:** 0.2.0  
**Seed:** 20260722  
**Construction order:** quality screen → exact-signal duplicate control → session/user assignment → forecast-window generation.

## Duplicate control

Endomondo contains no byte-identical full records, but signal fingerprints based on timestamp, heart rate, speed, altitude, latitude, and longitude identify 5,741 exact-signal duplicate groups involving 12,885 records. IDs, URLs, users, gender, and sport labels are excluded from this fingerprint. Of these groups, 295 span multiple users and 382 additional groups have conflicting v0.2 sport families.

The prespecified resolution is conservative:

- exclude every cross-user exact-signal group;
- exclude every remaining group with conflicting sport families;
- for a within-user, family-consistent group, retain one deterministic canonical record and exclude the redundant copies.

This excludes 7,821 Endomondo records and retains 5,064 canonical representatives from duplicate groups. After signal quality, ontology v0.2, and duplicate control, 201,823 Endomondo sessions remain provisionally analysis-eligible.

GoldenCheetah contains 102 byte-identical groups involving 211 CSV files. Five groups span users. The same policy excludes 114 records and retains 97 canonical representatives. The post-control analysis-eligible total is 32,587, of which 32,078 belong to the three primary external sport families.

## Endomondo unseen-user partition

Users are assigned by a seeded SHA-256 hash; no session or outcome value contributes to assignment.

| Partition | Users | Sessions |
|---|---:|---:|
| Train | 759 | 140,165 |
| Validation | 129 | 26,023 |
| Calibration | 97 | 16,199 |
| Test | 105 | 19,436 |

Automated pairwise checks find zero user overlap among all four partitions.

## Endomondo within-user temporal partition

Users with at least ten analysis-eligible sessions are sorted by session start time and divided approximately 70%/10%/10%/10% into train, validation, calibration, and test. A deterministic record key resolves ties. Users with fewer than ten eligible sessions are excluded from this protocol but may remain eligible for other protocols.

| Partition | Sessions |
|---|---:|
| Train | 140,510 |
| Validation | 20,203 |
| Calibration | 20,095 |
| Test | 20,578 |
| Insufficient user history | 437 |

Automated checks find zero chronological-order violations between adjacent partitions for every included user.

## Sport and joint-shift support

After duplicate control, the Endomondo families meeting at least 50 users and 1,000 sessions are:

- outdoor cycling: 109,748 sessions, 842 users;
- running: 83,929 sessions, 849 users;
- walking/hiking: 3,174 sessions, 243 users;
- indoor/virtual cycling: 2,736 sessions, 185 users.
- strength/cross-training: 1,245 sessions, 175 users.

Each family can therefore define a leave-one-family-out fold. Joint-shift test records are the intersection of the unseen-user test partition and the held-out family. Exact support will be recomputed after forecast-origin eligibility.

## Frozen external validation

All 32,078 duplicate-controlled GoldenCheetah sessions in the three primary families remain in the frozen primary external-test set:

- outdoor cycling: 25,964 sessions, 143 users;
- running: 3,835 sessions, 50 users;
- indoor/virtual cycling: 2,279 sessions, 41 users.

No GoldenCheetah session is used for Endomondo model selection or primary calibration.

For the separately labeled secondary adaptation analysis, users are selected by a family-stratified seeded hash. The union of family-specific calibration users yields 44 external-calibration users and leaves 100 external-test users. The user overlap is zero, and all three sport families occur in both subsets. This secondary partition does not alter the primary frozen external result.

## Assertions passed

- zero pairwise user overlap in Endomondo unseen-user partitions;
- zero within-user temporal-order violations;
- zero user overlap between GoldenCheetah secondary calibration and test;
- one unique session key per split-manifest row;
- split manifests generated before any forecast windows.

## Reproducible artifacts

- `outputs/manifests/endomondo_split_manifest_v0_2_0.csv`
- `outputs/manifests/goldencheetah_split_manifest_v0_2_0.csv`
- `outputs/audit/split_manifest_v0_2_0_summary.json`
- `outputs/manifests/endomondo_duplicate_resolution_v0_2_0.csv`
- `outputs/manifests/goldencheetah_duplicate_resolution_v0_2_0.csv`
- `src/audit_exact_duplicates.py`
- `src/resolve_duplicate_groups.py`
- `src/build_split_manifests.py`
