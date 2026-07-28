# Third-party data sources and integrity record

This project uses two public research-data resources. Raw records are third-party data, remain outside the repository, and are not redistributed by this project. Users of the code must obtain the data from the original source, review the applicable terms, and cite the source publications or repository record.

## 1. Endomondo HR

The Endomondo HR and metadata files are associated with the public release accompanying Ni et al., *Modeling Heart Rate and Activity Data for Personalized Fitness Recommendation* (WWW 2019; DOI: 10.1145/3308558.3313643). The [official FitRec page](https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/FitRec-Project.html) limits the files to academic use and asks users not to redistribute them or use them commercially. The manuscript cites both the dataset record and associated publication.

The exact local source snapshot used for this study contained:

| Logical source | Files | Bytes | SHA-256 of raw bytes |
|---|---:|---:|---|
| Endomondo HR JSON | 1 | 6,568,384,411 | `1c021e3f8cb6428aefcb14ceff9d58ab2b6b2163bcbedfca71cb9f55616836e5` |
| Endomondo metadata JSON | 1 | 10,620,961,796 | `90461964a32557bea0ade6f5c9f46714a83606e88e22bde4cd7430a120031410` |

These hashes identify the exact byte streams analysed; they do not grant redistribution rights or assert that every independently hosted copy is canonical.

## 2. GoldenCheetah OpenData

GoldenCheetah OpenData is hosted by the Open Science Framework:

- project record: <https://osf.io/6hfpz/>
- DOI: <https://doi.org/10.17605/OSF.IO/6HFPZ>

The extracted snapshot used here contained 51,620 regular files and 7,644,587,363 bytes. Its integrity values are:

- content-multiset SHA-256: `10db1f7f6bd4792c72dc29556d05d786df723d1bd77bb55a4412752e7f646322`
- deterministic tree SHA-256: `ad54886c23ced6744dbf7dd9397e9bf1441a24172d3384785945134cf50dd400`

The content-multiset hash combines the sorted raw per-file SHA-256 digests and retains duplicate contents; it is independent of file names. The tree hash additionally binds NFC-normalized relative paths to file hashes and therefore verifies both contents and extracted organization. The public audit records neither the relative paths nor a source-file manifest. The OSF record currently identifies CC0 1.0, but this project still does not mirror the raw activity files because licence status does not remove privacy, linkage, or third-party-rights risks.

## 3. Combined source snapshot

The three logical inputs totalled 24,833,933,570 bytes across 51,622 files: 51,470 CSV files and 152 JSON files. `outputs/audit/RAW_SOURCE_INTEGRITY_v0_1_0.json` records these non-identifying totals, algorithms, durations, and hashes. It contains no local absolute path, source-file-name inventory, user identifier, session/activity identifier, exact participant timestamp, or geolocation value.

## 4. Expected local organization

Keep the sources outside the repository. The default audit command expects a data root with this logical organization:

```text
<data-root>/
├── endomondoHR.json/
│   └── endomondoHR.json
├── endomondoMeta.json/
│   └── endomondoMeta.json
└── GoldenCheetah_extracted/
    └── <extracted athlete archives and activity files>
```

Alternative locations can be supplied explicitly without changing the repository:

```powershell
python src\audit_raw_source_integrity.py `
  --endomondo-hr <endomondo-hr-json> `
  --endomondo-metadata <endomondo-metadata-json> `
  --goldencheetah-root <goldencheetah-extracted-root>
```

The audit opens inputs read-only, rejects links/reparse points, snapshots file identity before hashing, and checks that source size, modification time, device, and file identity do not change during the run.

## 5. Reproduction boundary

A hash match establishes that another run starts from the same byte-level snapshot. It does not by itself reproduce preprocessing, splits, checkpoint selection, or results. Follow `REPRODUCING.md`, the version ledger in `protocol/FINAL_ANALYSIS_SPECIFICATION.md`, and the authoritative-artifact map in `protocol/RESULT_ARTIFACT_AUTHORITY.md`. Do not place raw records, extracted device files, row-level derived artifacts, or the full raw-label ontology in a public code archive: GoldenCheetah workout labels can contain linkable free text such as route or place names. The released mapping code regenerates the internal ontology from source data without republishing those labels.
