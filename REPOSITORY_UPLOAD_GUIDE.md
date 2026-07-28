# Code repository upload guide

## Upload artifact

Use the generated `release/code_repository_upload_v0_30_0.zip` as the repository
content bundle. Extract it locally, review the files, initialize the repository,
and upload the extracted directory contents rather than committing the ZIP file.

## Included

- executable Python analysis and validation code under `src/`;
- unit tests under `tests/`;
- the locked study configuration and protocol documentation;
- aggregate, non-identifying results used by the manuscript;
- publication figures as PDF, SVG, and PNG plus aggregate figure source data;
- manuscript Markdown, bibliography, reproducibility instructions, citation
  metadata, and deterministic SHA-256 integrity manifests.

## Excluded

- raw Endomondo and GoldenCheetah records;
- user/session identifiers, exact participant times, geolocation, free-text
  workout labels, row-level predictions, and private model-input arrays;
- model checkpoints and local normalization/calibration files;
- local environments, notebooks with unresolved machine paths, render caches,
  and operating-system temporary files;
- 600-dpi TIFF publication masters, which remain in the private manuscript
  workspace and are not needed for code reproduction.

## Author actions before public release

1. Add the final `repository-code` URL to `CITATION.cff`.
2. Select an institutionally approved software licence and add `LICENSE`.
3. Create a tagged release and archive that exact tag in Zenodo or another
   repository that provides a persistent DOI.
4. Insert the final repository URL and DOI into the manuscript's Code
   availability and Data availability sections.
5. Regenerate and verify the integrity manifest from the tagged commit.
