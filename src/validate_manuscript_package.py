"""Validate the editable manuscript package without modifying source files."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "main_manuscript.md"
BIBLIOGRAPHY = ROOT / "references" / "references.bib"
OUTPUT = ROOT / "outputs" / "audit" / "MANUSCRIPT_PACKAGE_VALIDATION.json"

TITLE = (
    "Uncertainty-Aware Exercise Heart-Rate Forecasting under User and Sport "
    "Distribution Shifts: A Leakage-Controlled Multi-Dataset Study"
)

FIGURE_STEMS = [
    "Figure_1_study_design",
    "Figure_2_primary_performance",
    "Figure_3_sport_shift",
    "Figure_4_uncertainty_calibration",
    "Supplementary_Figure_1_ablation_sensitivity",
]

OPTIONAL_FIGURE_STEMS = ["Graphical_Abstract"]

REQUIRED_FILES = [
    MANUSCRIPT,
    ROOT / "manuscript" / "BSPC_main_manuscript_draft.docx",
    ROOT / "manuscript" / "supplementary_material.md",
    ROOT / "manuscript" / "BSPC_supplementary_material.docx",
    ROOT / "manuscript" / "title_page.md",
    ROOT / "manuscript" / "AUTHOR_INPUT_FORM_CN.md",
    ROOT / "manuscript" / "highlights.txt",
    ROOT / "manuscript" / "BSPC_Highlights.docx",
    ROOT / "manuscript" / "BSPC_Figure_Captions.docx",
    ROOT / "manuscript" / "cover_letter.md",
    BIBLIOGRAPHY,
    ROOT / "references" / "LITERATURE_SEARCH_REPORT.md",
    ROOT / "references" / "ZOTERO_COLLECTION_REPORT.md",
    ROOT / "references" / "PRIOR_WORK_COMPARISON.md",
    ROOT / "protocol" / "FINAL_ANALYSIS_SPECIFICATION.md",
    ROOT / "protocol" / "PROTOCOL_DEVIATIONS.md",
    ROOT / "protocol" / "BSPC_SUBMISSION_READINESS_CHECKLIST.md",
    ROOT / "protocol" / "BSPC_SUBMISSION_FILE_INDEX.md",
    ROOT / "protocol" / "BSPC_LIVE_REQUIREMENTS_AUDIT_2026-07-22.md",
    ROOT / "DATA_SOURCES.md",
    ROOT / "REPRODUCING.md",
    ROOT / "PUBLIC_RELEASE_MANIFEST.md",
    ROOT / "outputs" / "audit" / "FINAL_RESULTS_VALIDATION.md",
    ROOT / "outputs" / "audit" / "REPORTED_NUMBER_VALIDATION.json",
    ROOT / "outputs" / "audit" / "RAW_SOURCE_INTEGRITY_v0_1_0.json",
    ROOT / "outputs" / "audit" / "DOCX_RENDER_QA.md",
    ROOT / "outputs" / "audit" / "SUPPLEMENT_DOCX_RENDER_QA.md",
    ROOT / "outputs" / "audit" / "HIGHLIGHTS_DOCX_RENDER_QA.md",
    ROOT / "outputs" / "audit" / "FIGURE_CAPTIONS_DOCX_RENDER_QA.md",
    ROOT / "outputs" / "audit" / "BSPC_DOCX_A11Y.json",
    ROOT / "outputs" / "audit" / "BSPC_SUPPLEMENT_A11Y.json",
    ROOT / "outputs" / "audit" / "BSPC_HIGHLIGHTS_A11Y.json",
    ROOT / "outputs" / "audit" / "BSPC_FIGURE_CAPTIONS_A11Y.json",
    ROOT / "outputs" / "audit" / "DOCX_SUBMISSION_STRUCTURE_VALIDATION.json",
    ROOT / "figures" / "FIGURE_QA.md",
]

MOJIBAKE_MARKERS = ["�", "鈥", "锛", "脳", "娓", "揺", "搒", "渦", "淭"]
PLACEHOLDER_PATTERNS = [
    r"\[[^\]\n]*(?:to be supplied|to be confirmed|decision required)[^\]\n]*\]",
    r"\[repository and persistent release DOI[^\]\n]*\]",
    r"\[institutional ethics determination[^\]\n]*\]",
    r"\[Author[^\]\n]*\]",
    r"\[Full name\]",
    r"\[Department,[^\]\n]*\]",
    r"\[(?:Postal address|Email|Telephone|Corresponding author name, degree|Institution|Address)\]",
]

FORBIDDEN_STALE_PHRASES = [
    "Causal forecast origins and inputs",
    "Frozen external validation showed lower accuracy without model failure",
    "GoldenCheetah is an independent data source",
]

GENERATED_DOCUMENT_SOURCES = {
    ROOT / "manuscript" / "BSPC_main_manuscript_draft.docx": [
        MANUSCRIPT,
        BIBLIOGRAPHY,
        ROOT / "src" / "build_bspc_docx.py",
        ROOT / "src" / "docx_table_geometry.py",
    ],
    ROOT / "manuscript" / "BSPC_supplementary_material.docx": [
        ROOT / "manuscript" / "supplementary_material.md",
        ROOT / "figures" / "Supplementary_Figure_1_ablation_sensitivity.png",
        ROOT / "src" / "build_supplementary_docx.py",
        ROOT / "src" / "build_bspc_docx.py",
        ROOT / "src" / "docx_table_geometry.py",
    ],
    ROOT / "manuscript" / "BSPC_Highlights.docx": [
        ROOT / "manuscript" / "highlights.txt",
        ROOT / "src" / "build_highlights_docx.py",
    ],
    ROOT / "manuscript" / "BSPC_Figure_Captions.docx": [
        MANUSCRIPT,
        ROOT / "src" / "build_figure_captions_docx.py",
    ],
}

RENDER_QA_EXPECTATIONS = {
    ROOT / "outputs" / "audit" / "DOCX_RENDER_QA.md": (
        ROOT / "outputs" / "docx_render_final7" / "main" / "BSPC_main_manuscript_draft.pdf",
        ROOT / "manuscript" / "BSPC_main_manuscript_draft.docx",
        ("outputs/docx_render_final7/main/BSPC_main_manuscript_draft.pdf", "All 18 pages"),
    ),
    ROOT / "outputs" / "audit" / "SUPPLEMENT_DOCX_RENDER_QA.md": (
        ROOT / "outputs" / "docx_render_final7" / "supplement" / "BSPC_supplementary_material.pdf",
        ROOT / "manuscript" / "BSPC_supplementary_material.docx",
        ("outputs/docx_render_final7/supplement/BSPC_supplementary_material.pdf", "All 19 pages"),
    ),
    ROOT / "outputs" / "audit" / "HIGHLIGHTS_DOCX_RENDER_QA.md": (
        ROOT / "outputs" / "docx_render_final7" / "highlights" / "BSPC_Highlights.pdf",
        ROOT / "manuscript" / "BSPC_Highlights.docx",
        ("outputs/docx_render_final7/highlights/BSPC_Highlights.pdf", "Verdict:** PASS"),
    ),
    ROOT / "outputs" / "audit" / "FIGURE_CAPTIONS_DOCX_RENDER_QA.md": (
        ROOT / "outputs" / "docx_render_final7" / "captions" / "BSPC_Figure_Captions.pdf",
        ROOT / "manuscript" / "BSPC_Figure_Captions.docx",
        ("outputs/docx_render_final7/captions/BSPC_Figure_Captions.pdf", "Verdict:** PASS"),
    ),
}

A11Y_AUDITS = {
    ROOT / "outputs" / "audit" / "BSPC_DOCX_A11Y.json": ROOT / "manuscript" / "BSPC_main_manuscript_draft.docx",
    ROOT / "outputs" / "audit" / "BSPC_SUPPLEMENT_A11Y.json": ROOT / "manuscript" / "BSPC_supplementary_material.docx",
    ROOT / "outputs" / "audit" / "BSPC_HIGHLIGHTS_A11Y.json": ROOT / "manuscript" / "BSPC_Highlights.docx",
    ROOT / "outputs" / "audit" / "BSPC_FIGURE_CAPTIONS_A11Y.json": ROOT / "manuscript" / "BSPC_Figure_Captions.docx",
}


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def section(text: str, start: str, end: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(start)}\s*$\n(.*?)(?=^{re.escape(end)}(?:\s|$))", text
    )
    if not match:
        raise ValueError(f"Missing section boundary: {start!r} -> {end!r}")
    return match.group(1).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    missing_required = [str(path.relative_to(ROOT)) for path in REQUIRED_FILES if not path.exists()]
    if missing_required:
        errors.append("Missing required files: " + ", ".join(missing_required))

    stale_generated_documents: list[str] = []
    for output, sources in GENERATED_DOCUMENT_SOURCES.items():
        if not output.exists() or not all(source.exists() for source in sources):
            continue
        newest_source = max(source.stat().st_mtime_ns for source in sources)
        if output.stat().st_mtime_ns < newest_source:
            stale_generated_documents.append(str(output.relative_to(ROOT)))
    if stale_generated_documents:
        errors.append(
            "Generated Word files are older than their source Markdown/BibTeX: "
            + ", ".join(stale_generated_documents)
        )

    render_qa_pass = True
    for path, (pdf_path, docx_path, expected_tokens) in RENDER_QA_EXPECTATIONS.items():
        if not path.exists():
            render_qa_pass = False
            continue
        qa_text = path.read_text(encoding="utf-8")
        required_tokens = ("23 July 2026", "PASS", *expected_tokens)
        missing_tokens = [token for token in required_tokens if token not in qa_text]
        if missing_tokens:
            render_qa_pass = False
            errors.append(
                f"Render QA is stale or incomplete ({path.name}): missing "
                + ", ".join(missing_tokens)
            )
        if not pdf_path.exists():
            render_qa_pass = False
            errors.append(f"Final render PDF is missing: {pdf_path.relative_to(ROOT)}")
        elif pdf_path.stat().st_mtime_ns < docx_path.stat().st_mtime_ns:
            render_qa_pass = False
            errors.append(f"Final render PDF is older than its DOCX: {pdf_path.relative_to(ROOT)}")
        elif path.stat().st_mtime_ns < pdf_path.stat().st_mtime_ns:
            render_qa_pass = False
            errors.append(f"Render QA record predates its PDF: {path.name}")

    a11y_pass = True
    for path, docx_path in A11Y_AUDITS.items():
        if not path.exists():
            a11y_pass = False
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            a11y_pass = False
            errors.append(f"Accessibility audit is unreadable ({path.name}): {exc}")
            continue
        counts = payload.get("counts", {})
        if any(int(counts.get(level, -1)) != 0 for level in ("high", "medium", "low")):
            a11y_pass = False
            errors.append(f"Accessibility audit has findings: {path.name}")
        if path.stat().st_mtime_ns < docx_path.stat().st_mtime_ns:
            a11y_pass = False
            errors.append(f"Accessibility audit predates its DOCX: {path.name}")

    reported_numbers_pass = False
    reported_numbers_path = ROOT / "outputs" / "audit" / "REPORTED_NUMBER_VALIDATION.json"
    if reported_numbers_path.exists():
        try:
            reported = json.loads(reported_numbers_path.read_text(encoding="utf-8"))
            reported_numbers_pass = (
                reported.get("status") == "PASS"
                and reported.get("check_count") == reported.get("checks_passed")
                and reported.get("authoritative_versions", {}).get("multiseed") == "0.22.0"
                and reported.get("authoritative_versions", {}).get("independent_zero_history")
                == "0.23.0"
                and reported.get("authoritative_versions", {}).get(
                    "balanced_calibration_and_interval_standardization"
                )
                == "0.24.0"
                and reported.get("authoritative_versions", {}).get("paired_user_bootstrap")
                == "0.25.0"
                and reported.get("authoritative_versions", {}).get(
                    "independent_persistence_conformal_baseline"
                )
                == "0.26.0"
                and reported.get("authoritative_versions", {}).get(
                    "matched_origin_sport_availability"
                )
                == "0.27.0"
                and reported.get("authoritative_versions", {}).get(
                    "deliberately_leaky_negative_control"
                )
                == "0.28.0"
                and reported.get("authoritative_versions", {}).get("reference_seed")
                == 20260722
            )
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"Reported-number audit is unreadable: {exc}")
    if not reported_numbers_pass:
        errors.append(
            "Reported-number audit is not a complete v0.22/v0.23/v0.24/v0.25/v0.26/v0.27/v0.28 PASS"
        )

    final_results_path = ROOT / "outputs" / "audit" / "FINAL_RESULTS_VALIDATION.md"
    final_results_pass = False
    if final_results_path.exists():
        final_results_text = final_results_path.read_text(encoding="utf-8")
        final_results_tokens = (
            "v0.28.0",
            "169 repository unit tests pass",
            "473/473 checks",
            "18 main-manuscript pages",
            "19 supplementary pages",
            "two figure-caption pages",
        )
        final_results_pass = all(
            token in final_results_text for token in final_results_tokens
        )
        final_results_sources = [
            reported_numbers_path,
            ROOT / "outputs" / "audit" / "DOCX_SUBMISSION_STRUCTURE_VALIDATION.json",
            *RENDER_QA_EXPECTATIONS.keys(),
            *A11Y_AUDITS.keys(),
        ]
        existing_sources = [path for path in final_results_sources if path.exists()]
        if existing_sources and final_results_path.stat().st_mtime_ns < max(
            path.stat().st_mtime_ns for path in existing_sources
        ):
            final_results_pass = False
            errors.append("Final-results validation report predates a defining QA artifact")
    if not final_results_pass:
        errors.append("Final-results validation report is stale or incomplete")

    figure_qa_path = ROOT / "figures" / "FIGURE_QA.md"
    figure_qa_pass = False
    if figure_qa_path.exists():
        figure_qa_text = figure_qa_path.read_text(encoding="utf-8")
        figure_qa_pass = all(
            token in figure_qa_text
            for token in ("23 July 2026", "PASS for manuscript assembly", "Twelve CSV files")
        )
    if not figure_qa_pass:
        errors.append("Figure QA record is stale or incomplete")

    text = MANUSCRIPT.read_text(encoding="utf-8")
    bib = BIBLIOGRAPHY.read_text(encoding="utf-8")
    submission_texts = {
        "main_manuscript.md": text,
        "title_page.md": (ROOT / "manuscript" / "title_page.md").read_text(encoding="utf-8"),
        "cover_letter.md": (ROOT / "manuscript" / "cover_letter.md").read_text(encoding="utf-8"),
    }

    first_line = text.splitlines()[0].removeprefix("# ").strip()
    if first_line != TITLE:
        errors.append(f"Title mismatch: {first_line!r}")

    abstract = section(text, "## Abstract", "**Keywords:**")
    abstract_words = word_count(abstract)
    if abstract_words >= 250:
        errors.append(
            f"Abstract is {abstract_words} words; it exceeds the conservative "
            "project gate of fewer than 250 words"
        )

    main_text = section(text, "## 1. Introduction", "## Data availability")
    main_text_words = word_count(main_text)

    title_page_text = submission_texts["title_page.md"]
    required_title_page_tokens = [
        TITLE,
        f"Abstract: {abstract_words} words",
        f"{main_text_words:,} words",
        "[Telephone]",
    ]
    missing_title_page_tokens = [
        token for token in required_title_page_tokens if token not in title_page_text
    ]
    if missing_title_page_tokens:
        errors.append(
            "Title-page template is stale or incomplete: "
            + "; ".join(missing_title_page_tokens)
        )

    cover_letter_text = submission_texts["cover_letter.md"]
    required_cover_letter_phrases = [
        "zero-history-trained strategy",
        "cross-source evaluation",
        "source- and composition-associated transport loss",
        "not under consideration elsewhere",
    ]
    missing_cover_letter_phrases = [
        phrase for phrase in required_cover_letter_phrases if phrase not in cover_letter_text
    ]
    if missing_cover_letter_phrases:
        errors.append(
            "Cover letter is stale or incomplete: "
            + "; ".join(missing_cover_letter_phrases)
        )

    keyword_match = re.search(r"(?m)^\*\*Keywords:\*\*\s*(.+)$", text)
    keywords = [item.strip() for item in keyword_match.group(1).split(";")] if keyword_match else []
    if not 1 <= len(keywords) <= 7:
        errors.append(f"Found {len(keywords)} keywords; expected 1-7")

    highlights_path = ROOT / "manuscript" / "highlights.txt"
    highlights = [line.strip() for line in highlights_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    highlight_lengths = [len(line) for line in highlights]
    if not 3 <= len(highlights) <= 5:
        errors.append(f"Found {len(highlights)} highlights; expected 3-5")
    too_long = [(line, len(line)) for line in highlights if len(line) > 85]
    if too_long:
        errors.append(f"Highlights exceed 85 characters: {too_long}")

    citation_keys = set(re.findall(r"@([A-Za-z0-9_:-]+)", text))
    bib_keys = set(re.findall(r"(?m)^@[A-Za-z]+\{([^,]+),", bib))
    missing_bib = sorted(citation_keys - bib_keys)
    uncited_bib = sorted(bib_keys - citation_keys)
    if missing_bib:
        errors.append("Citations missing from BibTeX: " + ", ".join(missing_bib))
    if uncited_bib:
        warnings.append("BibTeX entries not cited in manuscript: " + ", ".join(uncited_bib))

    dataset_entries = set(re.findall(r"(?m)^@dataset\{([^,]+),", bib))
    expected_dataset_entries = {"endomondo2019", "goldencheetah2018"}
    missing_dataset_markers = sorted(expected_dataset_entries - dataset_entries)
    if missing_dataset_markers:
        errors.append(
            "Dataset references are not typed for the required [dataset] marker: "
            + ", ".join(missing_dataset_markers)
        )

    mojibake = sorted(marker for marker in MOJIBAKE_MARKERS if marker in text)
    if mojibake:
        errors.append("Possible mojibake in manuscript: " + ", ".join(mojibake))

    stale_phrases = sorted(phrase for phrase in FORBIDDEN_STALE_PHRASES if phrase in text)
    if stale_phrases:
        errors.append("Stale or overclaiming phrases remain: " + "; ".join(stale_phrases))

    required_method_phrases = [
        "AI-assisted development and reproducibility controls",
        "same fitted checkpoint",
        "configuration-declared random seeds",
        "not prospectively registered",
    ]
    missing_method_phrases = [phrase for phrase in required_method_phrases if phrase not in text]
    if missing_method_phrases:
        errors.append(
            "Required reproducibility/limitation wording is missing: "
            + "; ".join(missing_method_phrases)
        )

    declaration_heading = (
        "## Declaration of generative AI and AI-assisted technologies in the "
        "manuscript preparation process"
    )
    declaration_position_ok = text.find(declaration_heading) >= 0 and text.find(
        declaration_heading
    ) < text.find("## References")
    if not declaration_position_ok:
        errors.append("The Elsevier AI declaration is missing or not placed before References")

    figure_files: dict[str, dict[str, bool]] = {}
    for stem in FIGURE_STEMS:
        status = {}
        for suffix in ("pdf", "svg", "tiff", "png"):
            present = (ROOT / "figures" / f"{stem}.{suffix}").exists()
            status[suffix] = present
            if not present:
                errors.append(f"Missing figure file: figures/{stem}.{suffix}")
        figure_files[stem] = status

    for stem in OPTIONAL_FIGURE_STEMS:
        status = {
            suffix: (ROOT / "figures" / f"{stem}.{suffix}").exists()
            for suffix in ("pdf", "svg", "tiff", "png")
        }
        figure_files[stem] = status
        if any(status.values()) and not all(status.values()):
            warnings.append(f"Optional graphical-abstract formats are incomplete: {status}")

    graphical_png = ROOT / "figures" / "Graphical_Abstract.png"
    graphical_dimensions = png_dimensions(graphical_png) if graphical_png.exists() else (0, 0)
    if graphical_png.exists() and (
        graphical_dimensions[0] < 1328 or graphical_dimensions[1] < 531
    ):
        errors.append(
            "Optional graphical abstract PNG is below 1328 x 531 px: "
            f"{graphical_dimensions[0]} x {graphical_dimensions[1]}"
        )
    if graphical_png.exists():
        warnings.append(
            "Graphical abstract is retained as an optional internal artifact; its "
            "submission requires the documented Elsevier AI-policy decision"
        )

    source_csvs = sorted((ROOT / "figures" / "source_data").glob("*.csv"))
    if not source_csvs:
        errors.append("No figure source-data CSV files found")

    placeholders = sorted(
        {
            f"{name}: {match.group(0)}"
            for name, source_text in submission_texts.items()
            for pattern in PLACEHOLDER_PATTERNS
            for match in re.finditer(pattern, source_text, flags=re.IGNORECASE)
        }
    )
    if placeholders:
        warnings.append(
            f"{len(placeholders)} author/institution-supplied placeholders remain; "
            "the technical package can pass, but the manuscript is not upload-ready"
        )

    docx_structure_path = (
        ROOT / "outputs" / "audit" / "DOCX_SUBMISSION_STRUCTURE_VALIDATION.json"
    )
    docx_structure_pass = False
    if docx_structure_path.exists():
        try:
            docx_structure = json.loads(docx_structure_path.read_text(encoding="utf-8"))
            docx_structure_pass = docx_structure.get("overall_pass") is True
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"DOCX structure audit is unreadable: {exc}")
        if not docx_structure_pass:
            errors.append(
                "DOCX submission structure audit has not passed; rebuild the Word "
                "files and rerun validate_docx_submission_structure.py"
            )

    if not 4500 <= main_text_words <= 5500:
        warnings.append(
            f"Main-text word count is {main_text_words}; full papers are "
            "usually about 5,000 words, so confirm that any deviation is justified"
        )

    technical_status = "PASS" if not errors else "FAIL"
    submission_status = (
        "AUTHOR_INPUT_REQUIRED" if not errors and placeholders else technical_status
    )

    report = {
        "status": technical_status,
        "submission_status": submission_status,
        "title": first_line,
        "abstract_words": abstract_words,
        "keyword_count": len(keywords),
        "keywords": keywords,
        "highlight_count": len(highlights),
        "highlight_character_counts": highlight_lengths,
        "main_text_words_including_headings": main_text_words,
        "bspc_full_paper_word_guidance": "normally about 5,000 words",
        "citation_key_count": len(citation_keys),
        "bibtex_key_count": len(bib_keys),
        "dataset_reference_keys": sorted(dataset_entries),
        "figure_files": figure_files,
        "graphical_abstract_png_dimensions": list(graphical_dimensions),
        "graphical_abstract_submission_policy": (
            "omit if optional, or document a policy-compliant human-controlled workflow"
        ),
        "source_data_csv_count": len(source_csvs),
        "placeholders": placeholders,
        "stale_phrase_check": "PASS" if not stale_phrases else "FAIL",
        "ai_declaration_position_check": "PASS" if declaration_position_ok else "FAIL",
        "docx_submission_structure_check": "PASS" if docx_structure_pass else "FAIL",
        "docx_render_qa_check": "PASS" if render_qa_pass else "FAIL",
        "docx_accessibility_check": "PASS" if a11y_pass else "FAIL",
        "reported_number_validation_check": "PASS" if reported_numbers_pass else "FAIL",
        "final_results_validation_check": "PASS" if final_results_pass else "FAIL",
        "figure_qa_check": "PASS" if figure_qa_pass else "FAIL",
        "stale_generated_documents": stale_generated_documents,
        "errors": errors,
        "warnings": warnings,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
