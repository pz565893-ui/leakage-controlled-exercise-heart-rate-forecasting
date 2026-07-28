from __future__ import annotations

import sys
import unittest
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_bspc_docx import (  # noqa: E402
    add_reference_list,
    clean_bib_value,
    format_author,
    format_reference,
    parse_bibtex,
)


class BspcReferenceFormattingTests(unittest.TestCase):
    def test_dataset_entries_receive_required_marker(self) -> None:
        entries = parse_bibtex((ROOT / "references" / "references.bib").read_text(encoding="utf-8"))
        for key in ("endomondo2019", "goldencheetah2018"):
            self.assertEqual(entries[key]["entry_type"], "dataset")
            self.assertTrue(format_reference(entries[key]).startswith("[dataset] "))

    def test_article_entries_are_not_marked_as_datasets(self) -> None:
        entries = parse_bibtex((ROOT / "references" / "references.bib").read_text(encoding="utf-8"))
        self.assertNotEqual(entries["ni2019"]["entry_type"], "dataset")
        self.assertFalse(format_reference(entries["ni2019"]).startswith("[dataset] "))

    def test_bibtex_accents_are_rendered_as_unicode(self) -> None:
        self.assertEqual(clean_bib_value(r"Blesi{\'c}"), "Blesić")
        self.assertEqual(clean_bib_value(r"Kne{\v{z}}evi{\'c}"), "Knežević")
        self.assertEqual(format_author(r"Romano, Jo{\~a}o Vitor"), "Romano JV")

    def test_reference_entries_are_left_aligned(self) -> None:
        entries = parse_bibtex(
            (ROOT / "references" / "references.bib").read_text(encoding="utf-8")
        )
        document = Document()
        add_reference_list(document, ["ni2019"], entries)
        self.assertEqual(document.paragraphs[-1].alignment, WD_ALIGN_PARAGRAPH.LEFT)


if __name__ == "__main__":
    unittest.main()
