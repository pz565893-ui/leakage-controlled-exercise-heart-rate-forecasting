from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_figure_captions_docx import build, extract_captions  # noqa: E402


class FigureCaptionDocumentTests(unittest.TestCase):
    def test_extracts_main_and_supplementary_captions(self) -> None:
        markdown = """# Test

## Figure captions

**Fig. 1. Design.** Main caption.

**Supplementary Fig. 1. Audit.** Supplement caption.

## Declaration
"""
        self.assertEqual(
            extract_captions(markdown),
            [
                ("Fig. 1. Design.", "Main caption."),
                ("Supplementary Fig. 1. Audit.", "Supplement caption."),
            ],
        )

    def test_builds_editable_docx_with_all_captions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.md"
            output = root / "captions.docx"
            source.write_text(
                "## Figure captions\n\n**Fig. 1. A.** First.\n\n**Fig. 2. B.** Second.\n",
                encoding="utf-8",
            )
            build(source, output)
            document = Document(output)
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            self.assertIn("Fig. 1. A. First.", text)
            self.assertIn("Fig. 2. B. Second.", text)


if __name__ == "__main__":
    unittest.main()
