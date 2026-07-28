from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate_docx_submission_structure import (  # noqa: E402
    inspect_document,
    validate_submission,
)


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

ROOT_RELS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL}">
  <Relationship Id="rId1" Type="{R}/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def document_xml(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>{body}<w:sectPr/></w:body></w:document>"""


def good_table_xml(*, vertical: bool = False, shading: bool = False, header: bool = True) -> str:
    vertical_value = "single" if vertical else "nil"
    header_border = (
        '<w:tcBorders><w:bottom w:val="single" w:sz="6"/></w:tcBorders>' if header else ""
    )
    shading_xml = '<w:shd w:val="clear" w:fill="D9EAF7"/>' if shading else ""
    return f"""
<w:tbl>
  <w:tblPr><w:tblBorders>
    <w:top w:val="single" w:sz="8"/><w:bottom w:val="single" w:sz="8"/>
    <w:left w:val="nil"/><w:right w:val="nil"/><w:insideH w:val="nil"/>
    <w:insideV w:val="{vertical_value}"/>
  </w:tblBorders></w:tblPr>
  <w:tblGrid><w:gridCol w:w="3000"/><w:gridCol w:w="3000"/></w:tblGrid>
  <w:tr>
    <w:tc><w:tcPr>{header_border}{shading_xml}</w:tcPr><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc>
    <w:tc><w:tcPr>{header_border}</w:tcPr><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc>
  </w:tr>
  <w:tr>
    <w:tc><w:tcPr/><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc>
    <w:tc><w:tcPr/><w:p><w:r><w:t>2</w:t></w:r></w:p></w:tc>
  </w:tr>
</w:tbl>"""


def write_docx(
    path: Path,
    body: str,
    *,
    media: bool = False,
    malformed: bool = False,
) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", ROOT_RELS)
        archive.writestr("word/document.xml", "<broken" if malformed else document_xml(body))
        if media:
            archive.writestr("word/media/image1.png", b"not-a-real-image")
            archive.writestr(
                "word/_rels/document.xml.rels",
                f'<Relationships xmlns="{REL}"><Relationship Id="rId2" '
                f'Type="{R}/image" Target="media/image1.png"/></Relationships>',
            )


class DocxSubmissionStructureTests(unittest.TestCase):
    def test_valid_four_file_package_passes_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            paths = {
                "main": directory / "main.docx",
                "supplement": directory / "supplement.docx",
                "highlights": directory / "highlights.docx",
                "captions": directory / "captions.docx",
            }
            write_docx(paths["main"], good_table_xml())
            write_docx(paths["supplement"], good_table_xml())
            write_docx(paths["highlights"], "<w:p><w:r><w:t>Highlights</w:t></w:r></w:p>")
            write_docx(
                paths["captions"],
                "<w:p><w:r><w:t>Fig. 1. A Fig. 2. B Fig. 3. C Fig. 4. D "
                "Supplementary Fig. 1. E</w:t></w:r></w:p>",
            )
            output = directory / "audit.json"

            audit = validate_submission(paths, output)

            self.assertTrue(audit["overall_pass"])
            self.assertTrue(output.is_file())
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(saved["overall_pass"])
            self.assertEqual(saved["documents"]["main"]["table_audit"]["tables_found"], 1)

    def test_visible_vertical_border_and_cell_fill_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad-table.docx"
            write_docx(path, good_table_xml(vertical=True, shading=True))

            result = inspect_document(path, "main")

            self.assertFalse(result["pass"])
            codes = {failure["code"] for failure in result["table_audit"]["failures"]}
            self.assertIn("visible_vertical_table_border", codes)
            self.assertIn("visible_cell_shading", codes)

    def test_missing_header_bottom_border_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing-header-rule.docx"
            write_docx(path, good_table_xml(header=False))

            result = inspect_document(path, "supplement")

            failures = result["table_audit"]["failures"]
            self.assertEqual(
                sum(failure["code"] == "missing_header_bottom_border" for failure in failures),
                2,
            )

    def test_main_file_rejects_duplicated_figure_captions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "main-with-caption.docx"
            write_docx(
                path,
                good_table_xml()
                + "<w:p><w:r><w:t>Fig. 1. Study design. Full caption.</w:t></w:r></w:p>",
            )

            result = inspect_document(path, "main")

            self.assertFalse(result["pass"])
            separation = result["caption_separation_audit"]
            self.assertEqual(separation["duplicated_figure_caption_count"], 1)
            self.assertEqual(
                separation["failures"][0]["code"],
                "figure_captions_duplicated_in_main_word",
            )

    def test_caption_file_rejects_missing_label_and_embedded_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "captions.docx"
            write_docx(
                path,
                "<w:p><w:r><w:t>Fig. 1. A Fig. 2. B Fig. 3. C "
                "Supplementary Fig. 1. E</w:t></w:r></w:p>",
                media=True,
            )

            result = inspect_document(path, "captions")

            self.assertFalse(result["pass"])
            audit = result["caption_audit"]
            self.assertFalse(audit["required_labels"]["Fig. 4"])
            codes = {failure["code"] for failure in audit["failures"]}
            self.assertIn("embedded_media_parts_present", codes)
            self.assertIn("image_relationships_present", codes)

    def test_malformed_xml_is_not_openable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "malformed.docx"
            write_docx(path, "", malformed=True)

            result = inspect_document(path, "highlights")

            self.assertFalse(result["pass"])
            self.assertFalse(result["package"]["openable"])
            self.assertTrue(
                any(error.startswith("malformed_xml:word/document.xml") for error in result["package"]["errors"])
            )


if __name__ == "__main__":
    unittest.main()
