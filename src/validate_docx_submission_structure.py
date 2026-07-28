"""Validate the OOXML structure of the four BSPC submission Word files.

The validator deliberately uses only the Python standard library.  It does
not launch Word or LibreOffice and it does not modify the inspected files.
Its scope is intentionally narrow: package readability, BSPC three-line
tables in the manuscript and supplement, and the separate caption file.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = {
    "main": ROOT / "manuscript" / "BSPC_main_manuscript_draft.docx",
    "supplement": ROOT / "manuscript" / "BSPC_supplementary_material.docx",
    "highlights": ROOT / "manuscript" / "BSPC_Highlights.docx",
    "captions": ROOT / "manuscript" / "BSPC_Figure_Captions.docx",
}
DEFAULT_OUTPUT = ROOT / "outputs" / "audit" / "DOCX_SUBMISSION_STRUCTURE_VALIDATION.json"
VALIDATOR_VERSION = "1.0.0"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
W = f"{{{W_NS}}}"
PKG_REL = f"{{{PKG_REL_NS}}}"
NS = {"w": W_NS}

NO_BORDER_VALUES = {"nil", "none"}
NO_FILL_VALUES = {"", "auto", "none", "nil", "ffffff"}
VERTICAL_EDGES = ("left", "right", "start", "end", "insideV")


class DocxReadError(Exception):
    """Raised when a DOCX package cannot be structurally read."""


def _w_attr(element: ET.Element, name: str) -> str | None:
    return element.get(f"{W}{name}")


def _border_visible(element: ET.Element | None) -> bool | None:
    """Return True/False for a declared border, or None when undeclared."""
    if element is None:
        return None
    return (_w_attr(element, "val") or "single").lower() not in NO_BORDER_VALUES


def _shading_visible(element: ET.Element | None) -> bool:
    if element is None:
        return False
    value = (_w_attr(element, "val") or "clear").lower()
    fill = (_w_attr(element, "fill") or "").lower()
    color = (_w_attr(element, "color") or "").lower()
    if value in {"nil", "none"}:
        return False
    if fill not in NO_FILL_VALUES:
        return True
    return value not in {"clear", "nil", "none"} and color not in NO_FILL_VALUES


def _read_docx(path: Path) -> tuple[dict[str, Any], dict[str, ET.Element], set[str]]:
    package: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "openable": False,
        "errors": [],
        "member_count": 0,
    }
    roots: dict[str, ET.Element] = {}
    members: set[str] = set()
    if not path.is_file():
        package["errors"].append("file_not_found")
        return package, roots, members
    if not zipfile.is_zipfile(path):
        package["errors"].append("not_a_zip_package")
        return package, roots, members

    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            package["member_count"] = len(members)
            bad_member = archive.testzip()
            if bad_member:
                package["errors"].append(f"crc_error:{bad_member}")

            required = {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
            for missing in sorted(required - members):
                package["errors"].append(f"missing_required_part:{missing}")

            for name in sorted(members):
                if not (name.endswith(".xml") or name.endswith(".rels")):
                    continue
                try:
                    roots[name] = ET.fromstring(archive.read(name))
                except (ET.ParseError, KeyError) as exc:
                    package["errors"].append(f"malformed_xml:{name}:{exc}")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        package["errors"].append(f"zip_read_error:{type(exc).__name__}:{exc}")
        return package, roots, members

    document = roots.get("word/document.xml")
    if document is not None and document.tag != f"{W}document":
        package["errors"].append("word/document.xml_has_wrong_root")

    relationships = roots.get("_rels/.rels")
    if relationships is not None:
        office_targets = {
            relationship.get("Target", "").replace("\\", "/").lstrip("/")
            for relationship in relationships.findall(f"{PKG_REL}Relationship")
            if (relationship.get("Type") or "").endswith("/officeDocument")
        }
        if "word/document.xml" not in office_targets:
            package["errors"].append("root_relationship_to_word/document.xml_missing")

    package["openable"] = not package["errors"]
    return package, roots, members


def _table_styles(styles_root: ET.Element | None) -> tuple[dict[str, ET.Element], str | None]:
    styles: dict[str, ET.Element] = {}
    default_style = None
    if styles_root is None:
        return styles, default_style
    for style in styles_root.findall("w:style", NS):
        if _w_attr(style, "type") != "table":
            continue
        style_id = _w_attr(style, "styleId")
        if style_id:
            styles[style_id] = style
            if _w_attr(style, "default") in {"1", "true", "on"}:
                default_style = style_id
    return styles, default_style


def _style_chain(
    style_id: str | None, styles: dict[str, ET.Element]
) -> list[ET.Element]:
    chain: list[ET.Element] = []
    seen: set[str] = set()
    while style_id and style_id not in seen and style_id in styles:
        seen.add(style_id)
        style = styles[style_id]
        chain.append(style)
        based_on = style.find("w:basedOn", NS)
        style_id = _w_attr(based_on, "val") if based_on is not None else None
    return chain


def _style_edge_possible(chain: list[ET.Element], edge: str) -> bool:
    for style in chain:
        for borders in style.findall(".//w:tblBorders", NS) + style.findall(
            ".//w:tcBorders", NS
        ):
            if _border_visible(borders.find(f"w:{edge}", NS)) is True:
                return True
    return False


def _style_first_row_bottom(chain: list[ET.Element]) -> bool:
    for style in chain:
        for conditional in style.findall("w:tblStylePr", NS):
            if _w_attr(conditional, "type") != "firstRow":
                continue
            for borders in conditional.findall(".//w:tcBorders", NS) + conditional.findall(
                ".//w:tblBorders", NS
            ):
                if _border_visible(borders.find("w:bottom", NS)) is True:
                    return True
    return False


def _style_has_shading(chain: list[ET.Element]) -> bool:
    return any(
        _shading_visible(shading)
        for style in chain
        for shading in style.findall(".//w:shd", NS)
    )


def _table_border(tbl_pr: ET.Element | None, edge: str) -> bool | None:
    if tbl_pr is None:
        return None
    borders = tbl_pr.find("w:tblBorders", NS)
    return _border_visible(borders.find(f"w:{edge}", NS)) if borders is not None else None


def _cell_border(cell: ET.Element, edge: str) -> bool | None:
    tc_pr = cell.find("w:tcPr", NS)
    borders = tc_pr.find("w:tcBorders", NS) if tc_pr is not None else None
    return _border_visible(borders.find(f"w:{edge}", NS)) if borders is not None else None


def _resolved_table_border(
    tbl_pr: ET.Element | None, edge: str, style_chain: list[ET.Element]
) -> bool:
    direct = _table_border(tbl_pr, edge)
    return _style_edge_possible(style_chain, edge) if direct is None else direct


def _audit_table(
    table: ET.Element,
    index: int,
    styles: dict[str, ET.Element],
    default_style: str | None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    tbl_pr = table.find("w:tblPr", NS)
    style_node = tbl_pr.find("w:tblStyle", NS) if tbl_pr is not None else None
    style_id = _w_attr(style_node, "val") if style_node is not None else default_style
    chain = _style_chain(style_id, styles)

    for edge in ("top", "bottom"):
        if not _resolved_table_border(tbl_pr, edge, chain):
            failures.append({"code": f"missing_visible_outer_{edge}_border"})

    for edge in VERTICAL_EDGES:
        if _resolved_table_border(tbl_pr, edge, chain):
            failures.append({"code": "visible_vertical_table_border", "edge": edge})

    if _resolved_table_border(tbl_pr, "insideH", chain):
        failures.append({"code": "visible_internal_horizontal_table_border"})

    for properties_name in ("w:tblPr", "w:trPr"):
        properties = table.find(properties_name, NS)
        if properties is not None and any(
            _shading_visible(shading) for shading in properties.findall("w:shd", NS)
        ):
            failures.append({"code": "visible_table_or_row_shading"})
    if _style_has_shading(chain):
        failures.append({"code": "visible_shading_in_applied_table_style", "style": style_id})

    rows = table.findall("w:tr", NS)
    if not rows:
        failures.append({"code": "table_has_no_rows"})
    first_row_style_bottom = _style_first_row_bottom(chain)
    header_cells = rows[0].findall("w:tc", NS) if rows else []
    if rows and not header_cells:
        failures.append({"code": "header_row_has_no_cells"})

    for row_index, row in enumerate(rows, start=1):
        cells = row.findall("w:tc", NS)
        row_pr = row.find("w:trPr", NS)
        if row_pr is not None and any(
            _shading_visible(shading) for shading in row_pr.findall("w:shd", NS)
        ):
            failures.append({"code": "visible_row_shading", "row": row_index})
        for cell_index, cell in enumerate(cells, start=1):
            tc_pr = cell.find("w:tcPr", NS)
            if tc_pr is not None and any(
                _shading_visible(shading) for shading in tc_pr.findall("w:shd", NS)
            ):
                failures.append(
                    {"code": "visible_cell_shading", "row": row_index, "cell": cell_index}
                )
            for edge in VERTICAL_EDGES:
                if _cell_border(cell, edge) is True:
                    failures.append(
                        {
                            "code": "visible_vertical_cell_border",
                            "row": row_index,
                            "cell": cell_index,
                            "edge": edge,
                        }
                    )
            for edge in ("insideH", "insideV"):
                if _cell_border(cell, edge) is True:
                    failures.append(
                        {
                            "code": "visible_internal_cell_border",
                            "row": row_index,
                            "cell": cell_index,
                            "edge": edge,
                        }
                    )
            for edge in ("top", "bottom"):
                visible = _cell_border(cell, edge)
                allowed = (
                    (edge == "top" and row_index == 1)
                    or (edge == "bottom" and row_index == 1)
                    or (edge == "bottom" and row_index == len(rows))
                )
                if visible is True and not allowed:
                    failures.append(
                        {
                            "code": "visible_extra_horizontal_cell_border",
                            "row": row_index,
                            "cell": cell_index,
                            "edge": edge,
                        }
                    )

    for cell_index, cell in enumerate(header_cells, start=1):
        if _cell_border(cell, "bottom") is not True and not first_row_style_bottom:
            failures.append(
                {"code": "missing_header_bottom_border", "row": 1, "cell": cell_index}
            )

    return {
        "table_index": index,
        "style": style_id,
        "row_count": len(rows),
        "pass": not failures,
        "failures": failures,
    }


def _audit_tables(document: ET.Element, styles_root: ET.Element | None) -> dict[str, Any]:
    styles, default_style = _table_styles(styles_root)
    tables = document.findall(".//w:tbl", NS)
    results = [
        _audit_table(table, index, styles, default_style)
        for index, table in enumerate(tables, start=1)
    ]
    failures: list[dict[str, Any]] = []
    if not results:
        failures.append({"code": "no_tables_found"})
    for result in results:
        failures.extend(
            {"table_index": result["table_index"], **failure}
            for failure in result["failures"]
        )
    return {
        "pass": bool(results) and not failures,
        "tables_found": len(results),
        "tables_passed": sum(result["pass"] for result in results),
        "failures": failures,
        "tables": results,
    }


def _document_text(document: ET.Element) -> str:
    return " ".join((node.text or "") for node in document.findall(".//w:t", NS))


def _audit_main_caption_separation(document: ET.Element) -> dict[str, Any]:
    duplicated: list[str] = []
    pattern = re.compile(
        r"^(?:Supplementary\s+)?Fig(?:ure)?\.?\s*\d+\s*[.:]",
        re.IGNORECASE,
    )
    for paragraph in document.findall(".//w:p", NS):
        text = re.sub(
            r"\s+",
            " ",
            "".join((node.text or "") for node in paragraph.findall(".//w:t", NS)),
        ).strip()
        if pattern.match(text):
            duplicated.append(text[:160])
    return {
        "pass": not duplicated,
        "duplicated_figure_caption_count": len(duplicated),
        "duplicated_figure_captions": duplicated,
        "failures": (
            []
            if not duplicated
            else [{"code": "figure_captions_duplicated_in_main_word"}]
        ),
    }


def _audit_captions(
    document: ET.Element, roots: dict[str, ET.Element], members: set[str]
) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", _document_text(document)).strip()
    supplementary_pattern = re.compile(
        r"\bSupplementary\s+Fig(?:ure)?\.?\s*1\s*[.:]?", re.IGNORECASE
    )
    supplementary_found = bool(supplementary_pattern.search(text))
    primary_text = supplementary_pattern.sub(" ", text)
    labels = {
        f"Fig. {number}": bool(
            re.search(
                rf"\bFig(?:ure)?\.?\s*{number}\s*[.:]",
                primary_text,
                re.IGNORECASE,
            )
        )
        for number in range(1, 5)
    }
    labels["Supplementary Fig. 1"] = supplementary_found

    media_members = sorted(name for name in members if name.startswith("word/media/"))
    image_relationships: list[dict[str, str]] = []
    for name, root in roots.items():
        if not name.endswith(".rels"):
            continue
        for relationship in root.findall(f"{PKG_REL}Relationship"):
            relationship_type = relationship.get("Type") or ""
            if relationship_type.endswith("/image"):
                image_relationships.append(
                    {
                        "part": name,
                        "target": relationship.get("Target", ""),
                        "target_mode": relationship.get("TargetMode", "Internal"),
                    }
                )
    drawing_tags = sum(
        len(document.findall(f".//{W}{tag}")) for tag in ("drawing", "pict", "object")
    )
    failures: list[dict[str, Any]] = [
        {"code": "missing_required_caption_label", "label": label}
        for label, found in labels.items()
        if not found
    ]
    if media_members:
        failures.append({"code": "embedded_media_parts_present", "count": len(media_members)})
    if image_relationships:
        failures.append(
            {"code": "image_relationships_present", "count": len(image_relationships)}
        )
    if drawing_tags:
        failures.append({"code": "drawing_or_picture_elements_present", "count": drawing_tags})
    return {
        "pass": not failures,
        "required_labels": labels,
        "media_members": media_members,
        "image_relationships": image_relationships,
        "drawing_or_picture_element_count": drawing_tags,
        "failures": failures,
    }


def inspect_document(path: Path, role: str) -> dict[str, Any]:
    package, roots, members = _read_docx(path)
    result: dict[str, Any] = {"role": role, "package": package, "pass": False}
    document = roots.get("word/document.xml")
    if not package["openable"] or document is None:
        return result
    if role in {"main", "supplement"}:
        result["table_audit"] = _audit_tables(document, roots.get("word/styles.xml"))
        if role == "main":
            result["caption_separation_audit"] = _audit_main_caption_separation(
                document
            )
            result["pass"] = bool(
                result["table_audit"]["pass"]
                and result["caption_separation_audit"]["pass"]
            )
        else:
            result["pass"] = bool(result["table_audit"]["pass"])
    elif role == "captions":
        result["caption_audit"] = _audit_captions(document, roots, members)
        result["pass"] = bool(result["caption_audit"]["pass"])
    else:
        result["pass"] = True
    return result


def validate_submission(
    paths: dict[str, Path] | None = None, output: Path | None = DEFAULT_OUTPUT
) -> dict[str, Any]:
    resolved_paths = dict(DEFAULT_PATHS if paths is None else paths)
    missing_roles = sorted(set(DEFAULT_PATHS) - set(resolved_paths))
    if missing_roles:
        raise ValueError(f"missing document roles: {', '.join(missing_roles)}")
    documents = {
        role: inspect_document(Path(resolved_paths[role]), role)
        for role in ("main", "supplement", "highlights", "captions")
    }
    audit = {
        "validator": "validate_docx_submission_structure.py",
        "validator_version": VALIDATOR_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": [
            "DOCX package and XML readability",
            "three-line table borders and absence of visible shading",
            "absence of separately supplied figure captions from the main Word file",
            "required caption labels and absence of embedded images",
        ],
        "overall_pass": all(document["pass"] for document in documents.values()),
        "documents": documents,
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for role, default in DEFAULT_PATHS.items():
        parser.add_argument(f"--{role}", type=Path, default=default)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {role: getattr(args, role) for role in DEFAULT_PATHS}
    audit = validate_submission(paths, args.output)
    print(json.dumps({"overall_pass": audit["overall_pass"], "output": str(args.output)}))
    return 0 if audit["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
