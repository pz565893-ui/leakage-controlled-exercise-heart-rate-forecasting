"""Repository-local Word table geometry helpers for ``python-docx``.

The same width is written to the table properties, table grid, and every
cell so the generated manuscript renders consistently across Word-compatible
applications.  This module deliberately has no dependency on Codex skills,
user profiles, or machine-specific paths.
"""

from __future__ import annotations

from collections.abc import Sequence

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Twips


DEFAULT_CELL_MARGINS_DXA = {"top": 80, "bottom": 80, "start": 120, "end": 120}
DEFAULT_TABLE_INDENT_DXA = DEFAULT_CELL_MARGINS_DXA["start"]


def _ensure_child(parent, tag: str):
    child = parent.find(qn(tag))
    if child is None:
        child = OxmlElement(tag)
        parent.append(child)
    return child


def _set_width(parent, tag: str, width_dxa: int) -> None:
    width = _ensure_child(parent, tag)
    width.set(qn("w:type"), "dxa")
    width.set(qn("w:w"), str(int(width_dxa)))


def _set_cell_margins(cell, margins_dxa: dict[str, int]) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = _ensure_child(tc_pr, "w:tcMar")
    for side in ("top", "bottom", "start", "end"):
        margin = _ensure_child(tc_mar, f"w:{side}")
        margin.set(qn("w:w"), str(int(margins_dxa[side])))
        margin.set(qn("w:type"), "dxa")


def _replace_table_grid(table, column_widths_dxa: Sequence[int]) -> None:
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in column_widths_dxa:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(int(width)))
        grid.append(grid_col)


def _set_border(parent, edge: str, *, value: str, size: int = 0) -> None:
    border = _ensure_child(parent, f"w:{edge}")
    border.set(qn("w:val"), value)
    if value != "nil":
        border.set(qn("w:sz"), str(int(size)))
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), "000000")


def apply_bspc_table_borders(table, *, outer_size: int = 8, header_size: int = 6) -> None:
    """Apply a submission-safe three-line table without shading or vertical rules."""
    table_properties = table._tbl.tblPr
    borders = _ensure_child(table_properties, "w:tblBorders")
    _set_border(borders, "top", value="single", size=outer_size)
    _set_border(borders, "bottom", value="single", size=outer_size)
    for edge in ("left", "right", "insideH", "insideV"):
        _set_border(borders, edge, value="nil")

    if not table.rows:
        return
    for cell in table.rows[0].cells:
        cell_properties = cell._tc.get_or_add_tcPr()
        cell_borders = _ensure_child(cell_properties, "w:tcBorders")
        _set_border(cell_borders, "bottom", value="single", size=header_size)


def apply_table_geometry(
    table,
    column_widths_dxa: Sequence[int],
    *,
    table_width_dxa: int | None = None,
    indent_dxa: int | None = None,
    cell_margins_dxa: dict[str, int] | None = None,
) -> None:
    """Apply exact table, grid, column, and cell widths in DXA/twips.

    Call this after all unmerged rows have been added.  By default the table
    border is indented by the start cell margin, matching surrounding body
    text.  ``indent_dxa=0`` can be passed for deliberate edge alignment.
    """

    widths = [int(width) for width in column_widths_dxa]
    if not widths:
        raise ValueError("column_widths_dxa must not be empty")
    if any(width <= 0 for width in widths):
        raise ValueError("all column widths must be positive")

    width_total = int(table_width_dxa if table_width_dxa is not None else sum(widths))
    if sum(widths) != width_total:
        raise ValueError(
            "column widths must sum to table_width_dxa: "
            f"sum={sum(widths)} width={width_total}"
        )

    cell_margins = dict(DEFAULT_CELL_MARGINS_DXA)
    if cell_margins_dxa:
        cell_margins.update({key: int(value) for key, value in cell_margins_dxa.items()})
    resolved_indent = (
        int(cell_margins.get("start", DEFAULT_TABLE_INDENT_DXA))
        if indent_dxa is None
        else int(indent_dxa)
    )

    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    table_properties = table._tbl.tblPr
    _set_width(table_properties, "w:tblW", width_total)

    table_indent = _ensure_child(table_properties, "w:tblInd")
    table_indent.set(qn("w:type"), "dxa")
    table_indent.set(qn("w:w"), str(resolved_indent))

    layout = _ensure_child(table_properties, "w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    _replace_table_grid(table, widths)

    for column_index, width in enumerate(widths):
        table.columns[column_index].width = Twips(width)

    for row in table.rows:
        if len(row.cells) != len(widths):
            raise ValueError(
                "apply_table_geometry expects unmerged rows: "
                f"row has {len(row.cells)} cells, expected {len(widths)}"
            )
        row.height = None
        row_properties = row._tr.get_or_add_trPr()
        if row_properties.find(qn("w:cantSplit")) is None:
            row_properties.append(OxmlElement("w:cantSplit"))
        for column_index, cell in enumerate(row.cells):
            width = widths[column_index]
            cell.width = Twips(width)
            cell_properties = cell._tc.get_or_add_tcPr()
            _set_width(cell_properties, "w:tcW", width)
            _set_cell_margins(cell, cell_margins)
