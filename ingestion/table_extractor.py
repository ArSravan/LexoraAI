"""Table extraction, keeping page and bbox.

Two detectors work together:
  * tables drawn with ruling lines (pdfplumber, the "lines" strategy);
  * tables WITHOUT ruling lines, which scientific papers and many reports use. They are found from their caption
    ("Table 2: ..." above the table): the rows below the caption are read from the positions of the words.
"""
from __future__ import annotations

import re
import pymupdf

from schemas import TableData


MAX_TABLE_COLUMNS = 20      # a grid wider than this whose cells average under MIN_CELL_LENGTH characters is a figure
MIN_CELL_LENGTH = 8

_LABEL = r"table\s+(?:[A-Z]?\d+(?:\.\d+)*|[IVX]+)\s*[:.](?:\s|$)"      # "Table 2: ..." / "Table 3.1. ..." - not "Table 2 shows"
_CAPTION = re.compile(r"(?i)^" + _LABEL)
_CAPTION_ANYWHERE = re.compile(r"(?im)^" + _LABEL)
_ROW_TOLERANCE = 5.0        # points: words whose vertical centres differ by less are in the same row
_CELL_GAP = 7.0             # points: words closer than this in a row belong to the same cell
_MAX_ROW_GAP = 26.0         # points: a bigger vertical gap ends the table
_MIN_ROWS = 3
_NARROW = 60.0              # points: a cell wider than this is a label or a text column, not a value


_POWER_OF_TEN = re.compile(r"([·×⋅]|\bx)\s*10\s?(\d{1,2})\b")


def _clean(cell) -> str:
    """One line of text. A power of ten whose exponent the PDF set as a superscript ("2.3 · 1019") is written
    as "2.3 · 10^19": flattened, "1019" reads as a different number and values can no longer be compared."""
    return _POWER_OF_TEN.sub(r"\1 10^\2", " ".join(str(cell or "").split()))


def _plausible(rows: list[list[str]]) -> bool:
    """False for fragments and drawings (an attention map read as a wide grid of tiny pieces)."""
    if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
        return False
    cells = [c for r in rows for c in r]
    filled = sum(1 for c in cells if c)
    if filled < 4:
        return False
    return not (max(len(r) for r in rows) > MAX_TABLE_COLUMNS and sum(len(c) for c in cells) / filled < MIN_CELL_LENGTH)


# ------------------------------------------------------------------------------ tables without ruling lines
def _page_lines(page) -> list[tuple[float, float, float, float, str]]:
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            text = "".join(s["text"] for s in l["spans"]).strip()
            if text:
                out.append((*l["bbox"], text))
    return sorted(out, key=lambda t: (round(t[1], 0), t[0]))


def _caption_blocks(lines, page_width: float) -> list[tuple[int, float, float, float, float, str]]:
    """(index of the first line, x0, x1, bottom, last index, caption text) for every "Table N:" caption.
    A caption goes on over the following lines that start at the same x while the previous line ran to the margin."""
    found = []
    for i, (x0, y0, x1, y1, text) in enumerate(lines):
        if not _CAPTION.match(text):
            continue
        parts, last, bx1, bottom = [text], i, x1, y1
        right_edge = max((l[2] for l in lines if abs(l[0] - x0) <= 3 and l[2] - l[0] >= 0.25 * page_width), default=x1)
        j = i + 1
        while j < len(lines) and lines[last][2] >= right_edge - 20:
            nx0, ny0, nx1, ny1, nt = lines[j]
            if abs(nx0 - x0) > 3 or ny0 - bottom > 6 or ny0 < y0:
                break
            parts.append(nt)
            last, bx1, bottom = j, max(bx1, nx1), ny1
            j += 1
        found.append((i, x0, bx1, bottom, last, " ".join(parts)))
    return found


def _table_lines(lines, caption_bottom: float, x0: float, x1: float, column_width: float):
    """The lines below a caption that belong to the table: up to the first line of running text or a big vertical gap."""
    taken, last_bottom = [], caption_bottom
    for lx0, ly0, lx1, ly1, text in lines:
        if ly0 < caption_bottom - 1 or lx1 < x0 - 20 or lx0 > x1 + 20:
            continue
        if (lx1 - lx0) >= 0.7 * column_width and len(text.split()) >= 6:
            break                                           # a full line of prose: the table is over
        if ly0 - last_bottom > _MAX_ROW_GAP and taken:
            break
        if ly0 - last_bottom > _MAX_ROW_GAP + 14 and not taken:
            break                                           # nothing close below the caption
        taken.append((lx0, ly0, lx1, ly1, text))
        last_bottom = max(last_bottom, ly1)
    return taken


def _row_cells(words) -> list[list[tuple[float, float, str]]]:
    """Group words into rows by their vertical centre, and the words of a row into cells by the gap between them."""
    words = sorted(words, key=lambda w: (w[1] + w[3]) / 2)
    rows: list[list] = []
    for w in words:
        c = (w[1] + w[3]) / 2
        if rows and abs(c - rows[-1][0]) <= _ROW_TOLERANCE:
            rows[-1][1].append(w)
            rows[-1][0] = sum((v[1] + v[3]) / 2 for v in rows[-1][1]) / len(rows[-1][1])
        else:
            rows.append([c, [w]])
    out = []
    for _, ws in rows:
        cells: list[list] = []
        for w in sorted(ws, key=lambda v: v[0]):
            if cells and w[0] - cells[-1][1] < _CELL_GAP:
                cells[-1][1] = max(cells[-1][1], w[2])
                cells[-1][2] += " " + w[4]
            else:
                cells.append([w[0], w[2], w[4]])
        out.append([(a, b, t) for a, b, t in cells])
    return out


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _bands(rows) -> list[tuple[float, float]]:
    """Column bands: the x-ranges that cells occupy. Narrow cells make the bands first; a wide cell (a row label, a
    text column) makes its own band only where no narrow cell is, and a cell that covers several bands (a header
    that spans columns, a note across the table) makes none."""
    every = [(a, b) for r in rows for a, b, _ in r]

    def bridges(c) -> bool:             # overlaps two cells that do not overlap each other
        near = [o for o in every if o is not c and min(c[1], o[1]) - max(c[0], o[0]) > 0]
        return any(x[1] + 1 < y[0] for x in near for y in near)

    cells = [c for c in every if not bridges(c)]
    narrow = _merge([(a, b) for a, b in cells if b - a <= _NARROW])
    wide = []
    for a, b in (c for c in cells if c[1] - c[0] > _NARROW):
        hit = [n for n in narrow if min(b, n[1]) - max(a, n[0]) > 0]
        if not hit:
            wide.append((a, b))
    return _merge(narrow + wide)


def _is_data_row(cells) -> bool:
    """A row of values: some cell after the first holds a digit (a unit such as "x106" in a header does not count)."""
    return any(re.search(r"\d", t) and not re.match(r"[×x(]", t) for _, _, t in cells[1:])


def _grid(rows, bands) -> list[list[str]]:
    """Put every cell in its column. A header cell that spans several columns is repeated in each of them."""
    first_data = next((i for i, r in enumerate(rows) if _is_data_row(r)), 0)
    grid = []
    for ri, cells in enumerate(rows):
        row = [""] * len(bands)
        for a, b, t in cells:
            overlap = [max(0.0, min(b, hb) - max(a, ha)) for ha, hb in bands]
            hit = [k for k, o in enumerate(overlap) if o > 0]
            if ri < first_data and len(hit) > 1:
                targets = hit                                   # a spanning header: belongs to each column it covers
            elif hit:
                targets = [max(hit, key=lambda k: overlap[k])]
            else:
                mid = (a + b) / 2
                targets = [min(range(len(bands)), key=lambda k: abs(mid - (bands[k][0] + bands[k][1]) / 2))]
            for k in targets:
                row[k] = (row[k] + " " + t).strip()
        grid.append(row)
    if first_data > 1:                                          # a header written on several lines becomes one row
        head = [" ".join(g[k] for g in grid[:first_data] if g[k]) for k in range(len(bands))]
        grid = [head] + grid[first_data:]
    return grid


def _borderless_tables(page, page_index: int, known: list[tuple]) -> list[TableData]:
    if not _CAPTION_ANYWHERE.search(page.get_text("text")):
        return []
    lines = _page_lines(page)
    page_width = page.rect.width
    words = page.get_text("words")
    out = []
    for _, cx0, cx1, cbottom, _, caption in _caption_blocks(lines, page_width):
        if any(cbottom - 5 <= k[1] < cbottom + 60 for k in known):
            continue                                            # this caption belongs to a table found by its lines
        # the text column the caption sits in: lines that start where the caption starts
        same_column = [l for l in lines if abs(l[0] - cx0) <= 15 and l[2] - l[0] >= 0.25 * page_width]
        right_edge = max((l[2] for l in same_column), default=cx1)
        column_width = right_edge - cx0
        two_columns = column_width < 0.6 * page_width            # then the other column must not leak into the table
        part = _table_lines(lines, cbottom, *( (cx0, right_edge) if two_columns else (0.0, 10_000.0) ), column_width)
        if len(part) < 4:
            continue
        bx0, by0 = min(p[0] for p in part), min(p[1] for p in part)
        bx1, by1 = max(p[2] for p in part), max(p[3] for p in part)
        inside = [w for w in words if bx0 - 1 <= (w[0] + w[2]) / 2 <= bx1 + 1 and by0 - 1 <= (w[1] + w[3]) / 2 <= by1 + 1]
        rows = _row_cells(inside)
        if len(rows) < _MIN_ROWS:
            continue
        bands = _bands(rows)
        if len(bands) < 2:
            continue
        grid = [[_clean(c) for c in r] for r in _grid(rows, bands)]
        grid = [r for r in grid if any(r)]
        if _plausible(grid):
            out.append(TableData(page_index=page_index, bbox=(bx0 - 1, by0 - 1, bx1 + 1, by1 + 1), rows=grid, caption=caption))
    return out


# ------------------------------------------------------------------------------ the entry point
def extract_tables(pdf_path: str, page_indices: list[int]) -> dict[int, list[TableData]]:
    """Return {page_index: [TableData]} for the requested pages.

    Tables with ruling lines are found first (pdfplumber, "lines" strategy; pages with fewer than 4 drawn
    lines/rectangles are skipped, which keeps long legal PDFs fast). Then every "Table N:" caption that has no such
    table below it is used to read a table without ruling lines. Degenerate results (one row, one column, fewer
    than 4 filled cells, drawings) are discarded: they are usually layout artefacts, not tables.
    """
    import pdfplumber

    result: dict[int, list[TableData]] = {}
    with pymupdf.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as pdf:
        for pi in page_indices:
            known: list[tuple] = []
            if len(fitz_doc[pi].get_drawings()) >= 4:
                try:
                    found = pdf.pages[pi].find_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines"})
                except Exception:  # noqa: BLE001 - malformed page: skip its tables, keep its text
                    found = []
                for t in found:
                    rows = [[_clean(c) for c in r] for r in (t.extract() or [])]
                    rows = [r for r in rows if any(r)]
                    if _plausible(rows):
                        result.setdefault(pi, []).append(TableData(page_index=pi, bbox=tuple(t.bbox), rows=rows))
                        known.append(tuple(t.bbox))
            try:
                extra = _borderless_tables(fitz_doc[pi], pi, known)
            except Exception:  # noqa: BLE001 - this detector must never lose a page's text
                extra = []
            if extra:
                result.setdefault(pi, []).extend(extra)
    return result
