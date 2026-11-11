"""OCR fallback for scanned (image-only) pages, using Tesseract through pytesseract."""
from __future__ import annotations

import io
import os
import re

import pymupdf
from PIL import Image

from schemas import TextLine

WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def configure_tesseract(cmd: str | None = None) -> bool:
    """Point pytesseract at the Tesseract binary. Returns True if OCR is usable.

    ``cmd`` is an explicit path (setting ``tesseract_cmd``). Otherwise the Windows default
    install path is tried, and finally whatever ``tesseract`` is on the PATH.
    """
    try:
        import pytesseract
    except ImportError:
        return False
    for candidate in (cmd, WINDOWS_DEFAULT):
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            break
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001 - any failure means OCR is unavailable
        return False


def render_page(page: "pymupdf.Page", dpi: int = 200) -> Image.Image:
    """Rasterise a PDF page (72 PDF points = 1 inch, so scale = dpi / 72)."""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


_BIG_O = re.compile(r"O\(([^()]{1,24})\)")
_BIG_O_TERM = re.compile(r"\s*([a-z])\s*([?²ý2�]?)\s*(?:[-–—·.•*]|$)\s*", re.I)
_CO2 = re.compile(r"\bCO[,.;:](?=\s*/\s*km|\s+emissions)")


def _big_o(m: "re.Match") -> str:
    """OCR reads the superscript in O(n·d²) as '?' (or '2') and the middle dot as '-': "O(n -d?)". When a marker
    follows a variable inside O(...) the variable was squared, so the formula is restored; otherwise it is left alone."""
    inner, pos, parts, marked = m.group(1), 0, [], False
    while pos < len(inner):
        t = _BIG_O_TERM.match(inner, pos)
        if not t or t.end() == pos:
            return m.group(0)
        parts.append(t.group(1) + ("²" if t.group(2) else ""))
        marked = marked or bool(t.group(2))
        pos = t.end()
    return "O(" + "·".join(parts) + ")" if marked and len(parts) >= 2 else m.group(0)


def fix_ocr_text(text: str) -> str:
    """Repair the OCR confusions that change meaning: squared variables in big-O formulas, and CO₂ read as 'CO,'."""
    return _CO2.sub("CO2", _BIG_O.sub(_big_o, text))


def ocr_page(page: "pymupdf.Page", page_index: int, dpi: int = 200, lang: str = "eng") -> list[TextLine]:
    """OCR one page. Returns lines with bounding boxes converted back to PDF points
    and a mean word confidence (0-100) per line."""
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(render_page(page, dpi), lang=lang, output_type=Output.DICT,
                                     config="--psm 3")
    scale = 72.0 / dpi                      # pixels -> PDF points
    grouped: dict[tuple, dict] = {}         # insertion order = Tesseract's reading order
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not word or conf < 0:            # empty cell or layout element, not a word
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        g = grouped.setdefault(key, {"words": [], "confs": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h})
        g["words"].append(word)
        g["confs"].append(conf)
        g["x0"], g["y0"] = min(g["x0"], x), min(g["y0"], y)
        g["x1"], g["y1"] = max(g["x1"], x + w), max(g["y1"], y + h)
    return [TextLine(text=fix_ocr_text(" ".join(g["words"])), page_index=page_index,
                     bbox=tuple(round(v * scale, 2) for v in (g["x0"], g["y0"], g["x1"], g["y1"])),
                     method="ocr", confidence=round(sum(g["confs"]) / len(g["confs"]), 1))
            for g in grouped.values()]
