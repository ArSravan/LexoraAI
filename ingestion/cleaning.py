"""Page clean-up: running headers/footers and page numbers."""
from __future__ import annotations

import re

from LexoraAI.schemas import ParsedPage

_PAGE_NUMBER = re.compile(r"(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?")


def _norm_margin(text: str) -> str:
    """Digits become '#', so 'Official Journal EN 7' and '... EN 8' compare equal."""
    return re.sub(r"\d+", "#", text.lower()).strip()


def _in_margin(line, page: ParsedPage) -> bool:
    return line.bbox[1] < 0.09 * page.height or line.bbox[3] > 0.92 * page.height


def strip_headers_footers(pages: list[ParsedPage]) -> int:
    """Remove running headers, footers and page numbers; returns how many lines were removed.

    A line in the top/bottom margin is removed when its digit-normalised text repeats on
    at least max(3, 30%) of the pages, or when it is just a page number ("12", "Page 3 of 10").
    Documents with fewer than 3 pages are left alone (not enough evidence of repetition).
    """
    n = len(pages)
    if n < 3:
        return 0
    counts: dict[str, int] = {}
    for p in pages:
        for key in {_norm_margin(l.text) for l in p.lines if _in_margin(l, p)}:
            counts[key] = counts.get(key, 0) + 1
    repeated = {k for k, v in counts.items() if v >= max(3, 0.3 * n)}
    removed = 0
    for p in pages:
        keep = []
        for l in p.lines:
            if _in_margin(l, p) and (_norm_margin(l.text) in repeated
                                     or _PAGE_NUMBER.fullmatch(l.text.strip().lower())):
                removed += 1
            else:
                keep.append(l)
        p.lines = keep
        p.char_count = sum(len(l.text) for l in p.lines)
    return removed
