"""Legal-aware chunking.

A chunk is normally ONE numbered clause together with its sub-points, so that a retrieved
chunk is a unit a lawyer would cite. A clause longer than ``max_chars`` is split at its
sub-point boundaries; a piece that is still too long becomes overlapping "child" chunks that
inherit the clause metadata. Documents without detectable structure fall back to
heading-aware, overlapping paragraph packing. Tables become their own chunks.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from ingestion.structure_detector import Segment, StructureDetector
from schemas import Chunk, ParsedDocument, Region, TextLine

STRUCT_KINDS = {"article_intro", "paragraph", "subpoint", "recital"}
MIN_STRUCT_SEGMENTS = 3          # fewer structural markers than this -> treat the document as unstructured


@dataclass
class ChunkResult:
    chunks: list[Chunk]
    structure: dict = field(default_factory=dict)


def _regions(lines: list[TextLine]) -> list[Region]:
    """One bounding box per page touched by these lines."""
    by_page: dict[int, list[float]] = {}
    for l in lines:
        b = by_page.setdefault(l.page_index, list(l.bbox))
        b[0], b[1] = min(b[0], l.bbox[0]), min(b[1], l.bbox[1])
        b[2], b[3] = max(b[2], l.bbox[2]), max(b[3], l.bbox[3])
    return [Region(page_index=p, bbox=[round(v, 2) for v in b]) for p, b in sorted(by_page.items())]


def _join(lines: list[TextLine]) -> str:
    """Join lines into running text, repairing words hyphenated across a line break."""
    out = ""
    for l in lines:
        t = l.text.strip()
        if out.endswith("-") and t[:1].islower() and out[-2:-1].isalpha():
            out = out[:-1] + t
        else:
            out = f"{out} {t}".strip()
    return re.sub(r"\s+", " ", out)


class _Builder:
    def __init__(self, doc: ParsedDocument, max_chars: int, overlap: int):
        self.doc, self.max, self.overlap = doc, max_chars, overlap
        self.chunks: list[Chunk] = []

    def add(self, lines: list[TextLine], *, text: str | None = None, section: str, article: str,
            clause: str, ctype: str, parent: str = "", lead_in: str = "",
            regions: list[Region] | None = None) -> Chunk:
        text = text if text is not None else _join(lines)
        regs = regions if regions is not None else _regions(lines)
        page_idx = min(r.page_index for r in regs)
        confs = [l.confidence for l in lines if l.confidence is not None]
        head = " | ".join(x for x in (self.doc.filename, section, clause and f"clause {clause}") if x)
        chunk = Chunk(
            chunk_id=f"{self.doc.document_id}-{len(self.chunks):05d}", document_id=self.doc.document_id,
            document_name=self.doc.filename, page_number=page_idx + 1, page_index=page_idx,
            page_end=max(r.page_index for r in regs) + 1, section=section, article=article, clause=clause,
            text=text, chunk_type=ctype, bbox=regs[0].bbox, regions=regs,
            extraction_method="ocr" if lines and lines[0].method == "ocr" else "digital",
            ocr_confidence=round(sum(confs) / len(confs), 1) if confs else None,
            parent_clause=parent, lead_in=lead_in,
            index_text=f"{head}\n{lead_in + ' ' if lead_in else ''}{text}")
        self.chunks.append(chunk)
        return chunk

    def split_lines(self, lines: list[TextLine]) -> list[list[TextLine]]:
        """Greedy packing of whole lines into pieces of at most ``max`` characters.
        Each new piece starts with the last lines of the previous one (up to ``overlap`` chars)."""
        pieces: list[list[TextLine]] = []
        cur: list[TextLine] = []
        size = fresh = 0                      # fresh = lines added since the last piece was cut
        for l in lines:
            n = len(l.text) + 1
            if cur and fresh and size + n > self.max:
                pieces.append(cur)
                carry: list[TextLine] = []
                csz = 0
                for p in reversed(cur):
                    if csz + len(p.text) + 1 > self.overlap:
                        break
                    carry.insert(0, p)
                    csz += len(p.text) + 1
                cur, size, fresh = carry, csz, 0
            cur.append(l)
            size += n
            fresh += 1
        if cur and fresh:
            pieces.append(cur)
        return pieces


def _group(segments: list[Segment]) -> list[list[Segment]]:
    """A group = one clause and everything that belongs to it (its sub-points, trailing text)."""
    groups: list[list[Segment]] = []
    for s in segments:
        starts = s.kind in ("paragraph", "article_intro", "heading", "recital", "table", "footnote")
        if starts or not groups or groups[-1][0].kind in ("table", "footnote"):
            groups.append([s])
        else:
            groups[-1].append(s)
    return groups


def chunk_document(doc: ParsedDocument, max_chars: int = 1000, overlap: int = 150) -> ChunkResult:
    det = StructureDetector()
    segs = det.run(doc)
    b = _Builder(doc, max_chars, overlap)
    structured = sum(1 for s in segs if s.kind in STRUCT_KINDS) >= MIN_STRUCT_SEGMENTS
    if structured:
        _chunk_structured(b, segs)
    else:
        _chunk_fallback(b, doc)
    structure = {
        "mode": "legal-structure" if structured else "fallback-paragraph",
        "labels": sorted(det.labels),
        "n_articles": len([h for h in det.headings if h["type"] == "article"]),
        "n_sections": len({c.section for c in b.chunks if c.section}),
        "n_clauses": len({c.clause for c in b.chunks if c.clause}),
        "n_tables": sum(len(p.tables) for p in doc.pages),
        "headings": det.headings[:500],
    }
    return ChunkResult(b.chunks, structure)


def _table_text(t) -> str:
    return ((t.caption + "\n") if t.caption else "") + t.to_text()


def _chunk_structured(b: _Builder, segs: list[Segment]) -> None:
    for group in _group(segs):
        head = group[0]
        if head.kind == "table":
            t = head.table
            text = _table_text(t)
            b.add([TextLine(text=text, page_index=t.page_index, bbox=t.bbox)], text=text,
                  section=head.section, article=head.article, clause=head.label, ctype="table",
                  parent=head.parent, regions=[Region(page_index=t.page_index, bbox=[round(v, 2) for v in t.bbox])])
            continue
        lines = [l for s in group for l in s.lines]
        if not lines or head.kind == "heading":      # chapter headings only provide context
            continue
        if head.kind == "footnote":
            b.add(lines, section=head.section, article=head.article, clause="", ctype="footnote")
            continue
        label = head.label
        ctype = "recital" if head.kind == "recital" else "clause"
        if head.kind == "text" and not head.label:
            ctype, label = "text", ""
        text = _join(lines)
        if head.kind == "article_intro" and len(group) == 1 and len(text) < 160 \
                and not re.search(r"[.;:]\s*$", text):
            continue                                  # a bare "Article N / Title" heading has no content
        if len(text) <= b.max:
            b.add(lines, text=text, section=head.section, article=head.article, clause=label,
                  ctype=ctype, parent=head.parent)
            continue
        subs = [s for s in group[1:] if s.kind == "subpoint"]
        if not subs:
            for piece in b.split_lines(lines):
                b.add(piece, section=head.section, article=head.article, clause=label,
                      ctype="child", parent=head.parent)
            continue
        # long clause with sub-points: the clause's own opening text, then one chunk per sub-point
        lead_lines = head.lines + [l for l in group[1].lines if group[1].kind == "text"]
        lead = _join(lead_lines)
        if len(lead) > 40:
            b.add(lead_lines, text=lead, section=head.section, article=head.article, clause=label,
                  ctype=ctype, parent=head.parent)
        buckets: list[tuple[Segment, list[TextLine]]] = []
        for s in group[1:]:
            if s.kind == "subpoint":
                buckets.append((s, list(s.lines)))
            elif s.kind in ("subsubpoint", "text") and buckets:
                buckets[-1][1].extend(s.lines)
        for s, ls in buckets:
            pieces = [ls] if len(_join(ls)) <= b.max else b.split_lines(ls)
            for piece in pieces:
                b.add(piece, section=s.section, article=s.article, clause=s.label,
                      ctype="subpoint" if len(pieces) == 1 else "child", parent=label, lead_in=lead[:350])


def _chunk_fallback(b: _Builder, doc: ParsedDocument) -> None:
    """Unstructured text: headings are lines set in a clearly larger font (or ALL CAPS)."""
    lines = [l for p in doc.pages for l in p.lines if not l.is_footnote]
    sizes = [l.size for l in lines if l.size]
    median = statistics.median(sizes) if sizes else 0
    section = ""
    run: list[TextLine] = []

    def flush() -> None:
        nonlocal run
        for piece in b.split_lines(run):
            b.add(piece, section=section, article="", clause="", ctype="text")
        run = []

    for l in lines:
        t = l.text.strip()
        if len(t) < 90 and not t.endswith((".", ",", ";")) and ((median and l.size >= 1.15 * median) or t.isupper()):
            flush()
            section = t
        else:
            run.append(l)
    flush()
    for p in doc.pages:                                # tables still become their own chunks
        for t in p.tables:
            text = _table_text(t)
            m = re.match(r"(?i)^table\s+\d+(?:\.\d+)*", t.caption or "")
            b.add([TextLine(text=text, page_index=t.page_index, bbox=t.bbox)], text=text, section="",
                  article="", clause=m.group(0).title() if m else f"Table (p.{t.page_index + 1})",
                  ctype="table", regions=[Region(page_index=t.page_index, bbox=[round(v, 2) for v in t.bbox])])
