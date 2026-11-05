"""Legal structure detection: articles, sections, clauses, sub-clauses, recitals, annexes.

Two families of documents are understood:
  * statutes / EU acts   "Article 83", "1.", "(a)", "(i)", recitals "(4)", "ANNEX I"
  * contracts            "Section 7", "7.2", "ARTICLE 3", "(a)", "(ii)"

The detector walks the lines of a parsed document in reading order and cuts them into
``Segment`` objects, each with a clause label such as ``83(5)(a)`` or ``7.2``.

Sequence checks (a -> b -> c, 1 -> 2 -> 3) stop a wrapped line that merely starts with
"(a)" or "2." from being mistaken for a new structural marker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from LexoraAI.schemas import ParsedDocument, TableData, TextLine

ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv"]

SEP = r"[-–—:.]"
RE_ARTICLE = re.compile(rf"^(?:ARTICLE|Article)\s+(\d+[A-Za-z]?)\s*(?:{SEP}\s*(.+))?$")
RE_CHAPTER = re.compile(rf"^(CHAPTER|Chapter|TITLE|Title|PART|Part)\s+([IVXLC]+|\d+)\s*(?:{SEP}\s*(.+))?$")
RE_SECTION = re.compile(rf"^(SECTION|Section|CLAUSE|Clause)\s+(\d+(?:\.\d+)*)\s*({SEP})?\s*(.*)$")
RE_ANNEX = re.compile(rf"^(ANNEX|Annex|SCHEDULE|Schedule|APPENDIX|Appendix)\s+([IVXLC]+|\d+|[A-Z])\b\s*(?:{SEP}\s*(.+))?$")
RE_NUM_CLAUSE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(\S.*)$")           # "7.2 text"
RE_PARA = re.compile(r"^(\d{1,3})\.(?:\s+(\S.*))?$")                    # "1." or "1. text"
RE_SUB = re.compile(r"^\(([a-z]{1,4}|\d{1,3})\)\s*(.*)$")               # "(a)" "(ii)" "(4)"
RE_END_PREAMBLE = re.compile(r"(?i)^(HAVE|HAS)\s+ADOPTED|AGREE\s+AS\s+FOLLOWS|HEREBY\s+AGREE|NOW,?\s+THEREFORE")
RE_TABLE_CAPTION = re.compile(r"(?i)^table\s+(\d+(?:\.\d+)*)")


@dataclass
class Segment:
    kind: str       # article_intro|paragraph|subpoint|subsubpoint|recital|text|table|footnote|heading
    label: str      # clause label, e.g. 83(5)(a), 7.2, Recital 4, Table 9.1
    parent: str     # label of the parent clause ("" for top level)
    article: str    # article / contract section number or annex name this segment belongs to
    section: str    # human-readable heading context, e.g. "Article 83 - General conditions ..."
    lines: list[TextLine] = field(default_factory=list)
    table: TableData | None = None
    depth: int = 0

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines)


def is_caps_heading(t: str) -> bool:
    letters = [c for c in t if c.isalpha()]
    return (3 <= len(t) <= 80 and len(letters) >= 3 and not t.endswith((".", ",", ";", ":"))
            and sum(c.isupper() for c in letters) / len(letters) >= 0.85)


def _short_title(t: str) -> bool:
    return (0 < len(t) <= 110 and not t.endswith((".", ";", ":", ",")) and t[:1].isupper()
            and not RE_SUB.match(t) and not RE_PARA.match(t))


class StructureDetector:
    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self.labels: set[str] = set()       # every clause label found (used to validate citations)
        self.headings: list[dict] = []      # {"type", "label", "page_index"} for the document outline

    # ---------------------------------------------------------------- input stream
    def run(self, doc: ParsedDocument) -> list[Segment]:
        """Interleave lines and tables by vertical position, then walk the stream."""
        items: list[tuple[str, object]] = []
        for page in doc.pages:
            tables = sorted(page.tables, key=lambda t: t.bbox[1])
            ti = 0
            for line in page.lines:
                while ti < len(tables) and tables[ti].bbox[1] <= line.bbox[1]:
                    items.append(("table", tables[ti]))
                    ti += 1
                items.append(("line", line))
            items += [("table", t) for t in tables[ti:]]
        self._walk(items)
        return self.segments

    # ---------------------------------------------------------------- the state machine
    def _walk(self, items) -> None:
        st = dict(chapter="", chapter_title="", section="", article="", article_title="",
                  in_preamble=True, in_annex=False, para_label="", sub=None, sub_label="",
                  roman=0, num_sub=0, last_para=0, contract_num="")
        cur: Segment | None = None
        resume: Segment | None = None       # clause interrupted by a footnote block
        pending_title: str | None = None    # heading that expects a title line next

        def context() -> str:
            if st["article"]:
                head = st["article"] if st["in_annex"] else f"Article {st['article']}"
                return f"{head} - {st['article_title']}".strip(" -")
            return st["section"] or st["chapter_title"]

        def open_seg(kind: str, label: str, parent: str, depth: int) -> Segment:
            seg = Segment(kind, label, parent, st["article"], context(), [], None, depth)
            self.segments.append(seg)
            if label:
                self.labels.add(label)
            return seg

        def reset_paragraph() -> None:
            st.update(para_label="", sub=None, sub_label="", roman=0, num_sub=0, last_para=0)

        for kind, obj in items:
            if kind == "table":
                t: TableData = obj                              # type: ignore[assignment]
                m = RE_TABLE_CAPTION.match(t.caption or "")
                label = f"Table {m.group(1)}" if m else f"Table (p.{t.page_index + 1})"
                seg = Segment("table", label, st["para_label"] or st["article"], st["article"],
                              context(), [], t, 1)
                self.segments.append(seg)
                self.labels.add(label)
                cur = None
                continue
            line: TextLine = obj                                # type: ignore[assignment]
            txt = line.text.strip()

            # footnotes become their own segments and never interrupt a clause
            if line.is_footnote:
                if cur is None or cur.kind != "footnote":
                    resume = cur if cur is not None else resume
                    cur = Segment("footnote", "", "", st["article"], context(), [], None, 0)
                    self.segments.append(cur)
                cur.lines.append(line)
                continue
            if cur is not None and cur.kind == "footnote":
                cur = resume

            # the line after "Article 5" / "CHAPTER II" is that heading's title
            if pending_title:
                if _short_title(txt) and not RE_ARTICLE.match(txt) and not RE_CHAPTER.match(txt):
                    key = "article_title" if pending_title == "article" else "chapter_title"
                    st[key] = (st[key] + " " + txt).strip()
                    if pending_title == "chapter":
                        st["section"] = st["chapter_title"]
                    pending_title = None
                    for seg in reversed(self.segments):         # keep the title with its heading
                        if seg.kind in ("article_intro", "heading"):
                            seg.section = context()
                            seg.lines.append(line)
                            break
                    continue
                pending_title = None

            if st["in_preamble"] and RE_END_PREAMBLE.search(txt):
                st["in_preamble"] = False
                cur = open_seg("text", "", "", 0)
                cur.lines.append(line)
                continue

            m = RE_ARTICLE.match(txt)
            if m and (not m.group(2) or len(txt) < 100):
                st.update(in_preamble=False, article=m.group(1), article_title=(m.group(2) or "").strip(),
                          in_annex=False)
                reset_paragraph()
                pending_title = None if m.group(2) else "article"
                self.headings.append({"type": "article", "label": m.group(1), "page_index": line.page_index})
                cur = open_seg("article_intro", m.group(1), "", 0)
                cur.lines.append(line)
                continue

            m = RE_ANNEX.match(txt)
            if m and (txt.split()[0].isupper() or len(txt) < 40) and len(txt) < 80:
                name = f"{m.group(1).title()} {m.group(2)}"
                st.update(in_preamble=False, article=name, article_title=(m.group(3) or "").strip(),
                          in_annex=True)
                reset_paragraph()
                pending_title = None if m.group(3) else "article"
                self.headings.append({"type": "annex", "label": name, "page_index": line.page_index})
                cur = open_seg("article_intro", name, "", 0)
                cur.lines.append(line)
                continue

            m = RE_CHAPTER.match(txt)
            if m and (txt.split()[0].isupper() or len(txt) < 40):
                st["chapter"] = f"{m.group(1).title()} {m.group(2)}"
                st["chapter_title"] = (m.group(3) or "").strip()
                if not st["in_annex"]:
                    st["article"] = ""
                st["article_title"] = ""
                st["section"] = f"{st['chapter']} {st['chapter_title']}".strip()
                pending_title = None if m.group(3) else "chapter"
                self.headings.append({"type": "chapter", "label": st["chapter"], "page_index": line.page_index})
                cur = open_seg("heading", "", "", 0)
                cur.lines.append(line)
                continue

            m = RE_SECTION.match(txt)
            if m:
                rest = (m.group(4) or "").strip()
                is_head = not rest or m.group(3) or (len(txt) < 80 and not txt.endswith((".", ",", ";")))
                if is_head and not (rest and rest[0].islower()):
                    num = m.group(2)
                    st.update(in_preamble=False, article="", article_title="", contract_num=num,
                              section=f"{m.group(1).title()} {num}" + (f" - {rest}" if rest else ""))
                    reset_paragraph()
                    st["para_label"] = num
                    self.headings.append({"type": "section", "label": num, "page_index": line.page_index})
                    cur = open_seg("paragraph", num, "", 0)
                    cur.lines.append(line)
                    continue

            m = RE_NUM_CLAUSE.match(txt)                        # contract clause "7.2 text"
            if m and not st["article"]:
                num = m.group(1)
                depth = num.count(".") + 1
                parent = num.rsplit(".", 1)[0] if depth > 1 else ""
                st.update(in_preamble=False, para_label=num, sub=None, sub_label="", roman=0)
                cur = open_seg("paragraph" if depth <= 2 else "subpoint", num, parent, depth - 1)
                cur.lines.append(line)
                continue

            m = RE_PARA.match(txt)                              # paragraph marker "1." / "1. text"
            if m:
                n, rest = int(m.group(1)), (m.group(2) or "")
                if n == st["last_para"] + 1:                    # sequence check: 1, 2, 3 ...
                    if rest and is_caps_heading(rest) and not st["article"]:
                        st.update(in_preamble=False, section=f"{n}. {rest}", last_para=n,
                                  para_label=str(n), contract_num=str(n))
                        self.labels.add(str(n))
                        self.headings.append({"type": "section", "label": str(n), "page_index": line.page_index})
                        cur = open_seg("paragraph", str(n), "", 0)
                        cur.lines.append(line)
                        continue
                    if st["article"]:
                        label = f"{st['article']}({n})"
                    elif st["contract_num"]:
                        label = f"{st['contract_num']}.{n}"
                    else:
                        label = str(n)
                    st.update(in_preamble=False, last_para=n, para_label=label, sub=None, sub_label="",
                              roman=0, num_sub=0)
                    cur = open_seg("paragraph", label, st["article"] or st["contract_num"], 1)
                    cur.lines.append(line)
                    continue

            m = RE_SUB.match(txt)                               # "(a)", "(ii)", "(4)"
            if m:
                handled = self._sub_marker(st, m.group(1), line, open_seg)
                if handled is not None:
                    cur = handled
                    continue

            if not st["in_preamble"] and not st["article"] and is_caps_heading(txt):
                st.update(section=txt.title(), para_label="", sub_label="")
                self.headings.append({"type": "heading", "label": txt.title(), "page_index": line.page_index})
                cur = open_seg("text", "", "", 0)
                cur.lines.append(line)
                continue

            if cur is None:                                     # continuation text
                cur = open_seg("text", "", "", 0)
            cur.lines.append(line)

    # ---------------------------------------------------------------- "(a)" "(ii)" "(4)"
    def _sub_marker(self, st: dict, tok: str, line: TextLine, open_seg) -> Segment | None:
        art = st["article"] or st["contract_num"] or ""
        if st["in_preamble"]:                                   # recitals: "(4) text"
            if tok.isdigit():
                st["section"] = "Recitals"
                seg = open_seg("recital", f"Recital {tok}", "", 0)
                seg.lines.append(line)
                return seg
            return None
        if tok.isdigit():                                       # numbered point "(1) 'data' means ..."
            n = int(tok)
            if n == st["num_sub"] + 1:
                label = f"{art}({n})" if art else f"({n})"
                st.update(num_sub=n, para_label=label, sub=None, sub_label="", roman=0)
                seg = open_seg("paragraph", label, art, 1)
                seg.lines.append(line)
                return seg
            return None
        parent = st["para_label"] or art
        letter = st["sub"]
        # a roman numeral nested under a lettered point: (i), (ii), (iii) ...
        if tok in ROMAN and st["sub_label"]:
            is_next_letter = letter is not None and len(tok) == 1 and ord(tok) == ord(letter) + 1
            if ROMAN.index(tok) + 1 == st["roman"] + 1 and not (is_next_letter and st["roman"] == 0):
                st["roman"] += 1
                seg = open_seg("subsubpoint", f"{st['sub_label']}({tok})", st["sub_label"], 3)
                seg.lines.append(line)
                return seg
        if len(tok) == 1 and tok == ("a" if letter is None else chr(ord(letter) + 1)):
            label = f"{parent}({tok})" if parent else f"({tok})"
            st.update(sub=tok, roman=0, sub_label=label)
            seg = open_seg("subpoint", label, parent, 2)
            seg.lines.append(line)
            return seg
        return None
