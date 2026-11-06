"""Cross-document conflict detection and numeric comparison.

Facts (amounts, percentages, dates) are extracted from the retrieved passages. Two facts about the same
topic with different values are REPORTED, never silently resolved:

  CONFLICT              different documents, same topic, different value, nothing explains the difference
  EXPLAINED_DIFFERENCE  a version/date qualifier explains it ("restated", "prior year", "amended" ...)

This is deliberately heuristic (topic = overlap of the words of the two sentences): it surfaces candidates
for a human to look at, it does not try to be exhaustive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from retrieval.bm25 import STOPWORDS
from schemas import AmountComparison, Conflict, EvidenceRef, RetrievedChunk
from textutil import clause_key, stem, words

MONTHS = "january february march april may june july august september october november december".split()
CURRENCY = r"(?:EUR|USD|GBP|CHF|€|\$|£|euros?|dollars?|pounds?)"
NUMBER = r"\d{1,3}(?:[ ,.  ]\d{3})+|\d+(?:[.,]\d+)?"
MULT = r"(?:million|billion|thousand|mn|bn|m|k)\b"
_MONEY = re.compile(
    rf"(?<![A-Za-z])(?P<c1>{CURRENCY})\s?(?P<n1>{NUMBER})\s*(?P<m1>{MULT})?"
    rf"|(?P<n2>{NUMBER})\s*(?P<m2>{MULT})?\s*(?P<c2>{CURRENCY})(?![A-Za-z])", re.I)
_PCT = re.compile(r"(\d+(?:[.,]\d+)?)\s?(?:%|per\s?cent|percent)", re.I)
_DATE = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b|\b(\d{4})-(\d{2})-(\d{2})\b", re.I)
_MULTIPLIER = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}
QUALIFIERS = re.compile(r"(?i)\b(restat\w*|prior[- ]year|previous\w*|former\w*|original\w*|amend\w*|revis\w*|"
                        r"supersed\w*|as at|as of|effective|updated|corrected|draft)\b")
_CUR_WORD = re.compile(r"(?<![A-Za-z])(EUR|USD|GBP|CHF)(?![A-Za-z])|([€$£])")


@dataclass
class Fact:
    kind: str                 # amount | percentage | date
    value: float | str
    display: str
    context: set[str]         # the topic: stems of the words around the figure
    sentence: str
    chunk: RetrievedChunk


def _currency(tok: str) -> str:
    t = tok.lower()
    return {"€": "EUR", "$": "USD", "£": "GBP"}.get(
        t, "EUR" if t.startswith("euro") else "USD" if t.startswith("dollar") else
        "GBP" if t.startswith("pound") else t.upper())


def _parse_amount(raw: str, mult: str | None) -> float | None:
    raw = re.sub(r"[   ]", "", raw)
    if re.fullmatch(r"\d{1,3}([,.]\d{3})+", raw):
        raw = re.sub(r"[,.]", "", raw)               # 10,000,000 / 10.000.000 (thousands separators)
    elif "," in raw:
        raw = raw.replace(",", ".")                  # decimal comma
    try:
        return float(raw) * _MULTIPLIER.get((mult or "").lower(), 1)
    except ValueError:
        return None


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+(?=[A-Z(])", text) if s.strip()]


def _topic(sentence: str) -> set[str]:
    return {stem(w) for w in words(sentence) if w not in STOPWORDS and len(w) > 2 and not w.isdigit()}


def _table_facts(hit: RetrievedChunk) -> list[Fact]:
    """Cells of a table carry no currency sign; the unit comes from the header, e.g. 'Amount (EUR)'."""
    text = hit.chunk.text
    m = _CUR_WORD.search(text)
    if not m:
        return []
    cur = _currency(m.group(1) or m.group(2))
    out = []
    for row in text.splitlines():
        cells = [c.strip() for c in row.split("|")]
        if len(cells) < 2 or not re.fullmatch(r"\d{1,3}(?:[,. ]\d{3})*(?:[.,]\d+)?", cells[-1] or "x"):
            continue
        v = _parse_amount(cells[-1], None)
        if v is not None:
            out.append(Fact("amount", v, f"{cur} {v:,.0f}", _topic(cells[0]), f"{cells[0]}: {cells[-1]}", hit))
    return out


def extract_facts(hit: RetrievedChunk) -> list[Fact]:
    facts: list[Fact] = []
    if hit.chunk.chunk_type == "table":
        facts += _table_facts(hit)
    text = hit.chunk.text
    for sent in _sentences(text) or [text]:
        topic = _topic(sent)
        for m in _MONEY.finditer(sent):
            raw, mult, cur = ((m.group("n1"), m.group("m1"), m.group("c1")) if m.group("n1")
                              else (m.group("n2"), m.group("m2"), m.group("c2")))
            v = _parse_amount(raw, mult)
            if v is not None:
                facts.append(Fact("amount", v, f"{_currency(cur)} {v:,.0f}", topic, sent, hit))
        for m in _PCT.finditer(sent):
            v = float(m.group(1).replace(",", "."))
            facts.append(Fact("percentage", v, f"{v:g}%", topic, sent, hit))
        for m in _DATE.finditer(sent):
            iso = (f"{m.group(3)}-{MONTHS.index(m.group(2).lower()) + 1:02d}-{int(m.group(1)):02d}" if m.group(1)
                   else f"{m.group(4)}-{m.group(5)}-{m.group(6)}")
            facts.append(Fact("date", iso, iso, topic, sent, hit))
    return facts


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _ref(hit: RetrievedChunk, text: str) -> EvidenceRef:
    c = hit.chunk
    return EvidenceRef(passage_id="", document=c.document_name, page=c.page_number, clause=c.clause,
                       supporting_text=text[:260])


def detect_conflicts(hits: list[RetrievedChunk], min_topic_similarity: float = 0.4) -> list[Conflict]:
    """Pairs of facts of the same kind and topic with different values (at most 5, hard conflicts first)."""
    facts = [f for h in hits for f in extract_facts(h)]
    out: list[Conflict] = []
    seen: set[tuple] = set()
    for i, a in enumerate(facts):
        for b in facts[i + 1:]:
            if a.kind != b.kind or a.chunk is b.chunk or a.value == b.value:
                continue
            same_doc = a.chunk.chunk.document_id == b.chunk.chunk.document_id
            if a.kind == "date" and same_doc:
                continue
            if _jaccard(a.context, b.context) < min_topic_similarity:
                continue
            qa, qb = QUALIFIERS.search(a.sentence), QUALIFIERS.search(b.sentence)
            qualifier_differs = bool(qa) != bool(qb) or (bool(qa) and bool(qb) and
                                {m.lower() for m in QUALIFIERS.findall(a.sentence)} !=
                                {m.lower() for m in QUALIFIERS.findall(b.sentence)})
            if same_doc and not qualifier_differs:
                continue            # two clauses of one document with different figures: different scopes
            key = (a.kind, a.display, b.display, a.chunk.chunk.chunk_id, b.chunk.chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            if qualifier_differs:
                cls = "EXPLAINED_DIFFERENCE"
                expl = (f"The figures differ ({a.display} vs {b.display}) but at least one passage carries a "
                        "version/date qualifier (e.g. restated, prior year, amended), so this is likely a "
                        "restatement or update rather than a contradiction. Both sources are shown; verify which applies.")
            else:
                cls = "CONFLICT"
                ca, cb = a.chunk.chunk, b.chunk.chunk
                expl = (f"Two documents state different {a.kind} values for the same topic ({a.display} in "
                        f"{ca.document_name} p.{ca.page_number} vs {b.display} in {cb.document_name} p.{cb.page_number}). "
                        "No date, version or scope qualifier explains the difference, so neither value is chosen automatically.")
            out.append(Conflict(kind=a.kind, topic=", ".join(sorted(a.context & b.context))[:80], classification=cls,
                                explanation=expl, sources=[_ref(a.chunk, a.sentence), _ref(b.chunk, b.sentence)],
                                values=[a.display, b.display]))
    out.sort(key=lambda c: c.classification != "CONFLICT")
    return out[:5]


# --------------------------------------------------------------------------- numeric comparison
_CMP = re.compile(r"(?i)\b(exceed\w*|greater|more than|less than|higher|lower|larger|smaller|bigger|"
                  r"compare\w*|versus|vs\.?|than|difference)\b")
_TOTAL = re.compile(r"(?i)\b(total|aggregate|combined|cap|limit|maximum|sum)\b")
_REFS = re.compile(r"(?i)\b(section|article|clause)\s+(\d+(?:\.\d+)*(?:\(\w+\))*)")
_SECTION_NUM = re.compile(r"(?i)(?:section|article|clause)?\s*(\d+(?:\.\d+)*)")


def _in_ref(hit: RetrievedChunk, ref: str) -> bool:
    """Does this chunk belong to 'Section/Article <ref>'? (by its clause label, article or section heading)"""
    c = hit.chunk
    key, ck = clause_key(ref), clause_key(c.clause)
    if ck == key or ck.startswith(key + "(") or ck.startswith(key + ".") or c.article == ref:
        return True
    m = _SECTION_NUM.match(c.section or "")
    return bool(m) and (m.group(1) == ref or m.group(1).startswith(ref + "."))


def compare_amounts(question: str, hits: list[RetrievedChunk]) -> list[AmountComparison]:
    """For a question such as "Does the cap in Section 7 exceed the total in Section 9?": find the amount in
    each referenced section among the retrieved passages and compare them, so the LLM need not do arithmetic."""
    if not _CMP.search(question):
        return []
    refs: list[str] = []
    for m in _REFS.finditer(question):
        if m.group(2) not in refs:
            refs.append(m.group(2))
    if len(refs) < 2:
        return []
    picked = []
    for r in refs[:2]:
        facts = [f for h in hits if _in_ref(h, r) for f in extract_facts(h) if f.kind == "amount"]
        if not facts:
            return []
        preferred = [f for f in facts if _TOTAL.search(f.sentence)] or facts      # 'total', 'cap', 'maximum' ...
        picked.append((r, max(preferred, key=lambda f: f.value)))
    (r1, f1), (r2, f2) = picked
    if f1.value == f2.value:
        result = f"Both amounts are equal ({f1.display})."
    else:
        verb = "exceeds" if f1.value > f2.value else "does not exceed"
        result = f"{f1.display} (ref. {r1}) {verb} {f2.display} (ref. {r2}); difference {abs(f1.value - f2.value):,.0f}."
    values = [{"ref": r, "display": f.display, "value": f.value, "document": f.chunk.chunk.document_name,
               "page": f.chunk.chunk.page_number, "clause": f.chunk.chunk.clause} for r, f in picked]
    return [AmountComparison(description=f"Automatic comparison of the amounts found in {r1} and {r2}",
                             values=values, result=result)]
