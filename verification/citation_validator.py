"""Independent citation validation.

The LLM's citations are NEVER trusted. Every cited evidence item is checked against the case's real
documents and against the passages that were actually retrieved for THIS question:

  1. document exists in the case          4. the quoted text occurs in the source passage
  2. page exists (and fits the passage)   5. the passage was retrieved for this question
  3. clause exists in the document        6. the evidence actually supports the claim
                                             (its numbers and key terms), not just "a valid page"

Status of one evidence item:  VERIFIED (all checks)  >  PARTIAL (quote found in a retrieved passage but a
check failed)  >  UNVERIFIED  >  FABRICATED (document/page do not exist, or nothing matches).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from retrieval.bm25 import STOPWORDS
from schemas import (CheckResult, EvidenceRef, LLMAnswer, RetrievedChunk, VerifiedClaim,
                              VerifiedEvidence)
from textutil import clause_key, normalize, numbers, remove_refs, stem, words


@dataclass
class CaseDoc:
    """What the verifier knows about one document of the case."""
    document_id: str
    filename: str
    page_count: int
    labels: set[str]            # canonical clause keys (clause_key) found in the document


def _doc_name(name: str) -> str:
    return normalize(name).removesuffix(".pdf")


def _find_doc(name: str, docs: list[CaseDoc]) -> CaseDoc | None:
    n = _doc_name(name)
    if not n:
        return None
    return next((d for d in docs if _doc_name(d.filename) == n or d.document_id == name), None)


def _fuzzy(q: str, s: str) -> float:
    """Share of ``q`` covered by common blocks (>= 4 chars) with ``s``, in order."""
    if not q or not s:
        return 0.0
    sm = difflib.SequenceMatcher(None, q, s, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks() if b.size >= 4) / len(q)


def quote_match(quote: str, source: str) -> float:
    """How much of ``quote`` appears in ``source`` (0..1). Case, spacing and typographic quotes are
    ignored. "..." inside a quote marks an omission: its pieces must appear in order."""
    q, s = normalize(quote), normalize(source)
    if not q or not s:
        return 0.0
    if q in s:
        return 1.0
    pieces = [p.strip() for p in q.split("...") if p.strip()]
    if len(pieces) > 1:
        pos, matched = 0, 0.0
        for p in pieces:
            i = s.find(p, pos)
            if i >= 0:
                matched += len(p)
                pos = i + len(p)
            else:
                matched += _fuzzy(p, s[pos:]) * len(p)
        return matched / sum(len(p) for p in pieces)
    return _fuzzy(q, s)


def supports_claim(claim: str, quote: str, passage: str) -> bool:
    """Every number in the claim must occur in the evidence, and at least half of its key terms.

    Clause references in the claim ("Section 7.2 limits ...") are ignored: a clause label usually sits in
    the passage header, not in its text, and naming the clause must not count against a correct claim."""
    evidence = f"{quote} {passage}"
    claim_numbers = numbers(claim)
    if claim_numbers and not claim_numbers <= numbers(evidence):
        return False
    terms = {stem(w) for w in words(remove_refs(claim)) if w not in STOPWORDS and len(w) > 2 and not w.isdigit()}
    if not terms:
        return True
    present = {stem(w) for w in words(evidence)}
    return len(terms & present) / len(terms) >= 0.5


_NO_CLAUSE = {"-", "\u2013", "\u2014", "n/a", "na", "none", "null", "unknown"}


def _clean_ref(ref: EvidenceRef) -> EvidenceRef:
    """The prompt prints 'clause: -' for passages without a clause number, so a model that copies the header
    exactly cites '-'. That means "no clause", not a clause called '-'."""
    if ref.clause.strip().lower() in _NO_CLAUSE:
        return ref.model_copy(update={"clause": ""})
    return ref


def _related(a: str, b: str) -> bool:
    """Same clause, or one is a sub-clause of the other ('3(2)' ~ '3(2)(a)', but '7' is not '70')."""
    return a == b or a.startswith(b + "(") or a.startswith(b + ".") or b.startswith(a + "(") or b.startswith(a + ".")


def _source_text(c) -> str:
    return f"{c.lead_in} {c.text}" if c.lead_in else c.text


def resolve_chunk(ref: EvidenceRef, hits: list[RetrievedChunk],
                   docs: list[CaseDoc]) -> tuple[RetrievedChunk | None, list[str]]:
    """Find the retrieved passage a citation points to: by id ("E2" = hits[1]), else by (document, page, clause)."""
    notes: list[str] = []
    ref = _clean_ref(ref)
    m = re.fullmatch(r"(?i)e?(\d+)", (ref.passage_id or "").strip())
    if m and 1 <= int(m.group(1)) <= len(hits):
        return hits[int(m.group(1)) - 1], notes
    if ref.passage_id:
        notes.append(f"Passage id '{ref.passage_id}' was not part of the retrieved evidence.")
    d = _find_doc(ref.document, docs)
    ck = clause_key(ref.clause) if ref.clause else ""
    for h in hits:
        c = h.chunk
        if d and ref.page and c.document_id == d.document_id and c.page_number <= ref.page <= c.page_end \
                and (not ck or clause_key(c.clause) == ck):
            return h, notes
    return None, notes


def validate_evidence(ref: EvidenceRef, claim: str, hits: list[RetrievedChunk], docs: list[CaseDoc],
                      quote_min: float = 0.85) -> VerifiedEvidence:
    ref = _clean_ref(ref)
    hit, notes = resolve_chunk(ref, hits, docs)
    ch = hit.chunk if hit else None
    ref = ref.model_copy()
    checks = CheckResult()

    if ch is not None:          # blanks are filled from the cited passage; what the LLM wrote is kept
        if not ref.document:
            ref.document = ch.document_name
            notes.append("Document name taken from the cited passage id.")
        if ref.page is None:
            ref.page = ch.page_number
            notes.append("Page taken from the cited passage id.")
        if not ref.clause:
            ref.clause = ch.clause

    # 1-3: do the document, page and clause exist?
    doc = _find_doc(ref.document, docs)
    checks.document_exists = doc is not None
    if not doc:
        notes.append(f"Document '{ref.document}' is not part of this case.")
    if doc and ref.page is not None:
        checks.page_exists = 1 <= ref.page <= doc.page_count
        if not checks.page_exists:
            notes.append(f"Page {ref.page} does not exist (document has {doc.page_count} pages).")
    if doc and ref.clause:
        checks.clause_exists = clause_key(ref.clause) in doc.labels
        if not checks.clause_exists:
            notes.append(f"Clause '{ref.clause}' was not found in the extracted structure of {doc.filename}.")
    elif doc:
        checks.clause_exists = ch is not None and not ch.clause    # unstructured text has no clause to cite

    # 4: is the quote really in the source passage? (maybe it comes from another retrieved passage)
    source = _source_text(ch) if ch is not None else ""
    ratio = quote_match(ref.supporting_text, source) if ref.supporting_text and source else 0.0
    if ref.supporting_text and ratio < quote_min:
        for h in hits:
            r = quote_match(ref.supporting_text, _source_text(h.chunk))
            if r > ratio and r >= quote_min:          # adopt another passage only if the quote really is there
                ratio, hit, ch, source = r, h, h.chunk, _source_text(h.chunk)
                notes.append("Quote found in a different retrieved passage than the one cited.")
    checks.match_ratio = round(ratio, 3)
    checks.text_in_passage = bool(ref.supporting_text) and ratio >= quote_min
    if not ref.supporting_text:
        notes.append("No supporting text was quoted.")
    elif not checks.text_in_passage:
        notes.append(f"Quoted text not found in the source passage (match {ratio:.0%}).")

    # 5: was the passage retrieved for this question? Do the cited page/clause match that passage?
    checks.passage_retrieved = hit is not None
    if ch is not None and ref.page is not None and not (ch.page_number <= ref.page <= ch.page_end):
        notes.append(f"Cited page {ref.page} differs from the passage page {ch.page_number}.")
        checks.page_exists = False
    if ch is not None and ref.clause and ch.clause and not _related(clause_key(ref.clause), clause_key(ch.clause)):
        notes.append(f"Cited clause '{ref.clause}' differs from the passage clause '{ch.clause}'.")
        checks.clause_exists = False

    # 6: does the evidence support what the claim says?
    checks.supports_claim = ch is not None and supports_claim(claim, ref.supporting_text, source)
    if ch is not None and not checks.supports_claim:
        notes.append("The cited text does not support all numbers/terms of the claim.")

    core = (checks.document_exists and checks.page_exists and checks.clause_exists
            and checks.text_in_passage and checks.passage_retrieved)
    if core and checks.supports_claim:
        status = "VERIFIED"
    elif checks.text_in_passage and checks.passage_retrieved:
        status = "PARTIAL"
    elif not (checks.document_exists and checks.page_exists) or (not checks.passage_retrieved and not checks.text_in_passage):
        status = "FABRICATED"
    else:
        status = "UNVERIFIED"
    return VerifiedEvidence(
        ref=ref, status=status, checks=checks, notes=notes, matched_chunk_id=ch.chunk_id if ch else None,
        document_id=ch.document_id if ch else (doc.document_id if doc else ""),
        page_index=ch.page_index if ch else None, bbox=ch.bbox if ch else None,
        section=ch.section if ch else "", source_text=ch.text if ch else "")


def validate_answer(answer: LLMAnswer, hits: list[RetrievedChunk], docs: list[CaseDoc],
                    quote_min: float = 0.85) -> list[VerifiedClaim]:
    """Verify every claim. A claim is VERIFIED if at least one of its evidence items is."""
    out = []
    for c in answer.claims:
        if not c.evidence:
            out.append(VerifiedClaim(claim=c.claim, status="UNVERIFIED", evidence=[]))
            continue
        ev = [validate_evidence(e, c.claim, hits, docs, quote_min) for e in c.evidence]
        statuses = {e.status for e in ev}
        status = "VERIFIED" if "VERIFIED" in statuses else ("PARTIAL" if "PARTIAL" in statuses else "UNVERIFIED")
        out.append(VerifiedClaim(claim=c.claim, status=status, evidence=ev))
    return out
