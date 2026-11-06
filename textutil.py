"""Small text helpers shared by retrieval and (later) verification."""
from __future__ import annotations

import re

_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
           "‐": "-", "‑": "-", " ": " ", " ": " ", "…": "..."}


def normalize(text: str) -> str:
    """Lower-case, straighten typographic quotes/dashes, collapse whitespace."""
    for k, v in _QUOTES.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip().lower()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalize(text))


_SUFFIXES = ("ations", "ation", "ings", "ing", "ied", "ies", "ed", "es", "s")


def stem(w: str) -> str:
    """A tiny suffix stemmer: fines / fined / fining / fine -> fin ; companies -> company.

    Deliberately crude: it only has to map different forms of a word to the SAME token on
    both the document side and the query side. Numbers and clause labels are left alone.
    """
    if not w.isalpha() or len(w) < 4:
        return w
    for suf in _SUFFIXES:
        if suf == "s" and w.endswith("ss"):          # process, access, class: not a plural
            continue
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
            return base[:-1] if base.endswith("e") else base
    return w[:-1] if w.endswith("e") else w


def clause_key(label: str) -> str:
    """Canonical clause label: 'Article 83(5)(a)' == '83(5)(a)', 'Section 7.2' == '7.2'."""
    s = normalize(label)
    s = re.sub(r"\b(article|art\.?|section|sec\.?|clause|paragraph|para\.?|§)\s*", "", s)
    return re.sub(r"[\s.]+$", "", s.replace(" ", ""))


# --------------------------------------------------------------------------- numbers
_MULT = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}
# "Article 83(5)", "Section 7.2", "page 14", "Table 9.1" ... : references, not quantities
_REF = re.compile(r"(?i)\b(article|art\.|section|clause|paragraph|para\.|page|pages|p\.|pp\.|table|annex|recital)\s*"
                  r"\d+(?:[.\-]\d+)*(?:\([a-z0-9]+\))*")
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:[ ,\u00a0\u202f]\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
                  r"\s*(thousand|million|billion|mn|bn|m|k|b)?\b", re.I)


def numbers(text: str, strip_refs: bool = True) -> set[float]:
    """Every quantity in ``text`` as a float, with unit multipliers applied.

    '20 000 000 EUR', 'EUR 20 million' and '20,000,000' all give 20000000.0, so a claim phrased one
    way can be checked against evidence phrased another way. Clause/page references such as
    'Article 83(5)' or 'page 14' are ignored unless ``strip_refs`` is False.
    """
    text = normalize(text)
    if strip_refs:
        text = _REF.sub(" ", text)
    out: set[float] = set()
    for m in _NUM.finditer(text):
        raw, mult = m.group(1), m.group(2)
        raw = re.sub(r"[\s\u00a0\u202f]", "", raw)
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", raw):
            raw = raw.replace(",", "")                  # 10,000,000
        elif "," in raw and "." not in raw:
            raw = raw.replace(",", ".")                 # decimal comma: 1,5 million
        try:
            v = float(raw)
        except ValueError:
            continue
        if mult:
            v *= _MULT.get(mult.lower(), 1)
        out.add(round(v, 4))
    return out
