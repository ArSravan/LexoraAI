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
