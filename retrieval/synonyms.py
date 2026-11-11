"""Query vocabulary: the words people use for a thing are often not the words the document uses.

A question about "vans" finds nothing in a regulation that only says "light commercial vehicles", and "EN-DE" is
"English-to-German" in running text. The document's wording is ADDED after the user's own words (never replacing
them: a table may really say "EN-DE"), so both forms can match, in BM25 as well as in the rerank and coverage checks.
"""
from __future__ import annotations

import re

# (pattern, words added when it matches)
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\bvans?\b|\bLCVs?\b"), "light commercial vehicles"),
    (re.compile(r"(?i)\bpassenger cars?\b|\bcars?\b"), "passenger cars"),
    (re.compile(r"(?i)\bEN[-– ]?DE\b|\bEN\s*(?:->|→)\s*DE\b"), "English-to-German"),
    (re.compile(r"(?i)\bEN[-– ]?FR\b|\bEN\s*(?:->|→)\s*FR\b"), "English-to-French"),
    (re.compile(r"(?i)\bZLEVs?\b"), "zero- and low-emission vehicles"),
    (re.compile(r"(?i)\bablations?\b"), "variations"),
    (re.compile(r"(?i)\bGDPR\b"), "General Data Protection Regulation"),
]


def expand_query(question: str) -> str:
    """The question followed by the document-style wording of its abbreviations and colloquial terms."""
    extra = [add for pat, add in _RULES if pat.search(question) and add.lower() not in question.lower()]
    return f"{question} ({'; '.join(extra)})" if extra else question
