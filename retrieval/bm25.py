"""BM25 keyword retrieval with a legal-aware tokenizer.

Legal questions hinge on exact terms and clause numbers ("Article 83(5)", "7.2"). The tokenizer
therefore keeps clause numbers as tokens (and also emits their parts), which a plain
word tokenizer would shred.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from schemas import Chunk
from textutil import stem

STOPWORDS = frozenset("""a an the of to in on for and or by with is are was were be been being as at from that this
these those it its which who whom shall may must can could would should do does did not no any all such
than then there their they them his her he she we you your our what when where how why under into over per
also other about between if so but""".split())

# 7.2  |  83(5)(a)  |  words (with inner ' or -)  |  plain numbers
_TOKEN = re.compile(r"\d+(?:\.\d+)+|\d+(?:\([a-z0-9]+\))+|[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+")


def tokenize(text: str) -> list[str]:
    """Lower-case, stemmed tokens. 'Article 83(5)(a)' -> article, 83(5)(a), 83, 5, a ;
    '7.2' -> 7.2, 7, 2 ; 'cross-border' -> cross-border, cross, border."""
    out: list[str] = []
    for t in _TOKEN.findall(text.replace(" ", " ")):
        t = t.lower()
        if t in STOPWORDS:
            continue
        out.append(stem(t))
        if "(" in t:
            out.extend(p for p in re.split(r"[()]+", t) if p)
        elif "." in t and t[0].isdigit():
            out.extend(t.split("."))
        elif re.search(r"['’-]", t):
            out.extend(stem(p) for p in re.split(r"['’-]+", t) if len(p) > 1 and p not in STOPWORDS)
    return out


class BM25Index:
    """BM25 over ``Chunk.index_text`` (document name + section + clause + text)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.index_text or c.text) for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int = 20, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        """[(chunk_id, score)] best first. Chunks with score <= 0 are never returned.
        ``allowed`` restricts the result to those chunk ids."""
        if not self.bm25:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = np.array(self.bm25.get_scores(q), dtype=float)
        if allowed is not None:
            scores = np.where([i in allowed for i in self.ids], scores, 0.0)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]
