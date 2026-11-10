"""Filesystem layout for cases. Identifiers are validated so a hostile id cannot escape storage."""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

from config import get_settings

_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def check_id(value: str) -> str:
    """Accept only letters, digits, '_' and '-' (so no '..', '/', '\\' or drive letters)."""
    if not _ID.match(value or ""):
        raise ValueError("Invalid identifier.")
    return value


def cases_dir() -> Path:
    return get_settings().storage_dir / "cases"


def case_dir(case_id: str) -> Path:
    return cases_dir() / check_id(case_id)


def doc_dir(case_id: str, doc_id: str) -> Path:
    return case_dir(case_id) / "docs" / check_id(doc_id)


def chunks_file(case_id: str) -> Path:
    """All chunks of a case (every document) as JSON. BM25 is built from this file."""
    return case_dir(case_id) / "chunks.json"


def expand_globs(values) -> list[Path]:
    """Expand ``*`` and ``?`` in file arguments. PowerShell and cmd pass ``sample_docs/*.pdf`` on unexpanded (bash expands it before
    the program starts), so the scripts do it themselves and the same command works in every shell. Matches are sorted, an explicit
    name stays where it was typed, duplicates are dropped, and a pattern that matches nothing is kept as typed so that the
    later "file not found" message names it."""
    out: list[Path] = []
    for v in values:
        found = sorted(glob.glob(str(v))) if any(c in str(v) for c in "*?[") else []
        for p in map(Path, found or [v]):
            if p not in out:
                out.append(p)
    return out


class GlobPaths(argparse.Action):
    """argparse action for file arguments: the values are expanded with :func:`expand_globs`."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, expand_globs(values))
