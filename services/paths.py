"""Filesystem layout for cases. Identifiers are validated so a hostile id cannot escape storage."""
from __future__ import annotations

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
