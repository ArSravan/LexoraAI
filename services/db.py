"""A small SQLite store for cases, documents and the question log.

SQLite is a database in a single file (storage/lexora.db): no server to install. Every call opens its
own short-lived connection, so the API's request threads and the background indexing threads never
share one. Documents' CONTENT is not stored here (see services/indexer.py): only their status and counts.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from typing import Iterator

from config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(
  id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT DEFAULT '', created_at REAL);
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY, case_id TEXT NOT NULL, filename TEXT, sha256 TEXT, size_bytes INTEGER,
  page_count INTEGER DEFAULT 0, status TEXT DEFAULT 'queued', ocr_status TEXT DEFAULT 'n/a',
  n_sections INTEGER DEFAULT 0, n_clauses INTEGER DEFAULT 0, n_chunks INTEGER DEFAULT 0,
  n_tables INTEGER DEFAULT 0, n_scanned INTEGER DEFAULT 0, warnings TEXT DEFAULT '[]',
  error TEXT DEFAULT '', created_at REAL);
CREATE TABLE IF NOT EXISTS qa_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, question TEXT, abstained INTEGER,
  confidence TEXT, mode TEXT, model TEXT, latency REAL, created_at REAL);
"""
# columns update_document may change (the keys come from our own code, but never build SQL from unchecked names)
UPDATABLE = {"filename", "page_count", "status", "ocr_status", "n_sections", "n_clauses", "n_chunks",
             "n_tables", "n_scanned", "warnings", "error"}


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:10]


@contextmanager
def conn() -> Iterator[sqlite3.Connection]:
    path = get_settings().storage_dir / "lexora.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=30)) as c:
        c.row_factory = sqlite3.Row
        c.executescript(SCHEMA)
        yield c
        c.commit()


def _row(r) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    if "warnings" in d:
        d["warnings"] = json.loads(d["warnings"] or "[]")
    return d


# ---- cases
def create_case(title: str, description: str = "", case_id: str | None = None) -> dict:
    cid = case_id or new_id("c")
    with conn() as c:
        c.execute("INSERT INTO cases VALUES (?,?,?,?)", (cid, title.strip(), description.strip(), time.time()))
    return get_case(cid)


def get_case(case_id: str) -> dict | None:
    with conn() as c:
        return _row(c.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone())


def list_cases() -> list[dict]:
    with conn() as c:
        rows = c.execute("""SELECT cs.*, (SELECT COUNT(*) FROM documents d WHERE d.case_id = cs.id) AS n_docs
                            FROM cases cs ORDER BY created_at DESC""").fetchall()
        return [dict(r) for r in rows]


def delete_case(case_id: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM documents WHERE case_id=?", (case_id,))
        c.execute("DELETE FROM qa_log WHERE case_id=?", (case_id,))
        c.execute("DELETE FROM cases WHERE id=?", (case_id,))


# ---- documents
def add_document(case_id: str, filename: str, sha256: str, size: int, doc_id: str | None = None) -> dict:
    did = doc_id or new_id("d")
    with conn() as c:
        c.execute("INSERT INTO documents(id,case_id,filename,sha256,size_bytes,created_at) VALUES (?,?,?,?,?,?)",
                  (did, case_id, filename, sha256, size, time.time()))
    return get_document(did)


def get_document(doc_id: str) -> dict | None:
    with conn() as c:
        return _row(c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone())


def list_documents(case_id: str) -> list[dict]:
    with conn() as c:
        return [_row(r) for r in c.execute("SELECT * FROM documents WHERE case_id=? ORDER BY created_at, id",
                                           (case_id,)).fetchall()]


def update_document(doc_id: str, **fields) -> None:
    unknown = set(fields) - UPDATABLE
    if unknown:
        raise ValueError(f"Cannot update column(s): {', '.join(sorted(unknown))}")
    if "warnings" in fields:
        fields["warnings"] = json.dumps(fields["warnings"])
    cols = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE documents SET {cols} WHERE id=?", (*fields.values(), doc_id))


def delete_document(doc_id: str) -> None:
    with conn() as c:
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))


# ---- activity
def log_qa(case_id: str, question: str, abstained: bool, confidence: str, mode: str, model: str,
           latency: float) -> None:
    with conn() as c:
        c.execute("INSERT INTO qa_log(case_id,question,abstained,confidence,mode,model,latency,created_at) "
                  "VALUES (?,?,?,?,?,?,?,?)",
                  (case_id, question, int(abstained), confidence, mode, model, latency, time.time()))


def stats() -> dict:
    with conn() as c:
        one = lambda q: c.execute(q).fetchone()[0]                      # noqa: E731
        return {
            "cases": one("SELECT COUNT(*) FROM cases"),
            "documents": one("SELECT COUNT(*) FROM documents WHERE status='indexed'"),
            "chunks": one("SELECT COALESCE(SUM(n_chunks),0) FROM documents WHERE status='indexed'"),
            "questions": one("SELECT COUNT(*) FROM qa_log"),
            "abstentions": one("SELECT COUNT(*) FROM qa_log WHERE abstained=1"),
            "recent": [dict(r) for r in c.execute(
                "SELECT q.*, cs.title AS case_title FROM qa_log q LEFT JOIN cases cs ON cs.id = q.case_id "
                "ORDER BY q.id DESC LIMIT 10").fetchall()],
        }
