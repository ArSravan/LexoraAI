"""Background jobs with progress (indexing a PDF can take minutes: OCR, embeddings).

``start`` runs a function in a thread and returns a job id at once, so an upload request can answer
immediately; the UI then polls ``get(job_id)`` to draw a progress bar. Jobs live in memory: they are
progress indicators, the durable result of indexing is written by the job itself (database, indexes).
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Callable

Progress = Callable[[float, str], None]

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def start(kind: str, fn: Callable[[Progress], object], meta: dict | None = None) -> str:
    """Run ``fn(progress)`` in a background thread. ``progress(fraction, message)`` updates the job."""
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "status": "running", "progress": 0.0, "message": "Queued",
           "error": "", "meta": meta or {}, "started": time.time(), "finished": None}
    with _lock:
        _jobs[jid] = job

    def update(frac: float, msg: str) -> None:
        with _lock:
            job["progress"], job["message"] = round(min(1.0, max(0.0, frac)), 3), msg

    def run() -> None:
        try:
            fn(update)
            outcome = {"status": "done", "progress": 1.0, "message": "Done"}
        except Exception as e:  # noqa: BLE001 - the job reports any failure instead of dying silently
            outcome = {"status": "failed", "error": str(e), "message": f"Failed: {e}"}
        with _lock:
            job.update(outcome, finished=time.time())

    threading.Thread(target=run, daemon=True, name=f"job-{jid}").start()
    return jid


def get(job_id: str) -> dict | None:
    """A snapshot of the job (a copy, safe to serialise), or None for an unknown id."""
    with _lock:
        job = _jobs.get(job_id)
        return dict(job, meta=dict(job["meta"])) if job else None


def wait(job_id: str, timeout: float = 60.0) -> dict:
    """Block until the job is no longer running (used by tests and scripts)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get(job_id)
        if job is None or job["status"] != "running":
            return job
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} still running after {timeout}s")
