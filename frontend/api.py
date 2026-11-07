"""Thin HTTP client for the Lexora API. The UI never touches the pipeline directly: everything goes through
the REST API, exactly as any other client would."""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterator

import requests


class ApiError(RuntimeError):
    """An error response, or no response at all. The message is meant to be shown to the user."""


def base_url() -> str:
    return os.getenv("LEXORA_API_URL", "http://127.0.0.1:8000").rstrip("/") + "/api"


def _check(r: requests.Response) -> requests.Response:
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if isinstance(detail, list):                       # FastAPI validation errors are a list of problems
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        raise ApiError(str(detail))
    return r


def _call(method: str, path: str, **kw) -> requests.Response:
    try:
        return _check(requests.request(method, f"{base_url()}{path}", **kw))
    except requests.ConnectionError as e:
        raise ApiError(f"Cannot reach the Lexora API at {base_url()}. Start it with `python main.py`.") from e
    except requests.Timeout as e:
        raise ApiError("The Lexora API did not answer in time.") from e


def get(path: str, **params):
    return _call("GET", path, params=params or None, timeout=60).json()


def post(path: str, json_body: dict | None = None, files=None, timeout: int = 600, **params):
    return _call("POST", path, json=json_body, files=files, params=params or None, timeout=timeout).json()


def delete(path: str):
    return _call("DELETE", path, timeout=60).json()


def image(path: str, **params) -> tuple[bytes | None, dict]:
    """(PNG bytes, highlight info) of a rendered page, or (None, {}) if it cannot be fetched."""
    try:
        r = requests.get(f"{base_url()}{path}", params=params, timeout=60)
    except requests.RequestException:
        return None, {}
    if not r.ok:
        return None, {}
    try:
        info = json.loads(r.headers.get("X-Highlight", "{}"))
    except ValueError:
        info = {}
    return r.content, info


def ask_stream(case_id: str, body: dict) -> Iterator[dict]:
    """Yield the events of POST /cases/{id}/ask/stream: status, token, then result (or error)."""
    try:
        with requests.post(f"{base_url()}/cases/{case_id}/ask/stream", json=body, stream=True, timeout=(10, 900)) as r:
            _check(r)
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)
    except requests.ConnectionError as e:
        raise ApiError(f"Cannot reach the Lexora API at {base_url()}.") from e


def wait_for_jobs(job_ids: list[str], on_progress: Callable[[float, str], None] | None = None,
                  poll: float = 0.5, timeout: float = 3600) -> list[dict]:
    """Poll until every job has finished; ``on_progress(mean fraction, message)`` is called on each poll."""
    deadline = time.time() + timeout
    while True:
        jobs = [get(f"/jobs/{j}") for j in job_ids]
        running = [j for j in jobs if j["status"] == "running"]
        if on_progress and jobs:
            on_progress(sum(j["progress"] for j in jobs) / len(jobs), (running[0] if running else jobs[-1])["message"])
        if not running:
            return jobs
        if time.time() > deadline:
            raise ApiError("Indexing is taking too long; check the API log.")
        time.sleep(poll)
