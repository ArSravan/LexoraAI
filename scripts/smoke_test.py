"""End-to-end smoke test of a RUNNING Lexora AI (Docker container or local run), over its HTTP API.

    python -m scripts.smoke_test                                  # needs the API on http://127.0.0.1:8000
    python -m scripts.smoke_test --with-llm                       # also asks a grounded question (needs Ollama + model)
    python -m scripts.smoke_test --pdf sample_docs/ePrivacy_Directive_2002_58.pdf --api http://127.0.0.1:8000

It creates a throw-away case, uploads a PDF, waits for indexing, fetches a page image, asks a question the
documents cannot answer (must be refused), optionally asks a grounded one, then deletes the case. The result is
printed as a Markdown table that can be pasted into docs/DOCKER_VERIFICATION.md. Exit code 0 only if every
check that was run passed.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

UNANSWERABLE = "What is the governing law of the subcontractor agreement?"
# a question of the held-out set (F5) that the default document, the ePrivacy Directive, answers
GROUNDED = ("What must a provider of a publicly available electronic communications service do to safeguard the security "
            "of its services under the ePrivacy Directive?")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False

    @property
    def verdict(self) -> str:
        return "SKIPPED" if self.skipped else "PASS" if self.ok else "FAIL"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", skipped: bool = False) -> bool:
        self.checks.append(Check(name, ok, detail, skipped))
        return ok

    @property
    def passed(self) -> bool:
        return all(c.ok or c.skipped for c in self.checks)

    def markdown(self) -> str:
        rows = ["| Check | Result | Detail |", "|---|---|---|"]
        rows += [f"| {c.name} | {c.verdict} | {c.detail.replace('|', '/')} |" for c in self.checks]
        return "\n".join(rows)


def _wait_for(http, url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if http.get(url, timeout=3).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def run(api: str, pdf: Path, with_llm: bool, keep: bool = False, http=None, wait_s: float = 60,
        index_timeout: float = 900, ask_timeout: float = 900, unanswerable: str = UNANSWERABLE,
        grounded: str = GROUNDED) -> Report:
    http = http or requests
    rep = Report()
    base = api.rstrip("/") + "/api"

    if not rep.add("API health", _wait_for(http, f"{base}/health", wait_s), f"GET {base}/health"):
        return rep                                  # nothing else can work

    st = http.get(f"{base}/status", timeout=30).json()
    ollama = st.get("ollama", {})
    rep.add("Ollama reachable from the API", bool(ollama.get("connected")), f"{ollama.get('base_url', '?')}",
            skipped=not with_llm and not ollama.get("connected"))
    rep.add("Model available in Ollama", bool(ollama.get("model_available")), str(ollama.get("model", "?")),
            skipped=not with_llm and not ollama.get("model_available"))
    rep.add("Tesseract (OCR) available in the API environment", bool(st.get("ocr_available")), "")

    case = http.post(f"{base}/cases", json={"title": "smoke test"}, timeout=30).json()
    cid = case["id"]
    try:
        with open(pdf, "rb") as f:
            up = http.post(f"{base}/cases/{cid}/documents", files={"file": (pdf.name, f, "application/pdf")}, timeout=120)
        if not rep.add("Upload accepted", up.ok, f"{pdf.name}: HTTP {up.status_code}"):
            return rep
        doc_id, job_id = up.json()["document"]["id"], up.json()["job_id"]

        t0, job = time.time(), {}
        while time.time() - t0 < index_timeout:
            job = http.get(f"{base}/jobs/{job_id}", timeout=30).json()
            if job.get("status") != "running":
                break
            time.sleep(1)
        rep.add("Indexing finished", job.get("status") == "done",
                f"{job.get('status')} in {time.time() - t0:.0f} s; {job.get('message', '')} {job.get('error', '')}".strip())
        if job.get("status") != "done":
            return rep

        info = http.get(f"{base}/documents/{doc_id}/info", timeout=60).json()
        pages = info.get("pages", [])
        rep.add("Pages and structure detected", len(pages) > 0,
                f"{len(pages)} pages, {sum(1 for p in pages if p.get('is_scanned'))} scanned, "
                f"{len(info.get('headings', []))} headings")

        img = http.get(f"{base}/documents/{doc_id}/pages/1/image", timeout=60)
        rep.add("Evidence page image rendered", img.ok and img.headers.get("content-type", "").startswith("image/png")
                and len(img.content) > 1000, f"{len(img.content)} bytes")

        t0 = time.time()
        res = http.post(f"{base}/cases/{cid}/ask", json={"question": unanswerable}, timeout=ask_timeout).json()
        rep.add("Unanswerable question is refused", bool(res.get("abstained")) and res.get("confidence") == "INSUFFICIENT EVIDENCE",
                f"abstained={res.get('abstained')}, confidence={res.get('confidence')}, {time.time() - t0:.1f} s")

        if with_llm:
            t0 = time.time()
            res = http.post(f"{base}/cases/{cid}/ask", json={"question": grounded}, timeout=ask_timeout).json()
            verified = [c for c in res.get("claims", []) if c.get("status") == "VERIFIED"]
            rep.add("Grounded question answered with verified citations",
                    not res.get("abstained") and len(verified) >= 1,
                    f"confidence={res.get('confidence')}, {len(verified)}/{len(res.get('claims', []))} claims verified, "
                    f"{time.time() - t0:.0f} s" + (f", error: {res['error']}" if res.get("error") else ""))
        else:
            rep.add("Grounded question answered with verified citations", False, "run with --with-llm", skipped=True)
    finally:
        if not keep:
            r = http.delete(f"{base}/cases/{cid}", timeout=60)
            rep.add("Test case deleted", r.ok, f"HTTP {r.status_code}")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="base URL of the API")
    ap.add_argument("--pdf", type=Path, default=Path("sample_docs/ePrivacy_Directive_2002_58.pdf"))
    ap.add_argument("--with-llm", action="store_true", help="also ask a grounded question (slow: needs Ollama)")
    ap.add_argument("--keep", action="store_true", help="do not delete the test case")
    a = ap.parse_args(argv)
    if not a.pdf.is_file():
        print(f"PDF not found: {a.pdf} (python -m scripts.download_sample_docs fetches the samples)", file=sys.stderr)
        return 2
    rep = run(a.api, a.pdf, a.with_llm, a.keep)
    print(rep.markdown())
    print("\nRESULT:", "all checks passed" if rep.passed else "FAILED")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
