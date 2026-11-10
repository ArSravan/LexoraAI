"""Ask a question about one or more PDFs (needs Ollama running with the model pulled).

    python -m scripts.ask_pdf contract.pdf annual_report.pdf -q "Does the cap in Section 7 exceed Section 9?"
    python -m scripts.ask_pdf a.pdf -q "..." --model qwen3:4b --no-rerank
    python -m scripts.ask_pdf a.pdf -q "..." --no-retrieval      # the same model WITHOUT documents (baseline)

The PDFs are indexed once (see scripts/search_pdf.py) and reused. The answer is printed with every claim,
its citations and the result of each verification check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from generation.answer_generator import AnswerService
from generation.ollama_client import OllamaClient
from schemas import AnswerResult
from scripts.search_pdf import ensure_indexed
from services.paths import GlobPaths

CHECKS = [("document_exists", "document"), ("page_exists", "page"), ("clause_exists", "clause"),
          ("text_in_passage", "quote"), ("passage_retrieved", "retrieved"), ("supports_claim", "supports")]


def render(res: AnswerResult) -> str:
    out = [f"QUESTION: {res.question}", "",
           f"ANSWER   [confidence: {res.confidence}]" + ("   [no retrieval: ungrounded]" if not res.use_retrieval else ""),
           res.answer.strip()]
    if res.confidence_factors:
        out += ["", "Why this confidence:"] + [f"  - {f}" for f in res.confidence_factors]
    if res.claims:
        out += ["", "CLAIMS"]
        for i, c in enumerate(res.claims, 1):
            out.append(f"{i}. [{c.status}] {c.claim}")
            for e in c.evidence:
                r = e.ref
                flags = " ".join(f"{label}:{'ok' if getattr(e.checks, f) else 'FAIL'}" for f, label in CHECKS)
                out.append(f"     {e.status}: {r.document} p.{r.page} clause {r.clause or '-'} - \"{r.supporting_text[:90]}\"")
                out.append(f"     checks: {flags}")
                out += [f"     note: {n}" for n in e.notes]
    for cmp in res.comparisons:
        out += ["", f"COMPARISON: {cmp.result}"]
    if res.conflicts:
        out += ["", "CONFLICTS / DIFFERENCES"] + [f"  [{c.classification}] {c.topic}: {c.explanation}" for c in res.conflicts]
    if res.retry_log:
        out += ["", "VERIFIER RETRIES"] + [f"  {r}" for r in res.retry_log]
    if res.abstention:
        a = res.abstention
        out += ["", "WHY NO ANSWER"] + [f"  - {r}" for r in a["reasons"]]
        out += [f"  related: {p['document']} p.{p['page']} clause {p['clause'] or '-'} (score {p['score']}) "
                f"{p['preview'][:60]!r}" for p in a["related_passages"]]
        if a.get("draft_rejected"):
            out.append(f"  rejected draft: {a['draft_rejected'][:120]!r}")
    if res.error:
        out += ["", f"ERROR: {res.error}"]
    t = res.timings
    out += ["", "timings: " + ", ".join(f"{k} {v}" for k, v in t.items()) + f" | tokens {res.tokens}"]
    return "\n".join(out)


def main(argv: list[str] | None = None, client=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", action=GlobPaths)
    ap.add_argument("-q", "--question", required=True)
    ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default=None)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-retrieval", action="store_true", help="ask the model without any documents")
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=None, help="passages given to the model")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):                       # Windows consoles: never crash on a character
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = client or OllamaClient.from_settings()
    if args.model:
        client.model = args.model
    if isinstance(client, OllamaClient):                         # fail fast, before the (slow) indexing
        st = client.status()
        if not st["connected"]:
            print(f"Ollama is not reachable at {st['base_url']}. Start it (ollama serve) and try again.", file=sys.stderr)
            return 2
        if not st["model_available"]:
            print(f"Model '{client.model}' is not installed. Run: ollama pull {client.model}\n"
                  f"Installed: {', '.join(st['models']) or 'none'}", file=sys.stderr)
            return 2
    case_id = ensure_indexed(args.pdfs)
    res = AnswerService(client).answer(case_id, args.question, args.mode, None, False if args.no_rerank else None,
                                       not args.no_retrieval, args.k,
                                       on_status=lambda m: print(f"[{m}]", file=sys.stderr))
    print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
