"""Assemble every evaluation result into ONE page, docs/RESULTS.md.

    python -m evaluation.report            # reads evaluation/results/*, writes docs/RESULTS.md and the results block of the README
    python -m evaluation.report --check    # exit 1 if either is not what the result files say (used by the tests)

Nothing is typed by hand: each section is the Markdown file that the corresponding script wrote, so the page cannot
disagree with the measurements. The "Reading the numbers" sentences are produced from summary.json, with the direction
words (raises, lowers) and every number computed, so they cannot go stale when the evaluation is run again. A section
whose script has not been run says which command to run. The README blocks (the headline results near the top and the full results) sit between marker comments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
TARGET = ROOT / "docs" / "RESULTS.md"
README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
H_START, H_END = "<!-- HIGHLIGHTS:START -->", "<!-- HIGHLIGHTS:END -->"
LABELS = ["BM25 only", "Dense only", "Hybrid (BM25 + dense, RRF)", "Hybrid + rerank", "Hybrid + rerank + query decomposition"]

# (file, heading, what it shows, the command that produces it)
SECTIONS = [
    ("summary.md", "Retrieval, answers, citations, manual grades",
     "Dense vs hybrid retrieval, no-retrieval vs retrieval answers, abstention, citation precision, region-level precision.",
     "python -m evaluation.evaluate sample_docs/*.pdf --gen-configs no_retrieval hybrid hybrid_noverify --export-grading"),
    ("diagnostics.md", "Abstention gate: threshold sensitivity (post hoc)",
     "Where a false abstention comes from, and what moving the threshold would cost. Analysis only; nothing was tuned on it.",
     "python -m evaluation.diagnostics sample_docs/*.pdf"),
    ("ocr_eval.md", "OCR accuracy and fallback (extended feature 1)", "Character and word error rates on degraded scans of real pages.",
     "python -m evaluation.ocr_eval sample_docs/*.pdf"),
    ("table_eval.md", "Table extraction accuracy (extended feature 2)", "Cell accuracy against a hand-written ground truth.",
     "python -m evaluation.table_eval sample_docs/*.pdf"),
    ("page_anomaly_eval.md", "Autoencoder page-anomaly detector (extended feature 8)",
     "AUROC against a plain baseline, false alarms, and why nothing is trained in advance.",
     "python -m evaluation.page_anomaly_eval sample_docs/*.pdf"),
    ("ablation_tables.md", "Ablation: text only vs text + tables (extended feature 7)", "Retrieval of table cells with and without table extraction.",
     "python -m evaluation.ablation_tables"),
    ("latency_report.md", "Latency and cost (extended feature 11)", "Seconds and tokens per question, and what the sufficiency gate saves.",
     "python -m evaluation.latency_report"),
    ("bench_indexing.md", "Measured optimisation: the embedding cache (extended feature 11)", "Cold against warm embedding.",
     "python -m evaluation.bench_indexing sample_docs/ePrivacy_Directive_2002_58.pdf"),
]


def demote(md: str, by: int = 2) -> str:
    """Move the headings of an included file down so that they nest under this page's section heading."""
    return "\n".join(("#" * by + line) if line.startswith("#") else line for line in md.splitlines())


def _move(a: float, b: float) -> str:
    return "raises" if b > a + 1e-9 else "lowers" if b < a - 1e-9 else "leaves unchanged"


def reading(summary: dict) -> str:
    """Plain sentences about the results, with every number and every direction computed from summary.json."""
    r, gen = summary.get("retrieval") or {}, summary.get("generation") or {}
    bm, de, hy, rr, dc = (r.get(k) for k in LABELS)
    out: list[str] = []
    if bm and de and hy:
        weakest = min(((v["recall@5"], k) for k, v in r.items()))[1]
        out.append(f"**Retrieval configurations.** recall@5 is {de['recall@5']:.2f} for dense only, {bm['recall@5']:.2f} for BM25 only and "
                   f"{hy['recall@5']:.2f} for hybrid (MRR {de['mrr']:.2f}, {bm['mrr']:.2f}, {hy['mrr']:.2f}). The weakest configuration on "
                   f"recall@5 is \"{weakest}\". With {summary.get('meta', {}).get('n_answerable', '?')} answerable questions, differences "
                   "below about 0.10 are noise.")
    if hy and rr:
        out.append(f"**Reranking** {_move(hy['recall@3'], rr['recall@3'])} recall@3 ({hy['recall@3']:.2f} to {rr['recall@3']:.2f}), "
                   f"{_move(hy['recall@5'], rr['recall@5'])} recall@5 ({hy['recall@5']:.2f} to {rr['recall@5']:.2f}) and "
                   f"{_move(hy['mrr'], rr['mrr'])} MRR ({hy['mrr']:.2f} to {rr['mrr']:.2f}); the retrieval latency goes from "
                   f"{hy['latency_ms']:.0f} ms to {rr['latency_ms']:.0f} ms. The gate and the confidence use the reranker's score, "
                   "which is why it stays in the pipeline either way.")
    if rr and dc:
        mh = {k: v for k, v in (summary.get("by_category") or {}).items() if k.startswith("multihop / ")}
        a, b = mh.get("multihop / Hybrid + rerank"), mh.get("multihop / Hybrid + rerank + query decomposition")
        extra = f" and multi-hop full@5 ({a['full@5']:.2f} to {b['full@5']:.2f})" if a and b else ""
        out.append(f"**Query decomposition** {_move(rr['full@5'], dc['full@5'])} full@5 ({rr['full@5']:.2f} to {dc['full@5']:.2f}){extra}. "
                   "It was designed after looking at multi-hop failures of a similar prototype, so any gain here is optimistic.")
    base, hyb = gen.get("no_retrieval"), gen.get("hybrid")
    manual = (summary.get("manual") or {}).get("configs", {})
    if base and hyb:
        if "hybrid" in manual and "no_retrieval" in manual:
            m0, m1 = manual["no_retrieval"], manual["hybrid"]
            out.append(f"**Retrieval against no retrieval, graded by hand ({summary['manual']['grader']}).** Answerable questions score "
                       f"{m1['answerable_mean_score']:.2f} with retrieval and {m0['answerable_mean_score']:.2f} without; unanswerable questions "
                       f"are handled correctly {m1['unanswerable_handled_correctly']:.2f} against {m0['unanswerable_handled_correctly']:.2f}. "
                       f"The price is time: {hyb['mean_total_s']:.0f} s per question against {base['mean_total_s']:.0f} s.")
        else:
            out.append(f"**Retrieval against no retrieval (automatic proxy; not yet graded by hand).** Key facts present: "
                       f"{hyb['answer_correct_proxy']:.2f} with retrieval, {base['answer_correct_proxy']:.2f} without; mean time "
                       f"{hyb['mean_total_s']:.0f} s against {base['mean_total_s']:.0f} s.")
    if hyb and hyb.get("citations_verified_rate") is not None:
        s = (f"**Citations.** {hyb['citations_verified_rate']:.2f} of the emitted citations pass the validator; citation precision against the "
             f"labelled evidence is {hyb['citation_precision']:.2f}, so {1 - hyb['citation_precision']:.0%} of the cited passages are not among "
             "the labelled evidence items (they may be valid alternatives; that is not examined).")
        reg = summary.get("region") or {}
        if reg.get("n_citations"):
            s += (f" At region level the precision is {reg['region_precision']:.2f} (page level {reg['page_precision']:.2f}); "
                  f"{reg['page_hits_outside_the_region']} citation(s) are on the right page but outside the labelled region.")
        out.append(s)
    if hyb:
        ids = hyb.get("answerable_refused_ids")
        n = hyb["n_answerable"]
        k = len(ids) if ids is not None else round((hyb.get("false_abstention_rate") or 0) * n)
        out.append(f"**False abstentions.** The hybrid configuration refused {k} of {n} answerable questions"
                   + (f" ({', '.join(ids)})." if ids else ".") + " See the diagnostics section of docs/RESULTS.md for the reason.")
    nv = gen.get("hybrid_noverify")
    if hyb and nv and nv.get("unsupported_claim_rate") is not None:
        out.append(f"**Without the verifier** {nv['unsupported_claim_rate']:.0%} of the claims shown are not verified, against "
                   f"{(hyb.get('unsupported_claim_rate') or 0):.0%} with it; the false-abstention rate is {nv['false_abstention_rate']:.2f} without "
                   f"and {hyb['false_abstention_rate']:.2f} with the verifier; mean time {nv['mean_total_s']:.0f} s and {hyb['mean_total_s']:.0f} s.")
    return "\n".join(f"* {b}" for b in out)


def highlights(summary: dict) -> str | None:
    """The few numbers a reader should see first, as two small tables, all taken from summary.json."""
    gen, r = summary.get("generation") or {}, summary.get("retrieval") or {}
    base, hyb = gen.get("no_retrieval"), gen.get("hybrid")
    manual = (summary.get("manual") or {}).get("configs", {})
    region = summary.get("region") or {}
    out: list[str] = []
    if base and hyb:
        def f(v, d=2):
            return "none" if v is None else f"{v:.{d}f}"
        m0, m1 = manual.get("no_retrieval"), manual.get("hybrid")
        n = (summary.get("meta") or {}).get("n_questions", "?")
        rows = []
        if m0 and m1:
            rows += [("Answerable questions: mean score graded by hand (0 to 1)", f(m0["answerable_mean_score"]), f(m1["answerable_mean_score"])),
                     ("Unanswerable questions handled correctly, graded by hand", f(m0["unanswerable_handled_correctly"]), f(m1["unanswerable_handled_correctly"]))]
        else:
            rows.append(("Answerable questions: key facts present (automatic proxy, not graded by hand)", f(base.get("answer_correct_proxy")), f(hyb.get("answer_correct_proxy"))))
        if "abstention_accuracy" in base and "abstention_accuracy" in hyb:
            rows.append((f"Right decision to answer or to refuse (automatic, all {n} questions)", f(base["abstention_accuracy"]), f(hyb["abstention_accuracy"])))
        if hyb.get("citations_verified_rate") is not None:
            rows.append(("Citations that pass the validator", "none", f(hyb["citations_verified_rate"])))
        if region.get("n_citations"):
            rows.append(("Citations whose quote lies in the labelled clause (region level)", "none", f(region["region_precision"])))
        if "mean_total_s" in base and "mean_total_s" in hyb:
            rows.append(("Seconds per question (local model on a CPU)", f"{base['mean_total_s']:.0f}", f"{hyb['mean_total_s']:.0f}"))
        out += ["| | No retrieval (the model alone) | Lexora AI (hybrid, rerank, verifier) |", "|---|---|---|"]
        out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    if r:
        if out:
            out.append("")
        out += ["| Retrieval configuration | recall@5 | MRR |", "|---|---|---|"]
        out += [f"| {label} | {m['recall@5']:.2f} | {m['mrr']:.2f} |" for label, m in r.items()]
    return "\n".join(out) if out else None


def summary_section(results: Path = RESULTS) -> str | None:
    """summary.md followed by the generated 'Reading the numbers', or None when there is no summary yet."""
    f = results / "summary.md"
    if not f.is_file():
        return None
    text = f.read_text(encoding="utf-8").strip()
    try:
        sentences = reading(json.loads((results / "summary.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        sentences = ""
    return text + ("\n\n### Reading the numbers\n\n" + sentences if sentences else "")


def readme_block(results: Path = RESULTS) -> str | None:
    """What goes between the markers of the README: the summary with its headings one level deeper."""
    text = summary_section(results)
    return None if text is None else demote(text, 1)


def _highlights_block(results: Path) -> str | None:
    try:
        return highlights(json.loads((results / "summary.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        return None


def _swap(text: str, start: str, end: str, block: str) -> str:
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n\n{block}\n\n{end}{tail}"


def update_readme(readme: Path, results: Path = RESULTS) -> bool:
    """Replace the text between the RESULTS markers (and between the HIGHLIGHTS markers when the README has them).
    False if the README has no RESULTS markers or there is no summary."""
    block, text = readme_block(results), readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if block is None or START not in text or END not in text:
        return False
    text = _swap(text, START, END, block)
    top = _highlights_block(results)
    if top is not None and H_START in text and H_END in text:
        text = _swap(text, H_START, H_END, top)
    readme.write_text(text, encoding="utf-8", newline="\n")
    return True


def readme_is_current(readme: Path, results: Path = RESULTS) -> bool:
    block, text = readme_block(results), readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if block is None or START not in text or END not in text or f"{START}\n\n{block}\n\n{END}" not in text:
        return False
    top = _highlights_block(results)
    return top is None or H_START not in text or f"{H_START}\n\n{top}\n\n{H_END}" in text


def build(results: Path = RESULTS) -> str:
    out = ["# Results", "",
           "Generated by `python -m evaluation.report` from the files in `evaluation/results/`. Do not edit by hand: run the "
           "command under a heading to refresh that section, then run this script again.", ""]
    for filename, heading, shows, command in SECTIONS:
        out += [f"## {heading}", "", f"_{shows}_", ""]
        f = results / filename
        if filename == "summary.md" and f.is_file():
            out += [summary_section(results), ""]               # its "###" headings nest under this page's "##" section
        elif f.is_file():
            out += [f.read_text(encoding="utf-8").strip(), ""]
        else:
            out += [f"**Not run yet.** Command: `{command}`", ""]
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="do not write; fail if the page is out of date")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--target", type=Path, default=TARGET)
    ap.add_argument("--readme", type=Path, default=README)
    a = ap.parse_args(argv)
    page = build(a.results)
    if a.check:
        ok_page = a.target.is_file() and a.target.read_text(encoding="utf-8") == page
        ok_readme = readme_is_current(a.readme, a.results)
        print("docs/RESULTS.md is " + ("up to date" if ok_page else "OUT OF DATE") + "; the README results block is "
              + ("up to date" if ok_readme else "OUT OF DATE") + (" (run python -m evaluation.report)" if not (ok_page and ok_readme) else ""))
        return 0 if ok_page and ok_readme else 1
    a.target.parent.mkdir(parents=True, exist_ok=True)
    a.target.write_text(page, encoding="utf-8", newline="\n")
    if not update_readme(a.readme, a.results):
        print(f"note: {a.readme.name} has no {START} / {END} markers or there is no summary yet; the README was not changed")
    missing = [s[0] for s in SECTIONS if not (a.results / s[0]).is_file()]
    print(f"wrote {a.target}" + (f"; sections without results yet: {', '.join(missing)}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
