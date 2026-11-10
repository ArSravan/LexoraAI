"""Evaluation of the autoencoder page-anomaly detector (assignment extended feature 8).

    python -m evaluation.page_anomaly_eval sample_docs/*.pdf              # about 2-3 minutes, no LLM

The detector (ingestion/page_anomaly.py) is fitted on each document's OWN pages and flags pages that do not look like
the rest. Three experiments, all with fixed seeds:

  1. Clean documents. Each real document is scored as it is. Every page flagged is a FALSE alarm (the documents are
     intact). This is the only experiment used to choose the threshold constant K, so no anomalous page influences it.
  2. Injected defects. In each document 10 % of the pages (at least 2) are replaced by a defective version, the
     detector is fitted on the whole (now contaminated) document and every page is scored. Defects:
       garbled          35 % of the letters replaced at random (a broken text layer)
       symbol noise     random symbols inserted (typical OCR junk)
       near empty       a page that is only a few characters (a stray page number)
       number dump      lines of digits only (a spreadsheet pasted in)
       repeated line    one line repeated 40 times (an extraction bug)
       bad scan (OCR)   the real page rasterised, degraded just short of unreadable and read by the real Tesseract
                        (only when Tesseract is installed; a page that yields no text at all is skipped)
     Reported per defect: AUROC (1.0 = every defective page scores above every intact one) for the autoencoder and for a
     plain baseline (largest robust z-score of any feature), and the share flagged at the chosen K.
  3. Cross-document. A model fitted on the OTHER documents scores a held-out one with the same threshold rule. The share
     of its intact pages flagged shows why nothing is trained in advance.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import tempfile
from pathlib import Path
from statistics import mean

import numpy as np
import pymupdf

from evaluation import ocr_eval
from ingestion import ocr
from ingestion.page_anomaly import (MIN_PAGES, ROBUST_K, page_features, reconstruction_errors, robust_standardise,
                                    robust_threshold)
from ingestion.pdf_parser import parse_pdf
from schemas import ParsedPage, TextLine

RESULTS = Path(__file__).resolve().parent / "results"
SEEDS = [42, 43, 44]
KS = [3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
MAX_FALSE_ALARM_RATE = 0.10
CONTAMINATION = 0.10
KINDS = ["garbled", "symbol noise", "near empty", "number dump", "repeated line", "bad scan (OCR)"]


# ------------------------------------------------------------------------------ making defective pages
def _with_lines(page: ParsedPage, lines: list[TextLine]) -> ParsedPage:
    return page.model_copy(update={"lines": lines, "tables": [], "char_count": sum(len(l.text) for l in lines)})


def _line(src: TextLine, text: str) -> TextLine:
    return src.model_copy(update={"text": text, "is_footnote": False})


def perturb(page: ParsedPage, kind: str, rng: random.Random) -> ParsedPage:
    lines = page.lines or [TextLine(text="x", page_index=page.page_index, bbox=(50, 50, 100, 60))]
    if kind == "garbled":
        def garble(s: str) -> str:
            return "".join(rng.choice(string.ascii_lowercase if c.islower() else string.ascii_uppercase)
                           if c.isalpha() and rng.random() < 0.35 else c for c in s)
        return _with_lines(page, [_line(l, garble(l.text)) for l in lines])
    if kind == "symbol noise":
        def noise(s: str) -> str:
            out = []
            for c in s:
                out.append(c)
                if rng.random() < 0.12:
                    out.append(rng.choice("~^|\\#@<>{}_*="))
            return "".join(out)
        return _with_lines(page, [_line(l, noise(l.text)) for l in lines])
    if kind == "near empty":
        return _with_lines(page, [_line(lines[0], lines[0].text[:8])])
    if kind == "number dump":
        return _with_lines(page, [_line(l, " ".join(str(rng.randint(0, 99999)) for _ in range(9))) for l in lines])
    if kind == "repeated line":
        return _with_lines(page, [_line(lines[0], lines[0].text)] * 40)
    raise ValueError(kind)


def bad_scan_pool(pdf: Path, pages: list[ParsedPage], n: int, tmp: Path) -> dict[int, ParsedPage]:
    """Up to ``n`` pages (index -> page) of the document as a badly scanned copy read by the real OCR."""
    out: dict[int, ParsedPage] = {}
    rich = [i for i, p in enumerate(pages) if p.char_count > 1500]
    picks = [rich[round(k * (len(rich) - 1) / max(1, n - 1))] for k in range(min(n, len(rich)))] if rich else []
    with pymupdf.open(pdf) as src:
        for pi in dict.fromkeys(picks):
            one = ocr_eval.single_page(src, pi, tmp / "one.pdf")
            with pymupdf.open(one) as d:
                scan = ocr_eval.scanned_pdf(d[0], tmp / "scan.pdf", dpi=150, noise=26.0, blur=1.0, rotate=1.5)
            try:
                page = parse_pdf(scan, "bad", tables_enabled=False, ocr_dpi=150).pages[0]
            except Exception:  # noqa: BLE001 - nothing readable at all: skipped
                continue
            out[pi] = page.model_copy(update={"page_index": pi, "page_number": pi + 1})
    return out


# ------------------------------------------------------------------------------ metrics
def auroc(normal: np.ndarray, anomalous: np.ndarray) -> float:
    """P(a defective page scores higher than an intact one), ties counting one half (Mann-Whitney U)."""
    n, a = np.asarray(normal), np.asarray(anomalous)
    if not len(n) or not len(a):
        return float("nan")
    return float(((a[:, None] > n[None, :]).sum() + 0.5 * (a[:, None] == n[None, :]).sum()) / (len(a) * len(n)))


def zscore_scores(X: np.ndarray) -> np.ndarray:
    med, spread = robust_standardise(X)
    return np.abs((X - med) / spread).max(axis=1)


def choose_k(clean_false_alarm: dict[float, float], limit: float = MAX_FALSE_ALARM_RATE) -> float:
    """The smallest K whose false-alarm rate on CLEAN documents is at most ``limit``."""
    ok = [k for k in sorted(clean_false_alarm) if clean_false_alarm[k] <= limit]
    return ok[0] if ok else max(clean_false_alarm)


def load_pages(pdf: Path) -> list[ParsedPage]:
    return parse_pdf(pdf, "ae-eval", ocr_enabled=False, tables_enabled=False).pages


def feature_matrix(pages: list[ParsedPage]) -> np.ndarray:
    return np.array([page_features(p) for p in pages])


# ------------------------------------------------------------------------------ the evaluation
def evaluate(pdfs: list[Path], with_ocr: bool | None = None) -> dict:
    with_ocr = ocr.configure_tesseract() if with_ocr is None else with_ocr
    docs = {p: load_pages(p) for p in pdfs}
    docs = {p: ps for p, ps in docs.items() if len(ps) >= MIN_PAGES}
    feats = {p: feature_matrix(ps) for p, ps in docs.items()}

    # 1. clean documents -> K
    clean_err = {p: reconstruction_errors(feats[p]) for p in docs}
    clean = {k: round(mean(float((e > robust_threshold(e, k)).mean()) for e in clean_err.values()), 4) for k in KS}
    k_star = choose_k(clean)

    # 2. injected defects
    with tempfile.TemporaryDirectory() as tmp:
        pools = {p: bad_scan_pool(p, docs[p], 8, Path(tmp)) for p in docs} if with_ocr else {}
        agg = {kind: {"auc_ae": [], "auc_z": [], "found": {k: [] for k in KS}, "fa": {k: [] for k in KS}, "n": 0} for kind in KINDS}
        for p, pages in docs.items():
            n = len(pages)
            for seed in SEEDS:
                rng = random.Random(seed)
                chosen = sorted(rng.sample(range(n), max(2, round(CONTAMINATION * n))))
                for kind in KINDS:
                    if kind == "bad scan (OCR)":
                        pool = pools.get(p, {})
                        if not pool:
                            continue
                        idx = sorted(random.Random(seed).sample(list(pool), min(len(pool), len(chosen))))
                        replaced = {i: pool[i] for i in idx}
                    else:
                        idx = chosen
                        replaced = {i: perturb(pages[i], kind, random.Random(seed * 1000 + i)) for i in idx}
                    X = feats[p].copy()
                    for i, pg in replaced.items():
                        X[i] = page_features(pg)
                    err = reconstruction_errors(X, seed=seed)
                    mask = np.zeros(n, bool)
                    mask[idx] = True
                    a = agg[kind]
                    a["n"] += int(mask.sum())
                    a["auc_ae"].append(auroc(err[~mask], err[mask]))
                    z = zscore_scores(X)
                    a["auc_z"].append(auroc(z[~mask], z[mask]))
                    for k in KS:
                        thr = robust_threshold(err, k)
                        a["found"][k].append(float((err[mask] > thr).mean()))
                        a["fa"][k].append(float((err[~mask] > thr).mean()))
    rows = [{"defect": kind, "pages": a["n"], "auroc_autoencoder": round(mean(a["auc_ae"]), 4),
             "auroc_zscore": round(mean(a["auc_z"]), 4),
             "detected": {str(k): round(mean(a["found"][k]), 4) for k in KS},
             "false_alarms_on_intact_pages": {str(k): round(mean(a["fa"][k]), 4) for k in KS}}
            for kind, a in agg.items() if a["auc_ae"]]

    # 3. cross-document
    cross = {}
    for held, X in feats.items():
        others = [q for q in feats if q != held]
        if not others:
            continue
        tr = np.vstack([feats[q] for q in others])
        err_tr = reconstruction_errors(tr)
        err_held = reconstruction_errors(tr, X_eval=X)
        cross[held.name] = round(float((err_held > robust_threshold(err_tr, k_star)).mean()), 4)
    return {"seeds": SEEDS, "documents": [p.name for p in docs], "pages": {p.name: len(ps) for p, ps in docs.items()},
            "contamination": CONTAMINATION, "tesseract": bool(with_ocr), "ks": KS,
            "clean_false_alarm_rate": {str(k): v for k, v in clean.items()}, "chosen_k": k_star,
            "max_false_alarm_rate": MAX_FALSE_ALARM_RATE, "defects": rows, "cross_document_false_alarm_rate": cross}


def markdown(r: dict) -> str:
    k = str(r["chosen_k"])
    head = ["| Defect | Defective pages | AUROC autoencoder | AUROC z-score baseline | Detected at K | False alarms on intact pages |",
            "|---|---|---|---|---|---|"]
    for d in r["defects"]:
        head.append(f"| {d['defect']} | {d['pages']} | {d['auroc_autoencoder']:.3f} | {d['auroc_zscore']:.3f} | "
                    f"{d['detected'][k]:.0%} | {d['false_alarms_on_intact_pages'][k]:.1%} |")
    curve = ["| K | " + " | ".join(f"{x:g}" for x in r["ks"]) + " |", "|---|" + "---|" * len(r["ks"]),
             "| false alarms on clean documents | " + " | ".join(f"{r['clean_false_alarm_rate'][str(x)]:.1%}" for x in r["ks"]) + " |"]
    cross = "; ".join(f"{d}: {v:.0%}" for d, v in r["cross_document_false_alarm_rate"].items())
    note = "" if r["tesseract"] else "\n(The bad-scan row is missing because Tesseract is not installed.)"
    pages = ", ".join(f"{d} ({n} pages)" for d, n in r["pages"].items())
    return (f"Autoencoder page-anomaly detector, fitted on each document's own pages. Documents: {pages}. Seeds {r['seeds']}; "
            f"{r['contamination']:.0%} of the pages of a document are replaced by the defective version.\n\n" + "\n".join(head) +
            f"\n\nThreshold = median + K x MAD of the document's own reconstruction errors. K = {r['chosen_k']:g} is the smallest "
            f"value whose false-alarm rate on the CLEAN documents is at most {r['max_false_alarm_rate']:.0%}; no defective page was "
            "used to choose it.\n\n" + "\n".join(curve) +
            f"\n\nCross-document (a model fitted on the other documents, same K): share of INTACT pages flagged: {cross}.{note}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--no-ocr", action="store_true", help="skip the bad-scan defect")
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    result = evaluate(a.pdfs, with_ocr=False if a.no_ocr else None)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "page_anomaly_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = markdown(result)
    (a.out / "page_anomaly_eval.md").write_text(md, encoding="utf-8")
    print(md)
    if result["chosen_k"] != ROBUST_K:
        print(f"NOTE: the module uses K = {ROBUST_K:g}; this run chose K = {result['chosen_k']:g}. "
              "Set ROBUST_K in ingestion/page_anomaly.py if you want to follow it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
