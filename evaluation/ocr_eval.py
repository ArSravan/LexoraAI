"""OCR accuracy on a sample, and the fallback when OCR is not available (assignment extended feature 1).

    python -m evaluation.ocr_eval sample_docs/*.pdf                  # about 1-2 minutes, no LLM, needs Tesseract

Method. A few real digital pages are taken from the given PDFs. Their text, extracted by the normal pipeline, is the
ground truth. The same page is rasterised, degraded in a fixed way (seed 42), saved as an image-only one-page PDF, and
read back by the SAME code path that handles an uploaded scan (``parse_pdf`` detects the missing text layer and calls
Tesseract). The two texts are compared with:

  CER          character error rate = edit distance / ground-truth characters   (lower is better)
  WER          word error rate (edit distance over words)                       (lower is better)
  word recall  share of ground-truth words that occur in the OCR text, ignoring order (robust to reading-order
               differences between the digital and the OCR extraction)

Fallback. With OCR disabled the scanned page must be flagged and never silently dropped; a PDF that is only scans must
be rejected with a clear message instead of being indexed empty.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import pymupdf
from PIL import Image, ImageEnhance, ImageFilter

from ingestion import ocr
from ingestion.pdf_parser import PDFValidationError, parse_pdf

RESULTS = Path(__file__).resolve().parent / "results"
SEED = 42

# name -> how the page image is degraded before it is "scanned"
CONDITIONS: dict[str, dict] = {
    "clean scan, 200 dpi": {"dpi": 200},
    "noisy scan": {"dpi": 200, "noise": 20.0},
    "skewed and blurred": {"dpi": 200, "rotate": 1.5, "blur": 1.0},
    "low contrast, JPEG 35, 150 dpi": {"dpi": 150, "contrast": 0.55, "jpeg": 35},
}


# ------------------------------------------------------------------------------ text comparison
def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or lists of words), vectorised row by row."""
    if len(a) < len(b):
        a, b = b, a
    if not len(b):
        return len(a)
    vocab: dict = {}
    bb = np.array([vocab.setdefault(x, len(vocab)) for x in b], dtype=np.int64)
    aa = [vocab.setdefault(x, len(vocab)) for x in a]
    idx = np.arange(len(bb) + 1)
    prev = idx.copy()
    for i, x in enumerate(aa, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        cur[1:] = np.minimum(prev[:-1] + (bb != x), prev[1:] + 1)      # substitute / delete
        prev = np.minimum.accumulate(cur - idx) + idx                  # insertions: cur[j] <= cur[j-1] + 1
    return int(prev[-1])


def normalise(text: str) -> str:
    return " ".join(text.split())


def compare(truth: str, ocr_text: str) -> dict[str, float]:
    t, o = normalise(truth), normalise(ocr_text)
    tw, ow = t.split(), o.split()
    have, want = Counter(ow), Counter(tw)
    return {"cer": edit_distance(t, o) / max(1, len(t)), "wer": edit_distance(tw, ow) / max(1, len(tw)),
            "word_recall": sum(min(n, have[w]) for w, n in want.items()) / max(1, len(tw)), "n_chars": len(t)}


# ------------------------------------------------------------------------------ making the scans
def degrade(img: Image.Image, *, noise: float = 0.0, blur: float = 0.0, rotate: float = 0.0, contrast: float = 1.0,
            seed: int = SEED) -> Image.Image:
    img = img.convert("L")
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if rotate:
        img = img.rotate(rotate, resample=Image.BICUBIC, fillcolor=255)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + np.random.RandomState(seed).normal(0.0, noise, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def scanned_pdf(page: "pymupdf.Page", out: Path, *, dpi: int, jpeg: int | None = None, **degradation) -> Path:
    """An image-only one-page PDF of ``page`` (same size in points), the way a scanner would produce it."""
    img = degrade(ocr.render_page(page, dpi), **degradation)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg) if jpeg else img.save(buf, format="PNG")
    doc = pymupdf.open()
    pg = doc.new_page(width=page.rect.width, height=page.rect.height)
    pg.insert_image(pg.rect, stream=buf.getvalue())
    doc.save(out)
    doc.close()
    return out


def single_page(src: "pymupdf.Document", index: int, out: Path) -> Path:
    doc = pymupdf.open()
    doc.insert_pdf(src, from_page=index, to_page=index)
    doc.save(out)
    doc.close()
    return out


def page_text(pdf: Path, **kw) -> tuple[str, float | None, bool]:
    """(text, mean OCR confidence, was it OCR'd) of a one-page PDF read through the real pipeline."""
    parsed = parse_pdf(pdf, "ocr-eval", **kw)
    p = parsed.pages[0]
    return " ".join(l.text for l in p.lines), p.ocr_confidence, p.method == "ocr"


def pick_pages(pdf: Path, n: int, min_chars: int = 1500) -> list[int]:
    """n text-rich pages, spread evenly over the document (deterministic)."""
    with pymupdf.open(pdf) as d:
        rich = [i for i in range(d.page_count) if len(d[i].get_text().strip()) >= min_chars]
    if len(rich) <= n:
        return rich
    return [rich[round(k * (len(rich) - 1) / max(1, n - 1))] for k in range(n)]


# ------------------------------------------------------------------------------ the evaluation
def evaluate(pdfs: list[Path], pages_per_doc: int = 2, dpi_override: int | None = None) -> dict:
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for pdf in pdfs:
            with pymupdf.open(pdf) as src:
                for pi in pick_pages(pdf, pages_per_doc):
                    digital = single_page(src, pi, tmp / "digital.pdf")
                    truth, _, _ = page_text(digital, tables_enabled=False)
                    with pymupdf.open(digital) as one:
                        for name, spec in CONDITIONS.items():
                            spec = dict(spec)
                            dpi = dpi_override or spec.pop("dpi")
                            spec.pop("dpi", None)
                            scan = scanned_pdf(one[0], tmp / "scan.pdf", dpi=dpi, **spec)
                            text, conf, used_ocr = page_text(scan, ocr_dpi=dpi, tables_enabled=False)
                            rows.append({"document": pdf.name, "page": pi + 1, "condition": name, "ocr_used": used_ocr,
                                         "ocr_confidence": conf, **compare(truth, text)})
        fallback = fallback_check(pdfs[0], tmp)
    summary = {}
    for name in CONDITIONS:
        rs = [r for r in rows if r["condition"] == name]
        if rs:
            summary[name] = {k: round(mean(r[k] for r in rs), 4) for k in ("cer", "wer", "word_recall", "ocr_confidence")}
            summary[name]["pages"] = len(rs)
    return {"seed": SEED, "pages_per_document": pages_per_doc, "documents": [p.name for p in pdfs],
            "summary": summary, "pages": rows, "fallback": fallback}


def fallback_check(pdf: Path, tmp: Path) -> dict:
    """What happens when OCR is switched off: the scan is flagged, never silently dropped or indexed empty."""
    with pymupdf.open(pdf) as src:
        pi = pick_pages(pdf, 1)[0]
        digital = single_page(src, pi, tmp / "fb_digital.pdf")
        with pymupdf.open(digital) as one:
            only_scan = scanned_pdf(one[0], tmp / "fb_scan.pdf", dpi=150)
            mixed = pymupdf.open()
            for part in (digital, only_scan):
                with pymupdf.open(part) as src_part:
                    mixed.insert_pdf(src_part)
            mixed.save(tmp / "fb_mixed.pdf")
            mixed.close()
    out = {}
    try:
        parse_pdf(only_scan, "fb", ocr_enabled=False)
        out["scan_only_without_ocr"] = "indexed (unexpected)"
    except PDFValidationError as e:
        out["scan_only_without_ocr"] = f"rejected: {e}"
    doc = parse_pdf(tmp / "fb_mixed.pdf", "fb2", ocr_enabled=False)
    scan_page = doc.pages[1]
    out["mixed_without_ocr"] = {"scanned_page_flagged": scan_page.needs_ocr, "page_warning": scan_page.warning,
                                "document_warning": next((w for w in doc.warnings if "need OCR" in w), ""),
                                "digital_page_indexed": doc.pages[0].char_count > 0}
    return out


def markdown(result: dict) -> str:
    rows = ["| Condition | Pages | CER | WER | Word recall | Mean OCR confidence |", "|---|---|---|---|---|---|"]
    for name, s in result["summary"].items():
        rows.append(f"| {name} | {s['pages']} | {s['cer']:.3f} | {s['wer']:.3f} | {s['word_recall']:.3f} | {s['ocr_confidence']:.1f} % |")
    fb = result["fallback"]
    mixed = fb["mixed_without_ocr"]
    return "\n".join([
        f"OCR accuracy on {len({(r['document'], r['page']) for r in result['pages']})} real pages from "
        f"{', '.join(result['documents'])} (seed {result['seed']}). Ground truth: the digital text of the same page.",
        "", *rows, "",
        "Fallback with OCR switched off:",
        f"* a PDF that consists only of scans: {fb['scan_only_without_ocr']}",
        f"* a mixed PDF: scanned page flagged = {mixed['scanned_page_flagged']}, digital page still indexed = "
        f"{mixed['digital_page_indexed']}; document warning: \"{mixed['document_warning']}\"", ""])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--pages-per-doc", type=int, default=2)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if not ocr.configure_tesseract():
        print("Tesseract is not available: install it (see the README) and run again.", file=sys.stderr)
        return 2
    result = evaluate(a.pdfs, a.pages_per_doc)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ocr_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = markdown(result)
    (a.out / "ocr_eval.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
