"""A small autoencoder that flags pages whose extracted text does not look like the rest of THEIR OWN document
(assignment extended feature 8: an autoencoder page-anomaly detector).

Why: a PDF can parse "successfully" and still give a garbage page (a bad scan, a broken text layer, a page that is
only a number, a spreadsheet dump). Such pages produce chunks that retrieve badly and cannot be cited, so the explorer
marks them for a human to look at.

  features   14 numbers per page: size (characters, lines, line length), character mix (letters, digits, capitals,
             punctuation, stray symbols), word shape (length, words without vowels, one-letter words), how much of the
             page the text covers, and the OCR confidence (1.0 for digital text)
  model      14 -> 6 -> 2 -> 6 -> 14, tanh, trained on the standardised features of the document's own pages
             (Adam, seed 42, one thread). Nothing is trained in advance: layouts differ so much between documents that
             a model trained on other documents flags EVERY page of a differently laid-out one (measured, see
             evaluation/page_anomaly_eval.py), while a model fitted on the document itself learns its own normal.
  score      reconstruction error divided by a robust threshold, median + K x MAD of the document's own errors.
             A score above 1 means "unusual page". Unsupervised: no anomalous example is needed, and a few anomalous
             pages inside the training pages do not matter much because the 2-unit bottleneck cannot memorise them.

The design (bottleneck size, the spread floor) was chosen after looking at the injected defects of the evaluation, so its
detection rates are optimistic; the threshold constant K was chosen from clean documents only.

Training uses PyTorch (already a dependency) and takes well under a second for a document of 100-500 pages. A
document with fewer than ``MIN_PAGES`` pages is not scored: there is no "rest of the document" to compare with.
"""
from __future__ import annotations

import re

import numpy as np

from schemas import ParsedPage

SEED = 42
MIN_PAGES = 12
SPREAD_FLOOR = 0.005
HIDDEN = (6, 2)
ROBUST_K = 3.0              # chosen from clean documents only, see evaluation/page_anomaly_eval.py
FEATURES = ["log_chars", "log_lines", "mean_line_len", "std_line_len", "alpha_ratio", "digit_ratio", "upper_ratio",
            "punct_ratio", "symbol_ratio", "mean_word_len", "vowelless_word_ratio", "single_char_word_ratio",
            "coverage", "ocr_confidence"]
_PUNCT = set(".,;:()[]-'\"%/&?!")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_VOWEL = re.compile(r"[aeiouyàáâäèéêëìíîïòóôöùúûüAEIOUY]")


def page_features(page: ParsedPage) -> np.ndarray:
    lines = [l.text for l in page.lines] + [" ".join(r) for t in page.tables for r in t.rows]
    text = "\n".join(lines)
    chars = [c for c in text if not c.isspace()]
    n = max(1, len(chars))
    letters = [c for c in chars if c.isalpha()]
    words = _WORD.findall(text)
    long_words = [w for w in words if len(w) >= 3]
    lens = np.array([len(x) for x in lines], dtype=float) if lines else np.zeros(1)
    area = max(1.0, page.width * page.height)
    covered = sum(max(0.0, l.bbox[2] - l.bbox[0]) * max(0.0, l.bbox[3] - l.bbox[1]) for l in page.lines)
    confs = [l.confidence for l in page.lines if l.confidence is not None]
    return np.array([
        np.log1p(len(chars)), np.log1p(len(lines)), lens.mean() / 100.0, lens.std() / 100.0,
        len(letters) / n, sum(c.isdigit() for c in chars) / n, sum(c.isupper() for c in letters) / max(1, len(letters)),
        sum(c in _PUNCT for c in chars) / n,
        sum(not (c.isalnum() or c in _PUNCT) for c in chars) / n,
        (sum(len(w) for w in words) / len(words) / 10.0) if words else 0.0,
        sum(not _VOWEL.search(w) for w in long_words) / max(1, len(long_words)),
        sum(len(w) == 1 for w in words) / max(1, len(words)),
        min(1.0, covered / area),
        (sum(confs) / len(confs) / 100.0) if confs else 1.0,
    ], dtype=float)


def robust_standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median and a MAD-based spread per feature (a few anomalous pages barely move them); never below SPREAD_FLOOR,
    so that a feature that is constant in a document (OCR confidence 1.0) does not divide by zero."""
    med = np.median(X, axis=0)
    spread = 1.4826 * np.median(np.abs(X - med), axis=0)
    return med, np.maximum(spread, SPREAD_FLOOR)


def reconstruction_errors(X: np.ndarray, seed: int = SEED, epochs: int = 300, lr: float = 1e-2,
                          hidden: tuple[int, int] = HIDDEN, X_eval: np.ndarray | None = None) -> np.ndarray:
    """Fit the autoencoder on the rows of X and return each row's mean squared reconstruction error (in standardised
    units); with ``X_eval`` return the errors of those rows instead (pages the model has not seen).
    Deterministic for a given X and seed."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    X = np.asarray(X, float)
    med, spread = robust_standardise(X)
    z = torch.tensor(np.clip((X - med) / spread, -20, 20), dtype=torch.float32)
    d, (h1, h2) = X.shape[1], hidden
    net = torch.nn.Sequential(torch.nn.Linear(d, h1), torch.nn.Tanh(), torch.nn.Linear(h1, h2), torch.nn.Tanh(),
                              torch.nn.Linear(h2, h1), torch.nn.Tanh(), torch.nn.Linear(h1, d))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.mse_loss(net(z), z).backward()
        opt.step()
    target = z if X_eval is None else torch.tensor(np.clip((np.asarray(X_eval, float) - med) / spread, -20, 20),
                                                  dtype=torch.float32)
    with torch.no_grad():
        return ((net(target) - target) ** 2).mean(dim=1).numpy().astype(float)


def robust_threshold(errors: np.ndarray, k: float = ROBUST_K) -> float:
    med = float(np.median(errors))
    mad = 1.4826 * float(np.median(np.abs(errors - med)))
    return max(med + k * mad, 1e-9)


def scores_from_features(X: np.ndarray, k: float = ROBUST_K, seed: int = SEED, epochs: int = 300) -> np.ndarray:
    err = reconstruction_errors(X, seed=seed, epochs=epochs)
    return err / robust_threshold(err, k)


def score_document(pages: list[ParsedPage], k: float = ROBUST_K) -> list[float] | None:
    """One score per page (> 1: unusual for this document), or None when the document is too short or scoring failed.
    Never raises: a problem here must not stop a document from being indexed."""
    if len(pages) < MIN_PAGES:
        return None
    try:
        return [round(float(s), 3) for s in scores_from_features(np.array([page_features(p) for p in pages]), k)]
    except Exception:  # noqa: BLE001 - this is an extra warning, never a reason to fail
        return None
