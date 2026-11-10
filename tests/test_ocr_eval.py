import random

import numpy as np
import pytest
from PIL import Image

from evaluation import ocr_eval
from ingestion import ocr
from tests.conftest import make_pdf


def naive_edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def test_edit_distance_matches_a_plain_implementation():
    rng = random.Random(1)
    for _ in range(60):
        a = "".join(rng.choice("abc ") for _ in range(rng.randint(0, 25)))
        b = "".join(rng.choice("abc ") for _ in range(rng.randint(0, 25)))
        assert ocr_eval.edit_distance(a, b) == naive_edit_distance(a, b), (a, b)
    words_a, words_b = "the cat sat on the mat".split(), "the cat sit on mat".split()
    assert ocr_eval.edit_distance(words_a, words_b) == naive_edit_distance(words_a, words_b) == 2
    assert ocr_eval.edit_distance("", "abc") == 3 and ocr_eval.edit_distance("abc", "") == 3


def test_compare_scores_identical_wrong_and_reordered_text():
    same = ocr_eval.compare("The fine is 20 000 000 EUR.", "The  fine is\n20 000 000 EUR.")
    assert same["cer"] == 0 and same["wer"] == 0 and same["word_recall"] == 1
    wrong = ocr_eval.compare("The fine is 20 000 000 EUR.", "The fina is 20 000 OOO EUR.")
    assert 0 < wrong["cer"] < 0.2 and wrong["wer"] == pytest.approx(2 / 7, abs=1e-6) and wrong["word_recall"] < 1
    reordered = ocr_eval.compare("alpha beta gamma delta", "gamma delta alpha beta")
    assert reordered["word_recall"] == 1 and reordered["wer"] > 0            # order-free recall ignores the reordering
    assert ocr_eval.compare("a b c", "")["word_recall"] == 0


def test_degradation_is_deterministic_and_changes_the_image():
    img = Image.new("L", (200, 100), 255)
    img.paste(0, (20, 20, 120, 40))
    a, b = ocr_eval.degrade(img, noise=15, blur=1, rotate=1.5, contrast=0.6), ocr_eval.degrade(img, noise=15, blur=1, rotate=1.5, contrast=0.6)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(img))
    assert not np.array_equal(np.asarray(ocr_eval.degrade(img, noise=15, seed=1)), np.asarray(ocr_eval.degrade(img, noise=15, seed=2)))


needs_ocr = pytest.mark.skipif(not ocr.configure_tesseract(), reason="Tesseract is not installed")


def _text_pdf(tmp_path, n_pages=3):
    lines = [(60, 760 - 22 * i, f"Article {i + 1} The controller shall notify the supervisory authority within 72 hours of the breach.", 12)
             for i in range(25)]
    return make_pdf(tmp_path / "doc.pdf", [lines for _ in range(n_pages)])


def test_pick_pages_is_deterministic_and_skips_thin_pages(tmp_path):
    pdf = make_pdf(tmp_path / "x.pdf", [[(60, 700, "short")]] + [[(60, 760 - 20 * i, "x" * 90) for i in range(25)]] * 4)
    pages = ocr_eval.pick_pages(pdf, 2)
    assert pages == ocr_eval.pick_pages(pdf, 2) and 0 not in pages and len(pages) == 2
    assert ocr_eval.pick_pages(pdf, 10) == [1, 2, 3, 4]


@needs_ocr
def test_the_scanned_page_goes_through_the_real_ocr_path_and_is_read_accurately(tmp_path):
    import pymupdf
    pdf = _text_pdf(tmp_path)
    with pymupdf.open(pdf) as src:
        digital = ocr_eval.single_page(src, 0, tmp_path / "one.pdf")
        truth, conf, used = ocr_eval.page_text(digital)
        assert not used and conf is None
        with pymupdf.open(digital) as one:
            scan = ocr_eval.scanned_pdf(one[0], tmp_path / "scan.pdf", dpi=200)
    with pymupdf.open(scan) as scan_doc:
        assert not scan_doc[0].get_text().strip()                            # really image-only
    text, conf, used = ocr_eval.page_text(scan, ocr_dpi=200)
    assert used and conf and conf > 60
    assert ocr_eval.compare(truth, text)["cer"] < 0.05


@needs_ocr
def test_evaluate_reports_every_condition_and_the_fallback(tmp_path):
    result = ocr_eval.evaluate([_text_pdf(tmp_path)], pages_per_doc=1)
    assert list(result["summary"]) == list(ocr_eval.CONDITIONS)
    assert all(r["ocr_used"] for r in result["pages"])
    assert all(s["cer"] < 0.2 and s["word_recall"] > 0.8 for s in result["summary"].values())
    fb = result["fallback"]
    assert fb["scan_only_without_ocr"].startswith("rejected: No text could be extracted")
    assert fb["mixed_without_ocr"]["scanned_page_flagged"] and fb["mixed_without_ocr"]["digital_page_indexed"]
    md = ocr_eval.markdown(result)
    assert "| Condition |" in md and "Fallback with OCR switched off" in md


def test_main_exits_cleanly_when_tesseract_is_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ocr, "configure_tesseract", lambda *a, **k: False)
    assert ocr_eval.main([str(tmp_path / "x.pdf")]) == 2
    assert "Tesseract is not available" in capsys.readouterr().err
