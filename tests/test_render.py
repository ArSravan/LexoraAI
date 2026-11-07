import io

import numpy as np
import pytest
from PIL import Image, ImageChops

from services.render import render_page_png
from tests.conftest import make_pdf

QUOTE = "administrative fines up to 20 000 000 EUR"


@pytest.fixture
def pdf(tmp_path):
    return str(make_pdf(tmp_path / "p.pdf", [
        [(60, 760, "Article 83"), (60, 742, f"Infringements shall be subject to {QUOTE}.")],
        [(60, 760, "Another page without the sentence.")]]))


def image(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


def yellow_pixels(img) -> int:
    a = np.asarray(img)
    return int(((a[..., 0] > 200) & (a[..., 1] > 170) & (a[..., 2] < 120)).sum())


def blue_pixels(img) -> int:
    a = np.asarray(img)
    return int(((a[..., 2] > 200) & (a[..., 0] < 60)).sum())


def test_returns_a_png_of_the_right_size(pdf):
    png, info = render_page_png(pdf, 0, zoom=1.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = image(png).size                                   # A4 is 595.28 x 841.89 points
    assert abs(w - 595) <= 1 and abs(h - 842) <= 1
    assert info == {"matches": 0, "highlighted": False, "region": False}
    w2, h2 = image(render_page_png(pdf, 0, zoom=2.0)[0]).size
    assert abs(w2 - 2 * w) <= 1 and abs(h2 - 2 * h) <= 1


def test_the_quote_is_found_and_painted_yellow(pdf):
    plain, _ = render_page_png(pdf, 0)
    png, info = render_page_png(pdf, 0, quote=QUOTE)
    assert info["highlighted"] and info["matches"] >= 1
    assert yellow_pixels(image(png)) > 500 and yellow_pixels(image(plain)) == 0


def test_a_long_or_wrapped_quote_falls_back_to_its_beginning(pdf):
    long_quote = f"{QUOTE}, or in the case of an undertaking, up to 4 % of the total worldwide annual turnover"
    _, info = render_page_png(pdf, 0, quote=long_quote)
    assert info["highlighted"]


def test_a_quote_that_is_not_on_the_page_highlights_nothing(pdf):
    png, info = render_page_png(pdf, 1, quote=QUOTE)
    assert info["matches"] == 0 and not info["highlighted"] and yellow_pixels(image(png)) == 0
    assert render_page_png(pdf, 0, quote="")[1]["matches"] == 0 and render_page_png(pdf, 0, quote="short")[1]["matches"] == 0


def test_the_clause_region_is_drawn_in_blue(pdf):
    plain, _ = render_page_png(pdf, 0)
    png, info = render_page_png(pdf, 0, bbox=[50, 40, 400, 80])
    assert info["region"] and ImageChops.difference(image(plain), image(png)).getbbox() is not None
    assert blue_pixels(image(png)) > 50 and blue_pixels(image(plain)) == 0


def test_a_page_that_does_not_exist_raises_index_error(pdf):
    for bad in (-1, 2, 99):
        with pytest.raises(IndexError):
            render_page_png(pdf, bad)


def test_rendering_never_modifies_the_pdf(pdf):
    before = open(pdf, "rb").read()
    render_page_png(pdf, 0, quote=QUOTE, bbox=[10, 10, 100, 100])
    assert open(pdf, "rb").read() == before
