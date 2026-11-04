"""Shared fixtures: small PDFs are generated on the fly, so no binary test data is committed."""
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def make_pdf(path: Path, pages: list[list[tuple[float, float, str]]]) -> Path:
    """Write a PDF; each page is a list of (x, y, text) in PDF points, origin bottom-left."""
    c = canvas.Canvas(str(path), pagesize=A4)
    for items in pages:
        c.setFont("Helvetica", 10.5)
        for x, y, text in items:
            c.drawString(x, y, text)
        c.showPage()
    c.save()
    return path


@pytest.fixture
def simple_pdf(tmp_path) -> Path:
    pages = [
        [(60, 760, "Article 1"), (60, 742, "This Regulation lays down rules on data protection.")],
        [(60, 760, "Article 2"), (60, 742, "Fines may be up to 20 000 000 EUR.")],
    ]
    return make_pdf(tmp_path / "simple.pdf", pages)