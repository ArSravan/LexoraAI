"""Shared fixtures: small PDFs are generated on the fly, so no binary test data is committed."""
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from config import get_settings
from schemas import Chunk


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Every test gets its own empty storage folder (databases, caches, indexes)."""
    monkeypatch.setenv("LEXORA_STORAGE_DIR", str(tmp_path / "storage"))
    get_settings.cache_clear()
    yield tmp_path / "storage"
    get_settings.cache_clear()


def make_chunk(chunk_id: str, text: str, clause: str = "", doc: str = "d1", name: str = "Doc.pdf",
               page: int = 1, section: str = "") -> Chunk:
    """A Chunk with the same index_text layout the real chunker produces."""
    head = " | ".join(x for x in (name, section, clause and f"clause {clause}") if x)
    return Chunk(chunk_id=chunk_id, document_id=doc, document_name=name, page_number=page,
                 page_index=page - 1, page_end=page, section=section, clause=clause, text=text,
                 index_text=f"{head}\n{text}")


def make_pdf(path: Path, pages: list[list[tuple[float, float, str]]]) -> Path:
    """Write a PDF; each page is a list of (x, y, text[, font_size]) in PDF points, origin bottom-left."""
    c = canvas.Canvas(str(path), pagesize=A4)
    for items in pages:
        for x, y, text, *size in items:
            c.setFont("Helvetica", size[0] if size else 10.5)
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


@pytest.fixture
def eu_style_pdf(tmp_path) -> Path:
    """EUR-Lex style statute (3 pages): recitals, a chapter, Article headings, '1.' paragraphs,
    (a)/(b) points and (i)/(ii) sub-points. A tuple row is a marker and its text on one visual row."""
    p1 = ["REGULATION (EU) 2024/1 OF THE EUROPEAN PARLIAMENT", "Having regard to the Treaty,", "Whereas:",
          ("(1)", "The protection of natural persons is a fundamental right."),
          ("(2)", "The principles of data protection should respect rights."),
          "HAVE ADOPTED THIS REGULATION:", "CHAPTER I", "General provisions",
          "Article 1", "Subject matter",
          "1.", "This Regulation lays down rules relating to the protection of data.",
          "2.", "This Regulation protects fundamental rights and freedoms."]
    p2 = ["Article 2", "Definitions", "For the purposes of this Regulation:",
          "(a) 'controller' means the body that determines the purposes of processing;",
          "(b) 'processor' means a body which processes data on behalf of the controller;",
          "(c) 'recipient' means a body to which the data are disclosed, namely:",
          "(i) a public authority;", "(ii) a private company.",
          "Article 3", "Fines",
          "1.", "Infringements shall be subject to administrative fines up to 10 000 000 EUR.",
          "2.", "Serious infringements shall be subject to fines up to 20 000 000 EUR, or 4 % of annual turnover:",
          "(a) breach of the basic principles;", "(b) breach of data subject rights."]
    p3 = ["Article 4", "Final provisions",
          "This Regulation shall enter into force on 25 May 2018.", "It shall be binding in its entirety."]
    pages = []
    for k, rows in enumerate((p1, p2, p3), 1):
        items = []
        for i, row in enumerate(rows):
            y = 780 - 18 * i
            if isinstance(row, tuple):
                items += [(60, y, row[0]), (90, y, row[1])]
            else:
                items.append((60, y, row))
        items += [(60, 815, f"Official Journal EN {k}"), (280, 25, f"- {k} -")]
        pages.append(items)
    return make_pdf(tmp_path / "Statute.pdf", pages)


@pytest.fixture
def contract_pdf(tmp_path) -> Path:
    """A contract with 'Section N.' headings, '7.1' clauses and a ruled table caption 'Table 9.1'."""
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    st = getSampleStyleSheet()["Normal"]
    P = lambda t: [Paragraph(t, st), Spacer(1, 5)]                      # noqa: E731
    table = Table([["Item", "Amount (EUR)"], ["Data breach", "5,000,000"],
                   ["IP infringement", "3,000,000"], ["Total", "8,000,000"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black)]))
    story = (P("This Agreement is made between Acme Ltd and Beta GmbH.")
             + P("NOW, THEREFORE the parties agree as follows:")
             + P("Section 7. Limitation of Liability")
             + P("7.1 The Supplier's total aggregate liability under this Agreement shall not exceed EUR 10,000,000.")
             + P("7.2 Nothing in this Section 7 limits liability for gross negligence.")
             + P("Section 9. Indemnities")
             + P("9.1 The Supplier shall indemnify the Customer for the amounts in Table 9.1.")
             + P("Table 9.1 Indemnity obligations") + [table, Spacer(1, 8)]
             + P("9.2 Indemnities are payable within thirty days of a written demand."))
    path = tmp_path / "Vendor_Agreement.pdf"
    SimpleDocTemplate(str(path)).build(story)
    return path
