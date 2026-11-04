import hashlib

import pymupdf
import pytest

from LexoraAI.ingestion.pdf_parser import PDFValidationError, safe_filename, sha256_file, validate_pdf
from tests.conftest import make_pdf


def test_valid_pdf_returns_page_count(simple_pdf):
    assert validate_pdf(simple_pdf) == 2


def test_missing_file(tmp_path):
    with pytest.raises(PDFValidationError, match="not found"):
        validate_pdf(tmp_path / "nope.pdf")


def test_wrong_extension(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    with pytest.raises(PDFValidationError, match="Only .pdf"):
        validate_pdf(f)


def test_pdf_extension_but_not_pdf_content(tmp_path):
    f = tmp_path / "fake.pdf"
    f.write_bytes(b"<html>definitely not a pdf</html>")
    with pytest.raises(PDFValidationError, match="%PDF"):
        validate_pdf(f)


def test_size_limit(tmp_path):
    f = tmp_path / "big.pdf"
    f.write_bytes(b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024))
    with pytest.raises(PDFValidationError, match="limit is 1 MB"):
        validate_pdf(f, max_mb=1)


def test_corrupt_pdf_that_has_the_header(tmp_path):
    f = tmp_path / "corrupt.pdf"
    f.write_bytes(b"%PDF-1.4\nthis is garbage\n")
    with pytest.raises(PDFValidationError):
        validate_pdf(f)


def test_encrypted_pdf_is_rejected(tmp_path, simple_pdf):
    out = tmp_path / "locked.pdf"
    doc = pymupdf.open(simple_pdf)
    doc.save(out, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    doc.close()
    with pytest.raises(PDFValidationError, match="Encrypted"):
        validate_pdf(out)


def test_sha256_matches_hashlib(simple_pdf):
    assert sha256_file(simple_pdf) == hashlib.sha256(simple_pdf.read_bytes()).hexdigest()


def test_identical_content_gives_identical_hash(tmp_path, simple_pdf):
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(simple_pdf.read_bytes())
    other = make_pdf(tmp_path / "other.pdf", [[(60, 760, "different")]])
    assert sha256_file(copy) == sha256_file(simple_pdf)
    assert sha256_file(other) != sha256_file(simple_pdf)


@pytest.mark.parametrize("raw,expected", [
    ("contract.pdf", "contract.pdf"),
    ("../../etc/passwd.pdf", "passwd.pdf"),
    ("C:\\Users\\me\\Report 2025.PDF", "Report 2025.PDF"),
    ("a$b&c.pdf", "a_b_c.pdf"),
])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_safe_filename_requires_pdf():
    with pytest.raises(PDFValidationError):
        safe_filename("malware.exe")