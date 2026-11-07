import json

import pytest

from retrieval import hybrid, vector_store
from retrieval.hybrid import search
from services import indexer
from services.paths import case_dir, check_id, chunks_file, doc_dir


@pytest.fixture
def indexed(fake_embedder, eu_style_pdf, contract_pdf):
    indexer.index_pdf("c1", "statute", eu_style_pdf, "Statute.pdf")
    indexer.index_pdf("c1", "contract", contract_pdf, "Vendor_Agreement.pdf")
    return "c1"


def test_both_stores_are_written_and_agree(indexed):
    chunks = indexer.case_chunks(indexed)
    assert {c.document_id for c in chunks} == {"statute", "contract"}
    assert vector_store.count(indexed) == len(chunks) == len(json.loads(chunks_file(indexed).read_text()))
    assert not chunks_file(indexed).with_suffix(".tmp").exists()


def test_end_to_end_a_question_finds_the_right_clause(indexed):
    top = search(indexed, "What is the maximum fine for serious infringements, 4 % of turnover?", mode="hybrid")[0]
    assert top.chunk.document_name == "Statute.pdf" and top.chunk.clause == "3(2)"
    top = search(indexed, "Section 7 limitation of liability cap EUR 10,000,000", mode="bm25")[0]
    assert top.chunk.document_name == "Vendor_Agreement.pdf" and top.chunk.clause == "7.1"


def test_a_table_is_searchable_by_its_cells(indexed):
    top = search(indexed, "Table 9.1 indemnity IP infringement 3,000,000", mode="bm25")[0]
    assert top.chunk.chunk_type == "table" and "3,000,000" in top.chunk.text


def test_reindexing_replaces_instead_of_duplicating(indexed, eu_style_pdf):
    before = len(indexer.case_chunks(indexed))
    indexer.index_pdf(indexed, "statute", eu_style_pdf, "Statute.pdf")
    assert len(indexer.case_chunks(indexed)) == before and vector_store.count(indexed) == before


def test_remove_document_keeps_the_other_documents(indexed):
    indexer.remove_document(indexed, "contract")
    assert {c.document_id for c in indexer.case_chunks(indexed)} == {"statute"}
    assert vector_store.count(indexed) == len(indexer.case_chunks(indexed))
    assert all(h.chunk.document_id == "statute" for h in search(indexed, "liability", mode="bm25"))


def test_remove_case_deletes_everything(indexed):
    indexer.remove_case(indexed)
    assert not case_dir(indexed).exists() and vector_store.count(indexed) == 0
    assert search(indexed, "liability") == []


def test_progress_is_reported(fake_embedder, eu_style_pdf):
    seen = []
    indexer.index_pdf("c2", "d", eu_style_pdf, progress=lambda f, m: seen.append((f, m)))
    assert seen[0][0] < seen[-1][0] == 1.0 and seen[-1][1] == "Indexed"


def test_an_unreadable_pdf_changes_nothing(fake_embedder, tmp_path):
    from ingestion.pdf_parser import PDFValidationError
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(PDFValidationError):
        indexer.index_pdf("c3", "d", bad)
    assert indexer.case_chunks("c3") == []


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "a b", "c:\\x", "x" * 65])
def test_hostile_identifiers_are_rejected(bad):
    with pytest.raises(ValueError):
        check_id(bad)
    with pytest.raises(ValueError):
        doc_dir("c1", bad)


def test_valid_identifiers_are_accepted():
    assert check_id("c1a2b3_c-4") == "c1a2b3_c-4"


def test_document_metadata_is_recorded_for_the_verifier(indexed):
    docs = {d["document_id"]: d for d in indexer.case_documents(indexed)}
    assert set(docs) == {"statute", "contract"}
    st = docs["statute"]
    assert st["filename"] == "Statute.pdf" and st["page_count"] == 3 and len(st["sha256"]) == 64
    assert {"3(2)", "3(2)(a)", "Recital 1"} <= set(st["labels"]) and st["n_chunks"] > 5
    assert docs["contract"]["page_count"] >= 1 and "7.1" in docs["contract"]["labels"]
    assert [d["filename"] for d in indexer.case_documents(indexed)] == ["Statute.pdf", "Vendor_Agreement.pdf"]


def test_metadata_is_replaced_on_reindex_and_removed_with_the_document(indexed, eu_style_pdf):
    indexer.index_pdf(indexed, "statute", eu_style_pdf, "Renamed.pdf")
    assert {d["filename"] for d in indexer.case_documents(indexed)} == {"Renamed.pdf", "Vendor_Agreement.pdf"}
    indexer.remove_document(indexed, "statute")
    assert [d["document_id"] for d in indexer.case_documents(indexed)] == ["contract"]


def test_a_case_without_documents_has_no_metadata():
    assert indexer.case_documents("nothing-here") == []


# ------------------------------------------------------------------------------ documents with a database record
from config import get_settings  # noqa: E402
from ingestion.pdf_parser import PDFValidationError  # noqa: E402
from services import db  # noqa: E402


def register(pdf, case_id="c1", doc_id="d1", name="Statute.pdf"):
    """What the upload endpoint does: a database record plus the PDF in its folder."""
    import shutil
    case = db.get_case(case_id) or db.create_case("Case", case_id=case_id)
    rec = db.add_document(case["id"], name, "sha", pdf.stat().st_size, doc_id)
    indexer.pdf_path(case_id, doc_id).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(pdf, indexer.pdf_path(case_id, doc_id))
    return rec


def test_index_document_updates_the_record_and_writes_the_caches(fake_embedder, eu_style_pdf):
    register(eu_style_pdf)
    rec = indexer.index_document("c1", "d1")
    assert rec["status"] == "indexed" and rec["error"] == "" and rec["page_count"] == 3
    assert rec["n_chunks"] > 5 and rec["n_clauses"] > 5 and rec["n_scanned"] == 0 and rec["ocr_status"] == "n/a"
    assert vector_store.count("c1") == rec["n_chunks"] == len(indexer.case_chunks("c1"))
    parsed = indexer.load_parsed("c1", "d1")
    assert parsed.page_count == 3 and indexer.load_structure("c1", "d1")["mode"] == "legal-structure"
    assert indexer.load_parsed("c1", "nope") is None and indexer.load_structure("c1", "nope") == {}


def test_the_extraction_cache_is_reused_unless_a_reparse_is_forced(fake_embedder, eu_style_pdf, monkeypatch):
    register(eu_style_pdf)
    indexer.index_document("c1", "d1")
    calls = []
    real = indexer.parse_pdf
    monkeypatch.setattr(indexer, "parse_pdf", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    messages = []
    indexer.index_document("c1", "d1", progress=lambda f, m: messages.append(m))
    assert calls == [] and "Using cached extraction" in messages
    indexer.index_document("c1", "d1", force_reparse=True)
    assert calls == [1]


def test_progress_runs_from_zero_to_one_in_order(fake_embedder, eu_style_pdf):
    register(eu_style_pdf)
    seen = []
    indexer.index_document("c1", "d1", progress=lambda f, m: seen.append(f))
    assert seen == sorted(seen) and seen[0] < 0.2 and seen[-1] == 1.0


def test_scanned_pages_are_counted_in_the_record(fake_embedder, tmp_path, monkeypatch):
    from tests.test_ocr import _scan_of
    monkeypatch.setenv("LEXORA_OCR_ENABLED", "false")
    get_settings.cache_clear()
    register(_scan_of(tmp_path), name="Scan.pdf")
    rec = indexer.index_document("c1", "d1")
    assert rec["status"] == "indexed" and rec["n_scanned"] == 1 and rec["ocr_status"] == "OCR 0/1 scanned pages"
    assert any("need OCR" in w for w in rec["warnings"])


def test_a_failed_document_keeps_its_error_and_leaves_the_indexes_alone(fake_embedder, eu_style_pdf, tmp_path):
    register(eu_style_pdf)
    indexer.index_document("c1", "d1")
    before = len(indexer.case_chunks("c1"))
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    register(bad, doc_id="d2", name="Bad.pdf")
    with pytest.raises(PDFValidationError):
        indexer.index_document("c1", "d2")
    rec = db.get_document("d2")
    assert rec["status"] == "failed" and "%PDF" in rec["error"]
    assert len(indexer.case_chunks("c1")) == before
    with pytest.raises(ValueError, match="Unknown document"):
        indexer.index_document("c1", "missing")


def test_an_unexpected_error_is_recorded_with_its_type(fake_embedder, eu_style_pdf, monkeypatch):
    register(eu_style_pdf)
    monkeypatch.setattr(indexer, "index_parsed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError):
        indexer.index_document("c1", "d1")
    assert db.get_document("d1")["status"] == "failed" and db.get_document("d1")["error"] == "RuntimeError: disk full"
