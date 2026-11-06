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
