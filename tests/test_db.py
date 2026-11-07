import pytest

from config import get_settings
from services import db


def test_case_lifecycle():
    c = db.create_case("  Acme vs Beta  ", " dispute about fees ")
    assert c["title"] == "Acme vs Beta" and c["description"] == "dispute about fees" and c["id"].startswith("c")
    assert db.get_case(c["id"]) == c and db.get_case("nope") is None
    db.create_case("Second")
    assert [x["title"] for x in db.list_cases()] == ["Second", "Acme vs Beta"]          # newest first
    assert all(x["n_docs"] == 0 for x in db.list_cases())


def test_document_lifecycle_and_json_warnings():
    c = db.create_case("C")
    d = db.add_document(c["id"], "a.pdf", "sha-a", 1234)
    assert d["status"] == "queued" and d["warnings"] == [] and d["page_count"] == 0 and d["ocr_status"] == "n/a"
    db.update_document(d["id"], status="indexed", page_count=7, n_chunks=40, warnings=["1 page needs OCR", "x"], error="")
    got = db.get_document(d["id"])
    assert (got["status"], got["page_count"], got["n_chunks"]) == ("indexed", 7, 40)
    assert got["warnings"] == ["1 page needs OCR", "x"]
    db.add_document(c["id"], "b.pdf", "sha-b", 1, doc_id="dfixed")
    assert [x["filename"] for x in db.list_documents(c["id"])] == ["a.pdf", "b.pdf"]
    assert db.list_cases()[0]["n_docs"] == 2
    db.delete_document(d["id"])
    assert db.get_document(d["id"]) is None and len(db.list_documents(c["id"])) == 1


def test_only_known_columns_can_be_updated():
    d = db.add_document(db.create_case("C")["id"], "a.pdf", "s", 1)
    with pytest.raises(ValueError, match="Cannot update"):
        db.update_document(d["id"], sha256="x")
    with pytest.raises(ValueError):
        db.update_document(d["id"], **{"status=1; DROP TABLE cases; --": "x"})
    assert db.get_document(d["id"])["sha256"] == "s" and db.list_cases()                    # nothing was damaged


def test_deleting_a_case_deletes_its_documents_and_log_but_not_other_cases():
    a, b = db.create_case("A"), db.create_case("B")
    db.add_document(a["id"], "a.pdf", "s1", 1)
    keep = db.add_document(b["id"], "b.pdf", "s2", 1)
    db.log_qa(a["id"], "q?", False, "HIGH", "hybrid", "m", 1.0)
    db.log_qa(b["id"], "q2?", True, "INSUFFICIENT EVIDENCE", "hybrid", "m", 2.0)
    db.delete_case(a["id"])
    assert db.get_case(a["id"]) is None and db.list_documents(a["id"]) == []
    assert db.get_document(keep["id"]) and db.stats()["questions"] == 1


def test_stats_count_indexed_documents_and_questions():
    c = db.create_case("C")
    d1, d2 = db.add_document(c["id"], "a.pdf", "1", 1), db.add_document(c["id"], "b.pdf", "2", 1)
    db.update_document(d1["id"], status="indexed", n_chunks=10)
    db.update_document(d2["id"], status="failed", n_chunks=99)
    db.log_qa(c["id"], "q1", False, "HIGH", "hybrid", "m", 1.5)
    db.log_qa(c["id"], "q2", True, "INSUFFICIENT EVIDENCE", "bm25", "m", 0.5)
    s = db.stats()
    assert (s["cases"], s["documents"], s["chunks"], s["questions"], s["abstentions"]) == (1, 1, 10, 2, 1)
    assert [r["question"] for r in s["recent"]] == ["q2", "q1"] and s["recent"][0]["case_title"] == "C"


def test_the_database_lives_in_the_storage_folder_and_empty_stats_are_zero():
    assert db.stats()["cases"] == 0
    assert (get_settings().storage_dir / "lexora.db").exists()


def test_ids_are_unique_and_prefixed():
    ids = {db.new_id("d") for _ in range(200)}
    assert len(ids) == 200 and all(i.startswith("d") and len(i) == 11 for i in ids)
