from services import indexer
from scripts.search_pdf import CONFIGS, main


def test_compare_prints_every_configuration(eu_style_pdf, contract_pdf, fake_embedder, fake_reranker, capsys):
    q = "What fines apply to serious infringements, and what liability cap does the supplier have?"
    assert main([str(eu_style_pdf), str(contract_pdf), "-q", q, "--compare", "--k", "3"]) == 0
    out = capsys.readouterr().out
    for label, *_ in CONFIGS:
        assert f"== {label}" in out
    assert "Statute.pdf p.2 clause 3(2)" in out and "Vendor_Agreement.pdf" in out
    assert "sub-queries:" in out


def test_single_configuration_and_nothing_found(eu_style_pdf, fake_embedder, fake_reranker, capsys):
    assert main([str(eu_style_pdf), "-q", "blockchain ledger", "--mode", "bm25", "--no-rerank"]) == 0
    assert "(nothing found)" in capsys.readouterr().out


def test_pdfs_are_indexed_only_once(eu_style_pdf, fake_embedder, fake_reranker, monkeypatch, capsys):
    calls = []
    real = indexer.index_pdf
    monkeypatch.setattr(indexer, "index_pdf", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    main([str(eu_style_pdf), "-q", "administrative fines", "--no-rerank"])
    main([str(eu_style_pdf), "-q", "consent of the data subject", "--no-rerank"])
    assert len(calls) == 1
