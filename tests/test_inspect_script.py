from scripts.inspect_pdf import main


def test_inspect_prints_summary_and_clause_text(eu_style_pdf, capsys):
    assert main([str(eu_style_pdf), "--chunks", "3"]) == 0
    out = capsys.readouterr().out
    assert "Statute.pdf: 3 pages" in out and "structure mode: legal-structure" in out
    assert out.count("[doc") == 0 and out.count("[inspect-") == 3

    assert main([str(eu_style_pdf), "--clause", "3(2)"]) == 0
    assert "Serious infringements" in capsys.readouterr().out
