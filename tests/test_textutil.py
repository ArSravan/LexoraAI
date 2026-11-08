import pytest

from textutil import clause_key, normalize, stem, words


def test_normalize_straightens_quotes_dashes_and_spaces():
    assert normalize("  The “Controller”’s   duty – here now ") == 'the "controller"\'s duty - here now'


def test_words_keep_decimal_numbers_together():
    assert words("Section 7.2 and 20 000 EUR") == ["section", "7.2", "and", "20", "000", "eur"]


@pytest.mark.parametrize("group", [
    ["fine", "fines", "fined", "fining"],
    ["process", "processes", "processed", "processing"],
    ["regulation", "regulations"],
    ["company", "companies"],
])
def test_different_forms_of_a_word_share_one_stem(group):
    assert len({stem(w) for w in group}) == 1


def test_words_ending_in_ss_keep_their_s():
    assert stem("process") == "process" and stem("access") == "access" and stem("cross") == "cross"


def test_numbers_short_words_and_labels_are_not_stemmed():
    assert stem("83(5)") == "83(5)" and stem("7.2") == "7.2" and stem("was") == "was" and stem("20") == "20"


def test_clause_key_ignores_the_word_before_the_number():
    assert clause_key("Article 83(5)(a)") == clause_key("83(5)(a)") == "83(5)(a)"
    assert clause_key("Section 7.2") == clause_key("7.2") == "7.2"
    assert clause_key("Art. 6 (1)(a)") == "6(1)(a)"
    assert clause_key("Recital 4") == "recital4"


from textutil import numbers  # noqa: E402


@pytest.mark.parametrize("text", ["20 000 000 EUR", "EUR 20 million", "20,000,000", "€20m", "20 mn euros"])
def test_the_same_amount_written_differently_gives_the_same_number(text):
    assert numbers(text) == {20_000_000.0}


def test_percentages_decimals_and_decimal_commas():
    assert numbers("up to 4 % of annual turnover") == {4.0}
    assert numbers("1.5 million") == {1_500_000.0} and numbers("1,5 million") == {1_500_000.0}
    assert numbers("1,500") == {1500.0}


def test_clause_and_page_references_are_not_quantities():
    assert numbers("Article 83(5) on page 14: fines up to 20 000 000 EUR, Section 7.2, Table 9.1") == {20_000_000.0}
    assert {83.0, 5.0, 14.0} <= numbers("Article 83(5) on page 14", strip_refs=False)


def test_dates_and_plain_text():
    assert numbers("entered into force on 25 May 2018") == {25.0, 2018.0} and numbers("no figures here") == set()


def test_remove_refs_blanks_only_the_references():
    from textutil import remove_refs
    out = remove_refs("Under Article 83(5) and Section 7.2 on page 14 the fine is 20 000 000 EUR")
    assert "83" not in out and "7.2" not in out and "14" not in out and "20 000 000 EUR" in out
