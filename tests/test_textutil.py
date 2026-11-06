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
