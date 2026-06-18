"""Regression tests for the Chukchi News Voice pipeline."""

from chukcha_news.mt.metrics import (
    character_error_rate,
    mean_character_error_rate,
    normalize_chukchi_detokenization,
)


def test_character_error_rate_identity() -> None:
    """
    Exercise the `test_character_error_rate_identity` behavior and guard against regressions.
    """
    assert character_error_rate("Ԓыгъоравэтԓьэн", "Ԓыгъоравэтԓьэн") == 0.0


def test_character_error_rate_single_substitution() -> None:
    """
    Exercise the `test_character_error_rate_single_substitution` behavior and guard against regressions.
    """
    assert character_error_rate("abc", "axc") == 1 / 3


def test_mean_character_error_rate() -> None:
    """
    Exercise the `test_mean_character_error_rate` behavior and guard against regressions.
    """
    assert mean_character_error_rate(["abc", "x"], ["abc", ""]) == 0.5


def test_normalize_chukchi_detokenization_removes_added_letter_spaces() -> None:
    """
    Exercise the `test_normalize_chukchi_detokenization_removes_added_letter_spaces` behavior and guard against regressions.
    """
    assert (
        normalize_chukchi_detokenization("Ԓ ыгъоравэтԓ ьэн ӄ ытгъэргъын ӈ инӄ эй.")
        == "Ԓыгъоравэтԓьэн ӄытгъэргъын ӈинӄэй."
    )
