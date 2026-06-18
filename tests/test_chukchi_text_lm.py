"""Regression tests for the Chukchi News Voice pipeline."""

from chukcha_news.mt.text_lm import CharNgramLM, normalize_text


def test_normalize_text_keeps_chukchi_letters() -> None:
    """
    Exercise the `test_normalize_text_keeps_chukchi_letters` behavior and guard against regressions.
    """
    assert normalize_text(" Ԓыгъоравэтԓьэн! ") == "ԓыгъоравэтԓьэн"


def test_chukchi_lm_prefers_seen_style_text() -> None:
    """
    Exercise the `test_chukchi_lm_prefers_seen_style_text` behavior and guard against regressions.
    """
    lm = CharNgramLM.train(
        [
            "ԓыгъоравэтԓьэн ӄытгъэргъын ӈинӄэй",
            "ытычечкэн оравэтԓьан нырэпԓиткуӈӄинэт",
            "аӈаӈатыпԓыккогыргыт еԓык нымкыӄинэт",
        ],
        order=3,
    )

    good = lm.average_log_probability("ԓыгъоравэтԓьэн ӄытгъэргъын")
    bad = lm.average_log_probability("ла ла ла бейби лав ю")

    assert good > bad
