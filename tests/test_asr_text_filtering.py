"""Regression tests for the Chukchi News Voice pipeline."""

from scripts.data.filter_asr_pseudolabels import (
    chukchi_specific_ratio,
    rejection_reasons,
    russian_score,
)


def test_russian_radio_text_scores_high() -> None:
    """
    Exercise the `test_russian_radio_text_scores_high` behavior and guard against regressions.
    """
    text = "новэчти нарадиё пурга сегодня в эфире программа"
    assert russian_score(text) >= 2.0


def test_chukchi_specific_ratio_detects_chukchi_letters() -> None:
    """
    Exercise the `test_chukchi_specific_ratio_detects_chukchi_letters` behavior and guard against regressions.
    """
    assert chukchi_specific_ratio("ӈоотӄэ ԓыгъоравэтԓьэн") > 0.05


def test_reject_russian_text_without_chukchi_signal() -> None:
    """
    Exercise the `test_reject_russian_text_without_chukchi_signal` behavior and guard against regressions.
    """
    row = {"transcript": "новости на радио пурга сегодня программа"}
    assert rejection_reasons(row, None, 0.015, 2.0, 4.0, 0.35) == ["russian_text"]


def test_reject_hard_russian_score_even_with_some_chukchi_letters() -> None:
    """
    Exercise the `test_reject_hard_russian_score_even_with_some_chukchi_letters` behavior and guard against regressions.
    """
    row = {
        "transcript": "почекрет вам раскажу дамы гсподатолкы вытнэ нарадэ пурга проԓуковски язык"
    }
    assert rejection_reasons(row, None, 0.015, 2.0, 4.0, 0.35) == ["russian_text"]


def test_keep_chukchi_like_text_even_with_music_background() -> None:
    """
    Exercise the `test_keep_chukchi_like_text_even_with_music_background` behavior and guard against regressions.
    """
    row = {"transcript": "етыӄыпаԓёмтэԓгыткытурпыӈэԓти энанымӈыԓявыԓьэгым"}
    audio = {"audio_top_labels": '[{"label": "Music", "score": 0.4}]'}
    assert rejection_reasons(row, audio, 0.015, 2.0, 4.0, 0.35) == []


def test_reject_likely_foreign_song_without_chukchi_signal() -> None:
    """
    Exercise the `test_reject_likely_foreign_song_without_chukchi_signal` behavior and guard against regressions.
    """
    row = {"transcript": "ла ла ла бейби лав ю"}
    audio = {"audio_top_labels": '[{"label": "Singing", "score": 0.8}]'}
    assert rejection_reasons(row, audio, 0.015, 2.0, 4.0, 0.35) == ["likely_foreign_song"]
