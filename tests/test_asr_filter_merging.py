"""Regression tests for the Chukchi News Voice pipeline."""

from scripts.data.merge_asr_filter_signals import final_decision, merge_row


def test_keep_only_when_text_and_lm_pass() -> None:
    """
    Exercise the `test_keep_only_when_text_and_lm_pass` behavior and guard against regressions.
    """
    row = {"segment_id": "a", "transcript": "ԓыгъоравэтԓьэн"}
    text_row = {"segment_id": "a", "text_filter_reasons": ""}
    score_row = {"segment_id": "a", "chukchi_text_class": "chukchi_text"}

    assert final_decision(row, text_row, score_row) == ("kept", [])


def test_reject_text_filter_reason() -> None:
    """
    Exercise the `test_reject_text_filter_reason` behavior and guard against regressions.
    """
    row = {"segment_id": "a", "transcript": "новости радио"}
    text_row = {"segment_id": "a", "text_filter_reasons": "russian_text"}
    score_row = {"segment_id": "a", "chukchi_text_class": "chukchi_text"}

    assert final_decision(row, text_row, score_row) == ("rejected", ["russian_text"])


def test_reject_chukchi_lm_reject() -> None:
    """Exercise the `test_reject_chukchi_lm_reject` behavior and guard against regressions."""
    row = {"segment_id": "a", "transcript": "ла ла ла"}
    text_row = {"segment_id": "a", "text_filter_reasons": ""}
    score_row = {"segment_id": "a", "chukchi_text_class": "reject"}

    assert final_decision(row, text_row, score_row) == ("rejected", ["chukchi_lm_reject"])


def test_uncertain_chukchi_lm_uncertain() -> None:
    """
    Exercise the `test_uncertain_chukchi_lm_uncertain` behavior and guard against regressions.
    """
    row = {"segment_id": "a", "transcript": "сомнительно"}
    text_row = {"segment_id": "a", "text_filter_reasons": ""}
    score_row = {"segment_id": "a", "chukchi_text_class": "uncertain"}

    assert final_decision(row, text_row, score_row) == ("uncertain", ["chukchi_lm_uncertain"])


def test_merge_preserves_audio_and_score_columns() -> None:
    """
    Exercise the `test_merge_preserves_audio_and_score_columns` behavior and guard against regressions.
    """
    row = {"segment_id": "a", "transcript": "ԓыгъоравэтԓьэн"}
    text_row = {"segment_id": "a", "text_filter_reasons": "", "russian_score": "0.000"}
    score_row = {"segment_id": "a", "chukchi_text_class": "chukchi_text"}
    audio_row = {"segment_id": "a", "audio_class": "music"}

    merged = merge_row(row, text_row, score_row, audio_row)

    assert merged["asr_clean_verdict"] == "kept"
    assert merged["audio_class"] == "music"
    assert merged["chukchi_text_class"] == "chukchi_text"
