"""Regression tests for the Chukchi News Voice pipeline."""

from scripts.data.promote_asr_pseudolabels import convert_row, normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    """
    Exercise the `test_normalize_text_collapses_whitespace` behavior and guard against regressions.
    """
    assert normalize_text("  аӈӄы\n  тэк ") == "аӈӄы тэк"


def test_convert_row_uses_tts_schema_assumptions() -> None:
    """
    Exercise the `test_convert_row_uses_tts_schema_assumptions` behavior and guard against regressions.
    """
    row = {
        "segment_id": "abc",
        "audio_path": "seg.wav",
        "source_path": "source.mp3",
        "start_sec": "1.0",
        "end_sec": "3.5",
        "duration_sec": "2.5",
        "transcript": "  Ԓыгъоравэтԓьэн  ",
        "confidence": "0.98",
        "label_source": "facebook/mms-1b-all",
    }

    converted = convert_row(row, assume_single_speaker=True, assume_no_music=True)

    assert converted["text"] == "Ԓыгъоравэтԓьэн"
    assert converted["single_speaker"] is True
    assert converted["has_music"] is False
    assert converted["duration_sec"] == 2.5
