"""Regression tests for the Chukchi News Voice pipeline."""

from scripts.data.classify_asr_audio import bucket_scores, decide_class


def test_bucket_scores_maps_audio_labels() -> None:
    """
    Exercise the `test_bucket_scores_maps_audio_labels` behavior and guard against regressions.
    """
    scores = bucket_scores(
        [
            {"label": "Speech", "score": 0.8},
            {"label": "Music", "score": 0.4},
            {"label": "Singing", "score": 0.6},
        ]
    )
    assert scores["speech"] == 0.8
    assert scores["music"] == 0.6


def test_decide_class_prefers_music_when_music_is_close() -> None:
    """
    Exercise the `test_decide_class_prefers_music_when_music_is_close` behavior and guard against regressions.
    """
    decision, confidence, margin = decide_class(
        [{"label": "Speech", "score": 0.42}, {"label": "Music", "score": 0.35}],
        speech_threshold=0.35,
        music_threshold=0.25,
        margin_threshold=0.10,
    )
    assert decision == "music"
    assert confidence == 0.35
    assert margin < 0


def test_decide_class_keeps_clear_speech() -> None:
    """
    Exercise the `test_decide_class_keeps_clear_speech` behavior and guard against regressions.
    """
    decision, confidence, margin = decide_class(
        [{"label": "Speech", "score": 0.8}, {"label": "Music", "score": 0.1}],
        speech_threshold=0.35,
        music_threshold=0.25,
        margin_threshold=0.10,
    )
    assert decision == "speech"
    assert confidence == 0.8
    assert margin > 0
