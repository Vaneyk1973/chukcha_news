"""Regression tests for the Chukchi News Voice pipeline."""

from scripts.data.prepare_asr import bound_regions, parse_silences, speech_regions


def test_parse_silences() -> None:
    """Exercise the `test_parse_silences` behavior and guard against regressions."""
    stderr = """
[silencedetect] silence_start: 1.25
[silencedetect] silence_end: 2.5 | silence_duration: 1.25
[silencedetect] silence_start: 8
[silencedetect] silence_end: 9 | silence_duration: 1
"""
    assert parse_silences(stderr) == [(1.25, 2.5), (8.0, 9.0)]


def test_speech_regions_drop_short_and_bound_long_regions() -> None:
    """
    Exercise the `test_speech_regions_drop_short_and_bound_long_regions` behavior and guard against regressions.
    """
    regions = speech_regions(
        duration=30.0,
        silences=[(1.0, 3.0), (25.0, 29.0)],
        min_segment=2.5,
        max_segment=10.0,
        padding=0.0,
    )
    assert regions == [(3.0, 13.0), (13.0, 22.5), (22.5, 25.0)]


def test_speech_regions_add_bounded_padding() -> None:
    """
    Exercise the `test_speech_regions_add_bounded_padding` behavior and guard against regressions.
    """
    regions = speech_regions(
        duration=10.0,
        silences=[(0.0, 2.0), (7.0, 10.0)],
        min_segment=2.5,
        max_segment=15.0,
        padding=0.2,
    )
    assert regions == [(1.8, 7.2)]


def test_bound_regions_splits_vad_output() -> None:
    """
    Exercise the `test_bound_regions_splits_vad_output` behavior and guard against regressions.
    """
    assert bound_regions([(2.0, 24.0)], 30.0, 2.5, 10.0, 0.0) == [
        (2.0, 12.0),
        (12.0, 21.5),
        (21.5, 24.0),
    ]
