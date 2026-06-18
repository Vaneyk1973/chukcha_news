"""Regression tests for the Chukchi News Voice pipeline."""

from pathlib import Path

import pytest

from scripts.data.prepare_fieldasr_dataset import (
    duration_matches,
    index_files,
    normalize_text,
    safe_member_path,
)


def test_normalize_fieldasr_text() -> None:
    """Exercise the `test_normalize_fieldasr_text` behavior and guard against regressions."""
    assert normalize_text("\ufeff  Ԓыгъоравэтԓьэн\n  текст  ") == "Ԓыгъоравэтԓьэн текст"


def test_duration_matches_absolute_or_relative_tolerance() -> None:
    """
    Exercise the `test_duration_matches_absolute_or_relative_tolerance` behavior and guard against regressions.
    """
    config = {"duration_tolerance_sec": 1.0, "duration_tolerance_ratio": 0.15}
    assert duration_matches(10.9, 10.0, config)
    assert duration_matches(112.0, 100.0, config)
    assert not duration_matches(120.0, 100.0, config)


def test_safe_member_path_rejects_archive_traversal(tmp_path: Path) -> None:
    """
    Exercise the `test_safe_member_path_rejects_archive_traversal` behavior and guard against regressions.
    """
    with pytest.raises(ValueError):
        safe_member_path(tmp_path, "../escape.wav")


def test_audio_index_matches_original_wav_name_to_mp3(tmp_path: Path) -> None:
    """
    Exercise the `test_audio_index_matches_original_wav_name_to_mp3` behavior and guard against regressions.
    """
    audio = tmp_path / "Knives_16.mp3"
    audio.touch()
    index = index_files([tmp_path], {".mp3"})
    assert index["knives_16"] == [audio]
