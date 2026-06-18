from pathlib import Path

from scripts.data.prepare_tts_training_dataset import convert_row, split_rows
from scripts.evaluate.evaluate_tts import summarize


def test_convert_row_uses_absolute_audio_path() -> None:
    row = {
        "segment_id": "abc",
        "audio_path": "data/interim/example.wav",
        "text": "  ԓыгъоравэтԓьэн  ",
        "duration_sec": 3.5,
    }

    converted = convert_row(row)

    assert converted["segment_id"] == "abc"
    assert Path(converted["audio"]).is_absolute()
    assert converted["text"] == "ԓыгъоравэтԓьэн"
    assert converted["speaker_id"] == 0


def test_split_rows_is_deterministic_and_keeps_eval() -> None:
    rows = [{"segment_id": str(index)} for index in range(10)]

    train_a, eval_a = split_rows(rows, train_ratio=0.8, seed=7)
    train_b, eval_b = split_rows(rows, train_ratio=0.8, seed=7)

    assert train_a == train_b
    assert eval_a == eval_b
    assert len(train_a) == 8
    assert len(eval_a) == 2


def test_summarize_rounds_cer() -> None:
    rows = [
        {"cer": "0.1", "synthetic_duration_sec": "1.0"},
        {"cer": "0.3", "synthetic_duration_sec": "3.0"},
    ]

    assert summarize(rows) == {
        "samples": 2,
        "mean_cer": 0.2,
        "median_cer": 0.3,
        "mean_synthetic_duration_sec": 2.0,
    }
