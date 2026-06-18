#!/usr/bin/env python3
"""Build a small high-confidence TTS manifest from cleaned ASR pseudo-labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "interim" / "asr_clean" / "kept.csv"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "tts_manifest_top1h.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "tts_top1h_selection.json"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def chars_per_second(row: dict) -> float:
    duration = as_float(row, "duration_sec")
    if duration <= 0:
        return 0.0
    return len(normalize_text(row.get("transcript", ""))) / duration


def quality_score(row: dict) -> float:
    confidence = as_float(row, "confidence")
    duration = as_float(row, "duration_sec")
    cps = chars_per_second(row)
    chukchi_ratio = as_float(row, "chukchi_specific_ratio")
    russian_score = as_float(row, "russian_score")
    song_score = as_float(row, "foreign_song_score")
    audio_class = row.get("audio_class", "")

    duration_penalty = abs(duration - 8.0) * 0.005
    pace_penalty = abs(cps - 11.0) * 0.002
    speech_bonus = 0.02 if audio_class == "speech" else 0.0
    return (
        confidence
        + min(chukchi_ratio, 0.12) * 0.4
        + speech_bonus
        - russian_score * 0.08
        - song_score * 0.03
        - duration_penalty
        - pace_penalty
    )


def usable(row: dict, min_duration: float, max_duration: float, min_confidence: float) -> bool:
    text = normalize_text(row.get("transcript", ""))
    duration = as_float(row, "duration_sec")
    if not text:
        return False
    if row.get("asr_clean_verdict") not in {"", "kept"}:
        return False
    if not min_duration <= duration <= max_duration:
        return False
    if as_float(row, "confidence") < min_confidence:
        return False
    if as_float(row, "russian_score") > 0:
        return False
    cps = chars_per_second(row)
    return 4.0 <= cps <= 22.0


def convert_row(row: dict) -> dict:
    return {
        "segment_id": row["segment_id"],
        "audio_path": row["audio_path"],
        "source_path": row["source_path"],
        "start_sec": as_float(row, "start_sec"),
        "end_sec": as_float(row, "end_sec"),
        "duration_sec": as_float(row, "duration_sec"),
        "text": normalize_text(row["transcript"]),
        "confidence": as_float(row, "confidence"),
        "label_source": row["label_source"],
        "single_speaker": True,
        "has_music": False,
        "chukchi_specific_ratio": as_float(row, "chukchi_specific_ratio"),
        "foreign_song_score": as_float(row, "foreign_song_score"),
        "audio_class": row.get("audio_class", ""),
        "ranking_score": round(quality_score(row), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-hours", type=float, default=1.0)
    parser.add_argument("--min-duration-sec", type=float, default=4.0)
    parser.add_argument("--max-duration-sec", type=float, default=12.0)
    parser.add_argument("--min-confidence", type=float, default=0.97)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    candidates = [
        row
        for row in rows
        if usable(row, args.min_duration_sec, args.max_duration_sec, args.min_confidence)
    ]
    candidates.sort(key=quality_score, reverse=True)

    target_seconds = args.target_hours * 3600
    selected: list[dict] = []
    duration = 0.0
    for row in candidates:
        selected.append(convert_row(row))
        duration += as_float(row, "duration_sec")
        if duration >= target_seconds:
            break

    if duration < target_seconds * 0.9:
        raise RuntimeError(
            f"Only selected {duration / 3600:.3f}h from {args.input}; "
            "lower --min-confidence or widen duration limits."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for row in selected:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "candidates": len(candidates),
        "selected": len(selected),
        "target_hours": args.target_hours,
        "duration_hours": round(duration / 3600, 4),
        "min_confidence": args.min_confidence,
        "min_duration_sec": args.min_duration_sec,
        "max_duration_sec": args.max_duration_sec,
        "mean_confidence": round(sum(row["confidence"] for row in selected) / len(selected), 6),
        "min_selected_confidence": round(min(row["confidence"] for row in selected), 6),
        "max_selected_confidence": round(max(row["confidence"] for row in selected), 6),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
