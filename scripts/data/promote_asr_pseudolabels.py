#!/usr/bin/env python3
"""Promote accepted ASR pseudo-labels into downstream text/audio artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "interim" / "asr_clean" / "kept.csv"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "tts_pseudo_labels.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "asr_pseudolabel_promotion.json"


def normalize_text(text: str) -> str:
    """Normalize text for this pipeline stage."""
    return re.sub(r"\s+", " ", text).strip()


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--assume-single-speaker", action="store_true", default=True)
    parser.add_argument("--assume-no-music", action="store_true", default=True)
    return parser.parse_args()


def convert_row(row: dict, assume_single_speaker: bool, assume_no_music: bool) -> dict:
    """Convert row for this pipeline stage."""
    return {
        "segment_id": row["segment_id"],
        "audio_path": row["audio_path"],
        "source_path": row["source_path"],
        "start_sec": float(row["start_sec"]),
        "end_sec": float(row["end_sec"]),
        "duration_sec": float(row["duration_sec"]),
        "text": normalize_text(row["transcript"]),
        "confidence": float(row["confidence"]),
        "label_source": row["label_source"],
        "single_speaker": assume_single_speaker,
        "has_music": not assume_no_music,
    }


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    rows = []
    skipped_empty = 0
    with args.input.open("r", encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            if not normalize_text(row.get("transcript", "")):
                skipped_empty += 1
                continue
            rows.append(
                convert_row(
                    row,
                    assume_single_speaker=args.assume_single_speaker,
                    assume_no_music=args.assume_no_music,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    duration_hours = sum(float(row["duration_sec"]) for row in rows) / 3600
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "accepted": len(rows),
        "skipped_empty": skipped_empty,
        "duration_hours": round(duration_hours, 4),
        "assumptions": {
            "single_speaker": args.assume_single_speaker,
            "has_music": not args.assume_no_music,
            "transcripts_are_trusted": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
