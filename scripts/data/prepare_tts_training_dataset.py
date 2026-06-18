#!/usr/bin/env python3
"""Prepare local JSONL train/eval files for VITS fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tts.yaml")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def convert_row(row: dict) -> dict:
    return {
        "segment_id": str(row["segment_id"]),
        "audio": str(resolve_path(row["audio_path"])),
        "text": str(row["text"]).strip(),
        "speaker_id": int(row.get("speaker_id", 0) or 0),
        "duration_sec": float(row["duration_sec"]),
    }


def split_rows(rows: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * train_ratio))
    if len(shuffled) > 1:
        split_at = min(split_at, len(shuffled) - 1)
    return shuffled[:split_at], shuffled[split_at:]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_config = config["data"]
    training_config = config["training"]
    manifest = resolve_path(data_config["manifest"])
    output_dir = resolve_path(data_config["training_dataset_dir"])

    rows = [convert_row(row) for row in read_jsonl(manifest)]
    rows = [row for row in rows if row["text"] and Path(row["audio"]).exists()]
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) < 2:
        raise RuntimeError(f"Need at least two TTS rows, got {len(rows)} from {manifest}")

    train_rows, eval_rows = split_rows(
        rows,
        float(training_config["train_split_ratio"]),
        int(training_config["seed"]),
    )
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    report = {
        "input": str(manifest),
        "output_dir": str(output_dir),
        "train": str(train_path),
        "eval": str(eval_path),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "duration_hours": round(sum(row["duration_sec"] for row in rows) / 3600, 4),
    }
    report_path = ROOT / "reports" / "tts_training_dataset.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
