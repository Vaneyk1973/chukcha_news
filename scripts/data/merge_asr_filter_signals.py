#!/usr/bin/env python3
"""Merge ASR filtering signals into final keep/uncertain/reject manifests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "interim" / "asr_manifest.csv"
DEFAULT_TEXT_FILTER = ROOT / "data" / "interim" / "asr_text_filtered" / "all.csv"
DEFAULT_CHUKCHI_SCORES = ROOT / "data" / "interim" / "asr_chukchi_text_scores" / "all.csv"
DEFAULT_AUDIO_CLASSES = ROOT / "data" / "interim" / "asr_audio_classes" / "all.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "asr_clean"
DEFAULT_REPORT = ROOT / "reports" / "asr_cleaning.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--text-filter", type=Path, default=DEFAULT_TEXT_FILTER)
    parser.add_argument("--chukchi-scores", type=Path, default=DEFAULT_CHUKCHI_SCORES)
    parser.add_argument("--audio-classes", type=Path, default=DEFAULT_AUDIO_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_rows(path: Path, limit: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return rows[:limit] if limit else rows


def read_optional_index(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["segment_id"]: row for row in read_rows(path)}


def split_reasons(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def final_decision(row: dict, text_row: dict | None, score_row: dict | None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    text_reasons = split_reasons((text_row or {}).get("text_filter_reasons"))
    if text_row is None:
        reasons.append("missing_text_filter")
    reasons.extend(text_reasons)

    chukchi_class = (score_row or {}).get("chukchi_text_class", "")
    if score_row is None:
        reasons.append("missing_chukchi_score")
    elif chukchi_class == "reject":
        reasons.append("chukchi_lm_reject")
    elif chukchi_class == "uncertain":
        reasons.append("chukchi_lm_uncertain")
    elif chukchi_class != "chukchi_text":
        reasons.append("unknown_chukchi_lm_class")

    if any(reason not in {"chukchi_lm_uncertain"} for reason in reasons):
        return "rejected", reasons
    if reasons:
        return "uncertain", reasons
    return "kept", reasons


def merge_row(row: dict, text_row: dict | None, score_row: dict | None, audio_row: dict | None) -> dict:
    verdict, reasons = final_decision(row, text_row, score_row)
    merged = dict(row)

    for source in (audio_row, text_row, score_row):
        if not source:
            continue
        for key, value in source.items():
            if key not in merged or merged[key] == "":
                merged[key] = value

    merged["asr_clean_verdict"] = verdict
    merged["asr_clean_reasons"] = ",".join(reasons)
    return merged


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fieldnames(rows: list[dict]) -> list[str]:
    seen = set()
    fields = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input, args.limit)
    text_by_id = read_optional_index(args.text_filter)
    scores_by_id = read_optional_index(args.chukchi_scores)
    audio_by_id = read_optional_index(args.audio_classes)

    merged = [
        merge_row(
            row,
            text_by_id.get(row["segment_id"]),
            scores_by_id.get(row["segment_id"]),
            audio_by_id.get(row["segment_id"]),
        )
        for row in rows
    ]

    by_verdict = {
        "kept": [row for row in merged if row["asr_clean_verdict"] == "kept"],
        "uncertain": [row for row in merged if row["asr_clean_verdict"] == "uncertain"],
        "rejected": [row for row in merged if row["asr_clean_verdict"] == "rejected"],
    }
    reason_counts = Counter(
        reason
        for row in merged
        for reason in split_reasons(row.get("asr_clean_reasons"))
    )

    fields = fieldnames(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all.csv", merged, fields)
    for verdict, verdict_rows in by_verdict.items():
        write_csv(args.output_dir / f"{verdict}.csv", verdict_rows, fields)

    report = {
        "input": str(args.input),
        "text_filter": str(args.text_filter) if args.text_filter.exists() else None,
        "chukchi_scores": str(args.chukchi_scores) if args.chukchi_scores.exists() else None,
        "audio_classes": str(args.audio_classes) if args.audio_classes.exists() else None,
        "output_dir": str(args.output_dir),
        "rows": len(merged),
        "counts": {key: len(value) for key, value in by_verdict.items()},
        "reason_counts": dict(reason_counts),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
