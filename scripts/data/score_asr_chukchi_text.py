#!/usr/bin/env python3
"""Score ASR transcripts by similarity to known Chukchi text."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.mt.text_lm import CharNgramLM, generic_cyrillic_score, normalize_text  # noqa: E402


DEFAULT_CORPUS = ROOT / "data" / "interim" / "chukchi_monolingual_trusted.txt"
DEFAULT_ASR = ROOT / "data" / "interim" / "asr_manifest.csv"
DEFAULT_MODEL = ROOT / "models" / "text_lm" / "chukchi_char4_trusted.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "asr_chukchi_text_scores"
DEFAULT_REPORT = ROOT / "reports" / "asr_chukchi_text_scores.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--input", type=Path, default=DEFAULT_ASR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--keep-threshold", type=float, default=-1.4)
    parser.add_argument("--reject-threshold", type=float, default=-2.1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def train_or_load(args: argparse.Namespace) -> CharNgramLM:
    if args.model.exists() and not args.retrain:
        return CharNgramLM.load(args.model)
    texts = args.corpus.read_text(encoding="utf-8").splitlines()
    lm = CharNgramLM.train(texts, order=args.order)
    lm.save(args.model)
    return lm


def read_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return rows[:limit] if limit else rows


def decision(score: float, keep_threshold: float, reject_threshold: float) -> str:
    if score >= keep_threshold:
        return "chukchi_text"
    if score < reject_threshold:
        return "reject"
    return "uncertain"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    lm = train_or_load(args)
    rows = read_rows(args.input, args.limit)
    scored = []
    counts = {"chukchi_text": 0, "uncertain": 0, "reject": 0}
    for row in rows:
        text = row.get("transcript", "")
        chukchi_score = lm.average_log_probability(text)
        generic_score = generic_cyrillic_score(text)
        margin = chukchi_score - generic_score
        label = decision(margin, args.keep_threshold, args.reject_threshold)
        counts[label] += 1
        scored.append(
            {
                **row,
                "normalized_transcript": normalize_text(text),
                "chukchi_lm_score": f"{chukchi_score:.6f}",
                "generic_cyrillic_score": f"{generic_score:.6f}",
                "chukchi_lm_margin": f"{margin:.6f}",
                "chukchi_text_class": label,
            }
        )

    fields = list(csv.DictReader(args.input.open("r", encoding="utf-8", newline="")).fieldnames or [])
    extra = [
        "normalized_transcript",
        "chukchi_lm_score",
        "generic_cyrillic_score",
        "chukchi_lm_margin",
        "chukchi_text_class",
    ]
    fields.extend(field for field in extra if field not in fields)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all.csv", scored, fields)
    for label in counts:
        write_csv(args.output_dir / f"{label}.csv", [row for row in scored if row["chukchi_text_class"] == label], fields)

    report = {
        "input": str(args.input),
        "corpus": str(args.corpus),
        "model": str(args.model),
        "rows": len(scored),
        "counts": counts,
        "thresholds": {
            "keep_margin": args.keep_threshold,
            "reject_margin": args.reject_threshold,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
