#!/usr/bin/env python3
"""Build a deduplicated Chukchi monolingual corpus from downloaded resources."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HSE_CORPUS = ROOT / "data" / "raw" / "hse_parallel_corpus" / "data.csv"
DEFAULT_FIELDASR_CORPUS = ROOT / "external" / "fieldasr" / "data" / "text_corpus.txt"
DEFAULT_ASR_MANIFEST = ROOT / "data" / "interim" / "asr_manifest.csv"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "chukchi_monolingual.txt"


def normalize_text(text: str) -> str:
    """Normalize text for this pipeline stage."""
    return re.sub(r"\s+", " ", text).strip()


def load_hse_sentences(path: Path) -> list[str]:
    """Load hse sentences for this pipeline stage."""
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [row["ckt"] for row in csv.DictReader(input_file) if row.get("ckt")]


def load_text_lines(path: Path) -> list[str]:
    """Load text lines for this pipeline stage."""
    return path.read_text(encoding="utf-8").splitlines()


def load_asr_transcripts(path: Path) -> list[str]:
    """Load asr transcripts for this pipeline stage."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [row["transcript"] for row in csv.DictReader(input_file) if row.get("transcript")]


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hse-corpus", type=Path, default=DEFAULT_HSE_CORPUS)
    parser.add_argument("--fieldasr-corpus", type=Path, default=DEFAULT_FIELDASR_CORPUS)
    parser.add_argument("--asr-manifest", type=Path, default=DEFAULT_ASR_MANIFEST)
    parser.add_argument("--include-asr", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    sentences = load_hse_sentences(args.hse_corpus)
    sentences.extend(load_text_lines(args.fieldasr_corpus))
    if args.include_asr:
        sentences.extend(load_asr_transcripts(args.asr_manifest))

    normalized = {normalize_text(sentence) for sentence in sentences}
    normalized.discard("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sorted(normalized)) + "\n", encoding="utf-8")
    print(f"Wrote {len(normalized)} unique Chukchi sentences to {args.output}")


if __name__ == "__main__":
    main()
