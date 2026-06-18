#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "data" / "raw" / "hse_parallel_corpus" / "data.csv"
TARGET_PATH = ROOT / "data" / "interim" / "parallel_corpus.csv"


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source corpus: {SOURCE_PATH}")

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SOURCE_PATH.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if not rows:
        raise ValueError("Parallel corpus is empty")

    fieldnames = list(rows[0].keys())
    with TARGET_PATH.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {TARGET_PATH}")


if __name__ == "__main__":
    main()
