#!/usr/bin/env python3
"""Prepare deterministic, deduplicated bidirectional MT splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402


def normalize_text(text: str) -> str:
    """Normalize text for this pipeline stage."""
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_name(key: str, seed: str, train_ratio: float, validation_ratio: float) -> str:
    """Split name for this pipeline stage."""
    digest = hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write jsonl for this pipeline stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mt.yaml")
    return parser.parse_args()


def word_count(text: str) -> int:
    """Word count for this pipeline stage."""
    return len(re.findall(r"[A-Za-zА-Яа-яЁёӃӄӇӈԒԓ']+", text))


def main() -> None:
    """Run the command-line workflow for this module."""
    config = load_yaml(parse_args().config)
    data_config = config["data"]
    source_path = resolve_path(data_config["source_path"])
    ratios = [
        data_config["train_ratio"],
        data_config["validation_ratio"],
        data_config["test_ratio"],
    ]
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("MT split ratios must sum to 1.0")

    accepted = []
    rejection_counts = Counter()
    seen_pairs = set()
    source_counts = Counter()
    exclude_sources = set(data_config.get("exclude_sources", []))
    max_per_source = data_config.get("max_per_source", {})
    min_source_words = int(data_config.get("min_source_words", 1))
    min_target_words = int(data_config.get("min_target_words", 1))
    with source_path.open("r", encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            ru = normalize_text(row.get("ru", ""))
            ckt = normalize_text(row.get("ckt", ""))
            try:
                score = float(row.get("score", 0.0))
            except ValueError:
                score = 0.0

            if not ru or not ckt:
                rejection_counts["empty"] += 1
                continue
            source = normalize_text(row.get("source", "")) or "unknown"
            if source in exclude_sources:
                rejection_counts["excluded_source"] += 1
                continue
            if score < data_config["min_score"]:
                rejection_counts["low_score"] += 1
                continue
            if word_count(ru) < min_source_words or word_count(ckt) < min_target_words:
                rejection_counts["too_few_words"] += 1
                continue
            if (
                len(ru) > data_config["max_source_chars"]
                or len(ckt) > data_config["max_target_chars"]
            ):
                rejection_counts["too_long"] += 1
                continue
            source_cap = max_per_source.get(source)
            if source_cap is not None and source_counts[source] >= int(source_cap):
                rejection_counts["source_cap"] += 1
                continue

            pair_key = f"{ru}\0{ckt}".casefold()
            if pair_key in seen_pairs:
                rejection_counts["duplicate"] += 1
                continue
            seen_pairs.add(pair_key)

            split = split_name(
                f"{source}\0{pair_key}",
                data_config["split_seed"],
                data_config["train_ratio"],
                data_config["validation_ratio"],
            )
            accepted.append(
                {"ru": ru, "ckt": ckt, "score": score, "source": source, "split": split}
            )
            source_counts[source] += 1

    output_root = resolve_path(data_config["output_dir"])
    stats = {
        "source_path": str(data_config["source_path"]),
        "accepted_pairs": len(accepted),
        "rejected": dict(rejection_counts),
        "splits": Counter(row["split"] for row in accepted),
        "sources": Counter(row["source"] for row in accepted),
    }

    for direction, direction_config in config["directions"].items():
        source_column = direction_config["source_column"]
        target_column = direction_config["target_column"]
        for split in ("train", "validation", "test"):
            rows = [
                {
                    "id": hashlib.sha256(f"{row['ru']}\0{row['ckt']}".encode("utf-8")).hexdigest()[
                        :16
                    ],
                    "source_text": row[source_column],
                    "target_text": row[target_column],
                    "score": row["score"],
                    "corpus_source": row["source"],
                }
                for row in accepted
                if row["split"] == split
            ]
            write_jsonl(resolve_path(direction_config[f"{split}_file"]), rows)

    output_root.mkdir(parents=True, exist_ok=True)
    stats_path = output_root / "stats.json"
    stats["splits"] = dict(stats["splits"])
    stats["sources"] = dict(stats["sources"])
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stats_path": str(stats_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
