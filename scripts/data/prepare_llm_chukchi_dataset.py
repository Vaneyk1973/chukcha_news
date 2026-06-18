#!/usr/bin/env python3
"""Prepare an experimental Chukchi LLM SFT dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402


CHUKCHI_CHARS = set("ӃӄӇӈԒԓ")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁёӃӄӇӈԒԓ]")
LATIN_RE = re.compile(r"[A-Za-z]")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm_chukchi.yaml")
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chukchi_ratio(text: str) -> float:
    cyrillic = CYRILLIC_RE.findall(text)
    if not cyrillic:
        return 0.0
    return sum(char in CHUKCHI_CHARS for char in cyrillic) / len(cyrillic)


def usable_chukchi(text: str, min_chars: int = 20, max_chars: int = 900) -> bool:
    text = normalize(text)
    if not min_chars <= len(text) <= max_chars:
        return False
    if HAN_RE.search(text) or LATIN_RE.search(text):
        return False
    if len(CYRILLIC_RE.findall(text)) < min_chars * 0.55:
        return False
    return chukchi_ratio(text) >= 0.015


def chat_example(system: str, user: str, assistant: str, source: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
    }


def load_monolingual(path: Path, limit: int, rng: random.Random) -> list[dict]:
    lines = [normalize(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if usable_chukchi(line, min_chars=40, max_chars=900)]
    rng.shuffle(lines)
    system = "Ты пишешь только на чукотском языке кириллицей."
    examples = []
    prompts = [
        "Напиши короткий фрагмент на чукотском языке.",
        "Продолжи в стиле чукотского текста.",
        "Составь короткое сообщение на чукотском языке.",
    ]
    for index, line in enumerate(lines[:limit]):
        examples.append(chat_example(system, prompts[index % len(prompts)], line, "monolingual"))
    return examples


def load_parallel(path: Path, limit: int, rng: random.Random) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            row = json.loads(line)
            ru = normalize(row.get("source_text", ""))
            ckt = normalize(row.get("target_text", ""))
            if ru and usable_chukchi(ckt, min_chars=8, max_chars=900):
                rows.append((ru, ckt))
    rng.shuffle(rows)
    system = (
        "Ты создаешь текст на чукотском языке кириллицей. "
        "Не объясняй ответ, не используй русский в ответе."
    )
    examples = []
    for ru, ckt in rows[:limit]:
        examples.append(
            chat_example(
                system,
                f"Напиши по-чукотски текст, соответствующий русскому смыслу: {ru}",
                ckt,
                "parallel_ru_ckt",
            )
        )
    return examples


def load_asr(path: Path, limit: int, rng: random.Random) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            text = normalize(row.get("transcript", ""))
            try:
                confidence = float(row.get("confidence", 0.0))
                russian_score = float(row.get("russian_score", 0.0) or 0.0)
            except ValueError:
                continue
            if confidence >= 0.97 and russian_score == 0.0 and usable_chukchi(text, 20, 700):
                rows.append(text)
    rng.shuffle(rows)
    system = "Ты пишешь короткие радиофразы на чукотском языке кириллицей."
    examples = []
    for text in rows[:limit]:
        examples.append(
            chat_example(system, "Напиши короткую чукотскую радиофразу.", text, "asr_clean")
        )
    return examples


def split_examples(
    examples: list[dict], validation_size: int, rng: random.Random
) -> tuple[list[dict], list[dict]]:
    shuffled = list(examples)
    rng.shuffle(shuffled)
    validation_size = min(validation_size, max(1, len(shuffled) // 20))
    return shuffled[validation_size:], shuffled[:validation_size]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data = config["data"]
    rng = random.Random(int(data["seed"]))

    monolingual = load_monolingual(
        resolve_path(data["monolingual_trusted"]),
        int(data["max_monolingual_examples"]),
        rng,
    )
    parallel = load_parallel(
        resolve_path(data["mt_ru_ckt_train"]),
        int(data["max_parallel_examples"]),
        rng,
    )
    asr = load_asr(resolve_path(data["asr_clean_kept"]), int(data["max_asr_examples"]), rng)
    examples = monolingual + parallel + asr
    if len(examples) < 100:
        raise RuntimeError(f"Too few LLM examples: {len(examples)}")

    train, validation = split_examples(examples, int(data["validation_size"]), rng)
    train_path = resolve_path(data["train_file"])
    validation_path = resolve_path(data["validation_file"])
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)

    report = {
        "train_file": str(train_path),
        "validation_file": str(validation_path),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "sources": {
            "monolingual": len(monolingual),
            "parallel_ru_ckt": len(parallel),
            "asr_clean": len(asr),
        },
        "warning": (
            "Experimental form/style adaptation dataset. It does not prove semantic correctness."
        ),
    }
    report_path = resolve_path(data["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
