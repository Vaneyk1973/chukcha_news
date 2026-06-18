#!/usr/bin/env python3
"""Diagnose which MT direction breaks semantic roundtrips."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402
from chukcha_news.mt.metrics import character_error_rate, normalize_chukchi_detokenization  # noqa: E402
from chukcha_news.mt.modeling import (  # noqa: E402
    configure_tokenizer,
    ensure_language_token,
    ensure_vocabulary_tokens,
    generation_kwargs,
    prefer_max_new_tokens,
)
from chukcha_news.mt.quality import (  # noqa: E402
    has_repetition_collapse,
    mt_generation_quality_args,
    repetition_features,
)


WORD_RE = re.compile(r"[А-Яа-яЁёӃӄӇӈԒԓA-Za-z']+")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mt.yaml")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", default="mt-diagnostic-v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, default=Path("reports/mt/roundtrip_100.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/mt/roundtrip_100.json"))
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    """Read jsonl for this pipeline stage."""
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def word_similarity(left: str, right: str) -> float:
    """Word similarity for this pipeline stage."""
    left_tokens = [token.casefold() for token in WORD_RE.findall(left)]
    right_tokens = [token.casefold() for token in WORD_RE.findall(right)]
    if not left_tokens or not right_tokens:
        return 0.0
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    jaccard = len(left_set & right_set) / max(len(left_set | right_set), 1)
    seq = difflib.SequenceMatcher(None, " ".join(left_tokens), " ".join(right_tokens)).ratio()
    return 0.55 * seq + 0.45 * jaccard


class MTModel:
    """Document the state and behavior for the `MTModel` component."""

    def __init__(self, config: dict, direction_key: str) -> None:
        """Implement the `__init__` protocol hook for this object."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.config = config
        self.direction_key = direction_key
        self.direction = config["directions"][direction_key]
        model_path = resolve_path(self.direction["output_dir"]) / "final"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        prefer_max_new_tokens(self.model)
        ensure_vocabulary_tokens(
            self.tokenizer, self.model, config["tokenizer"]["additional_tokens"]
        )
        chukchi_side = "target" if direction_key == "ru_ckt" else "source"
        ensure_language_token(
            self.tokenizer,
            self.model,
            self.direction[f"{chukchi_side}_language"],
            self.direction[f"initialize_{chukchi_side}_language_from"],
        )
        configure_tokenizer(self.tokenizer, self.direction)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def translate_batch(self, texts: list[str], batch_size: int) -> list[str]:
        """Translate batch for this pipeline stage."""
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["training"]["max_source_length"],
            ).to(self.device)
            generation = {
                "num_beams": self.config["training"]["generation_num_beams"],
                "max_new_tokens": self.config["training"]["max_target_length"],
            }
            generation.update(mt_generation_quality_args(self.direction_key))
            generation.update(generation_kwargs(self.tokenizer, self.direction))
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, **generation)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            if self.direction["target_column"] == "ckt":
                decoded = [normalize_chukchi_detokenization(text) for text in decoded]
            outputs.extend(decoded)
            print(
                f"[{self.direction_key}] {min(start + len(batch), len(texts))}/{len(texts)}",
                flush=True,
            )
        return outputs

    def close(self) -> None:
        """Close for this pipeline stage."""
        del self.model
        del self.tokenizer
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def mean(values: list[float]) -> float:
    """Mean for this pipeline stage."""
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    config = load_yaml(args.config)
    rows = read_jsonl(resolve_path(config["directions"]["ru_ckt"]["test_file"]))
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.limit, len(rows)))

    ru_sources = [row["source_text"] for row in sample]
    gold_ckt = [row["target_text"] for row in sample]

    ru_ckt = MTModel(config, "ru_ckt")
    generated_ckt = ru_ckt.translate_batch(ru_sources, args.batch_size)
    ru_ckt.close()

    ckt_ru = MTModel(config, "ckt_ru")
    gold_back_ru = ckt_ru.translate_batch(gold_ckt, args.batch_size)
    generated_back_ru = ckt_ru.translate_batch(generated_ckt, args.batch_size)
    ckt_ru.close()

    out_rows = []
    for index, row in enumerate(sample):
        direct_cer = character_error_rate(
            normalize_chukchi_detokenization(row["target_text"]),
            normalize_chukchi_detokenization(generated_ckt[index]),
        )
        gold_back_sim = word_similarity(row["source_text"], gold_back_ru[index])
        generated_back_sim = word_similarity(row["source_text"], generated_back_ru[index])
        repeat_collapse = has_repetition_collapse(generated_ckt[index])
        suspected_failure = "ru_ckt"
        if repeat_collapse:
            suspected_failure = "ru_ckt_repeat_collapse"
        elif gold_back_sim < 0.25 and generated_back_sim < 0.25:
            suspected_failure = "ckt_ru_or_both"
        elif gold_back_sim >= 0.25 and generated_back_sim < 0.25:
            suspected_failure = "ru_ckt"
        elif gold_back_sim < 0.25 and generated_back_sim >= 0.25:
            suspected_failure = "ckt_ru_on_gold"
        elif direct_cer > 0.75:
            suspected_failure = "ru_ckt_form"
        else:
            suspected_failure = "unclear_or_ok"
        out_rows.append(
            {
                **repetition_features(generated_ckt[index]),
                "repeat_collapse": repeat_collapse,
            }
            | {
                "id": row["id"],
                "corpus_source": row.get("corpus_source", ""),
                "ru_source": row["source_text"],
                "gold_ckt": row["target_text"],
                "generated_ckt": generated_ckt[index],
                "gold_ckt_to_ru": gold_back_ru[index],
                "generated_ckt_to_ru": generated_back_ru[index],
                "direct_ckt_cer": f"{direct_cer:.4f}",
                "gold_back_ru_similarity": f"{gold_back_sim:.4f}",
                "generated_back_ru_similarity": f"{generated_back_sim:.4f}",
                "suspected_failure": suspected_failure,
            }
        )

    args.output_csv = resolve_path(args.output_csv)
    args.report = resolve_path(args.report)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    report: dict[str, Any] = {
        "examples": len(out_rows),
        "output_csv": str(args.output_csv),
        "mean_direct_ckt_cer": mean([float(row["direct_ckt_cer"]) for row in out_rows]),
        "mean_gold_back_ru_similarity": mean(
            [float(row["gold_back_ru_similarity"]) for row in out_rows]
        ),
        "mean_generated_back_ru_similarity": mean(
            [float(row["generated_back_ru_similarity"]) for row in out_rows]
        ),
        "suspected_failure_counts": {},
        "repeat_collapse_count": sum(1 for row in out_rows if row["repeat_collapse"]),
    }
    for row in out_rows:
        key = row["suspected_failure"]
        report["suspected_failure_counts"][key] = report["suspected_failure_counts"].get(key, 0) + 1
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
