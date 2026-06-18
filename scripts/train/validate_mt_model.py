#!/usr/bin/env python3
"""Validate NLLB tokenizer compatibility without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml  # noqa: E402
from chukcha_news.mt.modeling import (  # noqa: E402
    configure_tokenizer,
    ensure_language_token,
    ensure_vocabulary_tokens,
    generation_kwargs,
)


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mt.yaml")
    return parser.parse_args()


def main() -> None:
    """Run the command-line workflow for this module."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install MT dependencies with: make setup-mt") from error

    config = load_yaml(parse_args().config)
    results = {}
    for name, direction in config["directions"].items():
        tokenizer = AutoTokenizer.from_pretrained(direction["base_model"])
        vocabulary_ids = ensure_vocabulary_tokens(
            tokenizer, None, config["tokenizer"]["additional_tokens"]
        )
        chukchi_side = "target" if name == "ru_ckt" else "source"
        language_id = ensure_language_token(
            tokenizer,
            None,
            direction[f"{chukchi_side}_language"],
            direction[f"initialize_{chukchi_side}_language_from"],
        )
        configure_tokenizer(tokenizer, direction)
        source_text = "Сегодня хорошие новости." if name == "ru_ckt" else "Ԓыгъоравэтԓьэн."
        encoded = tokenizer(source_text)
        chukchi_text = "Ԓыгъоравэтԓьэн ӄытгъэргъын ӈинӄэй."
        chukchi_encoded = tokenizer(chukchi_text, add_special_tokens=False)
        results[name] = {
            "model": direction["base_model"],
            "source_language": direction["source_language"],
            "target_language": direction["target_language"],
            "chukchi_token_id": language_id,
            "forced_bos_token_id": generation_kwargs(tokenizer, direction)["forced_bos_token_id"],
            "encoded_tokens": len(encoded["input_ids"]),
            "added_vocabulary_ids": vocabulary_ids,
            "chukchi_unknown_tokens": sum(
                value == tokenizer.unk_token_id for value in chukchi_encoded["input_ids"]
            ),
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
