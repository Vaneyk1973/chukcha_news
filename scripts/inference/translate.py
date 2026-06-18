#!/usr/bin/env python3
"""Translate text with a trained or baseline directional model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402
from chukcha_news.mt.metrics import normalize_chukchi_detokenization  # noqa: E402
from chukcha_news.mt.modeling import (  # noqa: E402
    configure_tokenizer,
    ensure_language_token,
    ensure_vocabulary_tokens,
    generation_kwargs,
    prefer_max_new_tokens,
)
from chukcha_news.mt.quality import mt_generation_quality_args  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--config", default="configs/mt.yaml")
    parser.add_argument("--direction", choices=["ru_ckt", "ckt_ru"], required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--model")
    return parser.parse_args()


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    config = load_yaml(args.config)
    direction = config["directions"][args.direction]
    model_path = args.model
    if not model_path:
        model_path = (
            direction["baseline_model"]
            if args.baseline
            else str(resolve_path(direction["output_dir"]) / "final")
        )

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install the mt dependency group: pip install -e '.[mt]'") from error

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    prefer_max_new_tokens(model)
    if not args.baseline:
        ensure_vocabulary_tokens(tokenizer, model, config["tokenizer"]["additional_tokens"])
        chukchi_side = "target" if args.direction == "ru_ckt" else "source"
        ensure_language_token(
            tokenizer,
            model,
            direction[f"{chukchi_side}_language"],
            direction[f"initialize_{chukchi_side}_language_from"],
        )
        configure_tokenizer(tokenizer, direction)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True).to(device)
    generation = {
        "num_beams": config["training"]["generation_num_beams"],
        "max_new_tokens": config["training"]["max_target_length"],
    }
    generation.update(mt_generation_quality_args(args.direction))
    if not args.baseline:
        generation.update(generation_kwargs(tokenizer, direction))
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation)
    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    if direction["target_column"] == "ckt":
        output = normalize_chukchi_detokenization(output)
    print(output)


if __name__ == "__main__":
    main()
