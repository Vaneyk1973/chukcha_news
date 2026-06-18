#!/usr/bin/env python3
"""Evaluate a directional MT model and save predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402
from chukcha_news.mt.metrics import (  # noqa: E402
    mean_character_error_rate,
    normalize_chukchi_detokenization,
)
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
    parser.add_argument("--config", default="configs/mt.yaml")
    parser.add_argument("--direction", choices=["ru_ckt", "ckt_ru"], required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--label")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--normalize-chukchi", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None) -> list[dict]:
    """Read jsonl for this pipeline stage."""
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    config = load_yaml(args.config)
    direction = config["directions"][args.direction]
    model_path = args.model
    model_label = "custom"
    if not model_path:
        if args.baseline:
            model_path = direction["baseline_model"]
            model_label = "baseline"
        else:
            model_path = str(resolve_path(direction["output_dir"]) / "final")
            model_label = "trained"
    if args.label:
        model_label = args.label

    try:
        import sacrebleu
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install the mt dependency group: pip install -e '.[mt]'") from error

    rows = read_jsonl(resolve_path(direction["test_file"]), args.limit)
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

    predictions = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        inputs = tokenizer(
            [row["source_text"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["training"]["max_source_length"],
        ).to(device)
        generation = {
            "num_beams": config["training"]["generation_num_beams"],
            "max_new_tokens": config["training"]["max_target_length"],
        }
        generation.update(mt_generation_quality_args(args.direction))
        if not args.baseline:
            generation.update(generation_kwargs(tokenizer, direction))
        with torch.inference_mode():
            generated = model.generate(**inputs, **generation)
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    if args.normalize_chukchi and direction["target_column"] == "ckt":
        predictions = [normalize_chukchi_detokenization(prediction) for prediction in predictions]

    references = [row["target_text"] for row in rows]
    metric_predictions = predictions
    metric_references = references
    if args.normalize_chukchi and direction["target_column"] == "ckt":
        metric_references = [
            normalize_chukchi_detokenization(reference) for reference in metric_references
        ]

    metrics = {
        "model": model_path,
        "label": model_label,
        "direction": args.direction,
        "examples": len(rows),
        "normalize_chukchi": args.normalize_chukchi,
        "bleu": sacrebleu.corpus_bleu(metric_predictions, [metric_references]).score,
        "chrf": sacrebleu.corpus_chrf(metric_predictions, [metric_references]).score,
        "cer": mean_character_error_rate(metric_references, metric_predictions),
        "by_source": {},
    }
    sources = sorted({row["corpus_source"] for row in rows})
    for source in sources:
        indexes = [index for index, row in enumerate(rows) if row["corpus_source"] == source]
        source_predictions = [metric_predictions[index] for index in indexes]
        source_references = [metric_references[index] for index in indexes]
        metrics["by_source"][source] = {
            "examples": len(indexes),
            "bleu": sacrebleu.corpus_bleu(source_predictions, [source_references]).score,
            "chrf": sacrebleu.corpus_chrf(source_predictions, [source_references]).score,
            "cer": mean_character_error_rate(source_references, source_predictions),
        }

    output_dir = ROOT / "reports" / "mt" / args.direction / model_label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output_file:
        for row, prediction in zip(rows, predictions):
            output_file.write(
                json.dumps({**row, "prediction": prediction}, ensure_ascii=False) + "\n"
            )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
