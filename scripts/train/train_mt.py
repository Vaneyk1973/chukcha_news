#!/usr/bin/env python3
"""Fine-tune one directional NLLB translation model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402
from chukcha_news.mt.modeling import (  # noqa: E402
    configure_tokenizer,
    ensure_language_token,
    ensure_vocabulary_tokens,
)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mt.yaml")
    parser.add_argument("--direction", choices=["ru_ckt", "ckt_ru"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def validate(config: dict, direction: str) -> dict:
    direction_config = config["directions"][direction]
    files = {
        split: resolve_path(direction_config[f"{split}_file"])
        for split in ("train", "validation", "test")
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prepared MT files: {missing}")
    return {
        "direction": direction,
        "base_model": direction_config["base_model"],
        "source_language": direction_config["source_language"],
        "target_language": direction_config["target_language"],
        "output_dir": str(resolve_path(direction_config["output_dir"])),
        "examples": {split: count_jsonl(path) for split, path in files.items()},
        "training_enabled_in_config": config["training"]["enabled"],
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    summary = validate(config, args.direction)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if not config["training"]["enabled"]:
        raise RuntimeError(
            "Training is disabled. Set training.enabled=true in configs/mt.yaml explicitly."
        )

    try:
        import evaluate
        import numpy as np
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install the mt dependency group before training: pip install -e '.[mt]'"
        ) from error

    direction_config = config["directions"][args.direction]
    training = config["training"]
    if training["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("MT training requires CUDA. Install a CUDA-enabled PyTorch build.")
    set_seed(training["seed"])

    dataset = load_dataset(
        "json",
        data_files={
            split: str(resolve_path(direction_config[f"{split}_file"]))
            for split in ("train", "validation", "test")
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(direction_config["base_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(direction_config["base_model"])
    ensure_vocabulary_tokens(tokenizer, model, config["tokenizer"]["additional_tokens"])
    chukchi_side = "target" if args.direction == "ru_ckt" else "source"
    ensure_language_token(
        tokenizer,
        model,
        direction_config[f"{chukchi_side}_language"],
        direction_config[f"initialize_{chukchi_side}_language_from"],
    )
    configure_tokenizer(tokenizer, direction_config)
    model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(
        direction_config["target_language"]
    )
    model.config.use_cache = not training["gradient_checkpointing"]

    def tokenize(batch: dict) -> dict:
        model_inputs = tokenizer(
            batch["source_text"],
            max_length=training["max_source_length"],
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=training["max_target_length"],
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_prediction) -> dict:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = [
            text.strip() for text in tokenizer.batch_decode(predictions, skip_special_tokens=True)
        ]
        decoded_labels = [
            text.strip() for text in tokenizer.batch_decode(labels, skip_special_tokens=True)
        ]
        return {
            "bleu": sacrebleu.compute(
                predictions=decoded_predictions, references=[[x] for x in decoded_labels]
            )["score"],
            "chrf": chrf.compute(
                predictions=decoded_predictions, references=decoded_labels
            )["score"],
        }

    output_dir = resolve_path(direction_config["output_dir"])
    args_dict = {
        key: value
        for key, value in training.items()
        if key
        in {
            "seed",
            "num_train_epochs",
            "learning_rate",
            "weight_decay",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "gradient_checkpointing",
            "fp16",
            "bf16",
            "optim",
            "eval_steps",
            "save_steps",
            "logging_steps",
            "save_total_limit",
            "predict_with_generate",
            "report_to",
        }
    }
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        generation_num_beams=training["generation_num_beams"],
        **args_dict,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
