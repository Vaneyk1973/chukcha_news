#!/usr/bin/env python3
"""Train an experimental Chukchi-form LoRA adapter for a local LLM."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm_chukchi.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return max(checkpoints)[1]


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data = config["data"]
    training = config["training"]

    train_file = resolve_path(data["train_file"])
    eval_file = resolve_path(data["validation_file"])
    if not train_file.exists() or not eval_file.exists():
        raise RuntimeError("Run scripts/data/prepare_llm_chukchi_dataset.py first.")
    if not training["enabled"]:
        raise RuntimeError("LLM training is disabled in config.")

    output_dir = resolve_path(training["output_dir"])
    if args.dry_run:
        print(
            {
                "base_model": config["model"]["base_model"],
                "train_file": str(train_file),
                "eval_file": str(eval_file),
                "output_dir": str(output_dir),
            }
        )
        return

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError("Install LLM dependencies with: python3 -m pip install -e '.[llm]'") from error

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if bool(training["load_in_4bit"]):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bool(training["bf16"]) else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16 if bool(training["bf16"]) else torch.float16,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    def format_example(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    do_eval = bool(training.get("do_eval", True))
    data_files = {"train": str(train_file)}
    if do_eval:
        data_files["eval"] = str(eval_file)
    dataset = load_dataset("json", data_files=data_files)
    max_train_samples = training.get("max_train_samples")
    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(
            range(min(int(max_train_samples), len(dataset["train"])))
        )
    max_eval_samples = training.get("max_eval_samples")
    if do_eval and max_eval_samples is not None:
        dataset["eval"] = dataset["eval"].select(
            range(min(int(max_eval_samples), len(dataset["eval"])))
        )

    lora_config = LoraConfig(
        r=int(training["lora_r"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(training["target_modules"]),
    )

    sft_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training["num_train_epochs"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "warmup_steps": int(training.get("warmup_steps", 0)),
        "logging_steps": int(training["logging_steps"]),
        "save_steps": int(training["save_steps"]),
        "eval_strategy": "steps" if do_eval else "no",
        "save_strategy": "steps",
        "bf16": bool(training["bf16"]),
        "fp16": bool(training["fp16"]),
        "prediction_loss_only": bool(training.get("prediction_loss_only", True)),
        "report_to": [],
        "dataset_text_field": "text",
        "remove_unused_columns": True,
    }
    if do_eval and training.get("eval_steps") is not None:
        sft_kwargs["eval_steps"] = int(training["eval_steps"])
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in sft_params:
        sft_kwargs["max_length"] = int(training["max_seq_length"])
    elif "max_seq_length" in sft_params:
        sft_kwargs["max_seq_length"] = int(training["max_seq_length"])
    if "loss_type" in sft_params and training.get("loss_type"):
        sft_kwargs["loss_type"] = str(training["loss_type"])
    sft_config = SFTConfig(**sft_kwargs)

    train_dataset = dataset["train"].map(lambda row: {"text": format_example(row)})
    eval_dataset = (
        dataset["eval"].map(lambda row: {"text": format_example(row)}) if do_eval else None
    )
    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "peft_config": lora_config,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    resume_from = latest_checkpoint(output_dir) if bool(training.get("auto_resume", True)) else None
    if resume_from:
        print({"resume_from_checkpoint": str(resume_from)}, flush=True)
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print({"adapter": str(output_dir)})


if __name__ == "__main__":
    main()
