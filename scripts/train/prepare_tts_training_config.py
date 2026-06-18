#!/usr/bin/env python3
"""Write the JSON config consumed by the external VITS fine-tuning script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tts.yaml")
    return parser.parse_args()


def optional_int(value):
    return None if value is None else int(value)


def capped_optional_int(value, path: Path):
    requested = optional_int(value)
    if requested is None:
        return None
    with path.open("r", encoding="utf-8") as input_file:
        available = sum(1 for line in input_file if line.strip())
    return min(requested, available)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data = config["data"]
    model = config["model"]
    training = config["training"]

    dataset_dir = resolve_path(data["training_dataset_dir"])
    output_dir = resolve_path(training["output_dir"])
    train_file = dataset_dir / "train.jsonl"
    eval_file = dataset_dir / "eval.jsonl"
    if not train_file.exists() or not eval_file.exists():
        raise RuntimeError("Run scripts/data/prepare_tts_training_dataset.py first.")

    trainer_config = {
        "project_name": config["experiment"]["name"],
        "push_to_hub": False,
        "report_to": ["tensorboard"],
        "overwrite_output_dir": bool(training["overwrite_output_dir"]),
        "output_dir": str(output_dir),
        "dataset_name": "json",
        "train_data_file": str(train_file),
        "eval_data_file": str(eval_file),
        "audio_column_name": "audio",
        "text_column_name": "text",
        "speaker_id_column_name": data["speaker_column"],
        "override_speaker_embeddings": False,
        "full_generation_sample_text": "ԓыгъоравэтԓьэн ӄытгъэргъын",
        "max_duration_in_seconds": float(config["pseudo_labeling"]["max_duration_sec"]),
        "min_duration_in_seconds": float(config["pseudo_labeling"]["min_duration_sec"]),
        "max_tokens_length": 500,
        "model_name_or_path": str(resolve_path(model["training_model"])),
        "preprocessing_num_workers": 1,
        "do_train": True,
        "num_train_epochs": int(training["num_train_epochs"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "gradient_checkpointing": False,
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "adam_beta1": 0.8,
        "adam_beta2": 0.99,
        "warmup_steps": int(training["warmup_steps"]),
        "group_by_length": False,
        "do_eval": True,
        "eval_steps": int(training["eval_steps"]),
        "save_steps": int(training["save_steps"]),
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        "max_train_samples": capped_optional_int(training.get("max_train_samples"), train_file),
        "max_eval_samples": capped_optional_int(training.get("max_eval_samples"), eval_file),
        "do_step_schedule_per_epoch": True,
        "weight_disc": 3,
        "weight_fmaps": 1,
        "weight_gen": 1,
        "weight_kl": 1.5,
        "weight_duration": 1,
        "weight_mel": 35,
        "fp16": bool(training["fp16"]),
        "seed": int(training["seed"]),
    }

    output_path = resolve_path(training["config_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trainer_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"config": str(output_path), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
