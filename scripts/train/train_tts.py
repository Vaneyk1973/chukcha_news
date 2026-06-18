#!/usr/bin/env python3
"""Launch MMS/VITS TTS fine-tuning through the external training script."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402

EXTERNAL = ROOT / "external" / "finetune-hf-vits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tts.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if not config["training"]["enabled"]:
        raise RuntimeError("TTS training is disabled in configs/tts.yaml")
    training_config = resolve_path(config["training"]["config_path"])
    if not training_config.exists():
        raise RuntimeError("Run scripts/train/prepare_tts_training_config.py first.")
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--mixed_precision",
        "fp16",
        "--dynamo_backend",
        "no",
        "run_vits_finetuning.py",
        str(training_config),
    ]
    env = dict(os.environ)
    env.setdefault("TENSORBOARD_LOGGING_DIR", "runs")
    subprocess.run(command, cwd=EXTERNAL, env=env, check=True)


if __name__ == "__main__":
    main()
