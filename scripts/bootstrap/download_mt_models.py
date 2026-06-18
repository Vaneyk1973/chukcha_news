#!/usr/bin/env python3
"""Download all model repositories needed by the MT pipeline into the HF cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mt.yaml")
    parser.add_argument("--skip-baselines", action="store_true")
    return parser.parse_args()


def model_ids(config: dict, include_baselines: bool = True) -> list[str]:
    """Model ids for this pipeline stage."""
    models = {direction["base_model"] for direction in config["directions"].values()}
    if include_baselines:
        models.update(direction["baseline_model"] for direction in config["directions"].values())
    return sorted(models)


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    config = load_yaml(args.config)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install MT dependencies with: make setup-mt") from error

    for model_id in model_ids(config, include_baselines=not args.skip_baselines):
        print(f"Downloading {model_id}...", flush=True)
        path = snapshot_download(repo_id=model_id)
        print(f"Cached {model_id} at {path}", flush=True)


if __name__ == "__main__":
    main()
