#!/usr/bin/env python3
"""Report training pipeline readiness without starting any training."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def csv_has_rows(path: str) -> bool:
    resolved = ROOT / path
    if not resolved.exists():
        return False
    with resolved.open("r", encoding="utf-8", newline="") as input_file:
        return next(csv.DictReader(input_file), None) is not None


def main() -> None:
    mt = load_config("mt")
    asr = load_config("asr")
    tts = load_config("tts")
    fieldasr_ready = all(
        csv_has_rows(f"data/processed/asr/fieldasr/{split}.csv")
        for split in ("train", "validation", "test")
    )

    status = {
        "order": ["mt", "asr", "tts"],
        "mt": {
            "ready": all(
                exists(direction[f"{split}_file"])
                for direction in mt["directions"].values()
                for split in ("train", "validation", "test")
            ),
            "training_enabled": mt["training"]["enabled"],
            "next_command": "make mt-model-smoke",
        },
        "asr": {
            "ready": csv_has_rows(asr["data"]["audio_manifest"]),
            "supervised_data_ready": fieldasr_ready,
            "training_enabled": asr["training"]["enabled"],
            "next_command": "make prepare-fieldasr",
            "blocker": "ASR manifest with non-empty segment pseudo-labels is not prepared.",
        },
        "tts": {
            "ready": exists(tts["data"]["manifest"]),
            "training_enabled": tts["training"]["enabled"],
            "blocker": "Automatically filtered pseudo-label manifest is not prepared.",
        },
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
