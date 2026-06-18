#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import requests
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "resources.yaml"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, target_path: Path) -> None:
    ensure_parent(target_path)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target_path.write_bytes(response.content)


def clone_repo(url: str, target_path: Path) -> None:
    if target_path.exists():
        return
    ensure_parent(target_path)
    subprocess.run(["git", "clone", "--depth", "1", url, str(target_path)], check=True)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()

    for entry in config.get("datasets", {}).values():
        target_path = ROOT / entry["target_path"]
        if not target_path.exists():
            print(f"Downloading dataset: {entry['description']}")
            download_file(entry["url"], target_path)

    for entry in config.get("repositories", {}).values():
        target_path = ROOT / entry["target_path"]
        print(f"Ensuring repository: {entry['description']}")
        clone_repo(entry["url"], target_path)

    for entry in config.get("metadata", {}).get("model_cards", []):
        target_path = ROOT / entry["target_path"]
        if not target_path.exists():
            print(f"Downloading model card: {entry['name']}")
            download_file(entry["url"], target_path)

    print("Bootstrap resources are ready.")


if __name__ == "__main__":
    main()
