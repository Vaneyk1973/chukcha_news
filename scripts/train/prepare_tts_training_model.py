#!/usr/bin/env python3
"""Create a local MMS TTS training checkpoint with discriminator weights."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build_monotonic_align() -> None:
    package_dir = EXTERNAL / "monotonic_align" / "monotonic_align"
    shared_objects = list(package_dir.glob("core*.so"))
    if shared_objects:
        return
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").touch()
    run([sys.executable, "setup.py", "build_ext", "--inplace"], EXTERNAL / "monotonic_align")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_config = config["model"]
    target = resolve_path(model_config["training_model"])
    if target.exists() and not args.force:
        print(json.dumps({"training_model": str(target), "status": "exists"}, indent=2))
        return

    build_monotonic_align()
    target.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "convert_original_discriminator_checkpoint.py",
            "--language_code",
            model_config["language_code"],
            "--pytorch_dump_folder_path",
            str(target),
        ],
        EXTERNAL,
    )
    print(json.dumps({"training_model": str(target), "status": "created"}, indent=2))


if __name__ == "__main__":
    main()
