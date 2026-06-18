"""Reusable package module for Chukchi News Voice."""

from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve path for this pipeline stage."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict:
    """Load yaml for this pipeline stage."""
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
