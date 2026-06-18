"""Regression tests for the Chukchi News Voice pipeline."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap" / "download_mt_models.py"
SPEC = importlib.util.spec_from_file_location("download_mt_models", SCRIPT)
assert SPEC and SPEC.loader
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


def test_model_ids_are_unique_and_include_baselines() -> None:
    """
    Exercise the `test_model_ids_are_unique_and_include_baselines` behavior and guard against regressions.
    """
    config = {
        "directions": {
            "a": {"base_model": "base", "baseline_model": "baseline-a"},
            "b": {"base_model": "base", "baseline_model": "baseline-b"},
        }
    }

    assert downloader.model_ids(config) == ["base", "baseline-a", "baseline-b"]
    assert downloader.model_ids(config, include_baselines=False) == ["base"]
