#!/usr/bin/env python3
"""Package heavy runtime artifacts needed by `make serve`.

The output is meant to be uploaded as a private Hugging Face dataset/model repo
or copied to another machine. It intentionally excludes training checkpoints,
optimizer states, ASR segments, eval outputs, and virtualenvs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts" / "runtime_bundle"
CHUNK_SIZE = 16 * 1024 * 1024
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
HF_BASE_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": "models/hf/Qwen--Qwen2.5-7B-Instruct",
    "facebook/mms-tts-ckt": "models/hf/facebook--mms-tts-ckt",
}

RUNTIME_PATHS = [
    # Configs needed by the local server runtime and model selectors.
    "configs/server.yaml",
    "configs/mt.yaml",
    "configs/mt_clean.yaml",
    "configs/tts.yaml",
    "configs/tts_top1h.yaml",
    # Translation-memory corpora used by the local server.
    "data/processed/mt",
    "data/processed/mt_clean",
    # MT runtime models exposed in server selectors.
    "models/checkpoints/mt/ru_ckt/final",
    "models/checkpoints/mt/ckt_ru/final",
    "models/checkpoints/mt_clean/ru_ckt/final",
    "models/checkpoints/mt_clean/ckt_ru/final",
    # Direct Chukchi LoRA runtime files only; checkpoint-* dirs are excluded below.
    "models/checkpoints/llm_chukchi_lora/README.md",
    "models/checkpoints/llm_chukchi_lora/adapter_config.json",
    "models/checkpoints/llm_chukchi_lora/adapter_model.safetensors",
    "models/checkpoints/llm_chukchi_lora/chat_template.jinja",
    "models/checkpoints/llm_chukchi_lora/tokenizer.json",
    "models/checkpoints/llm_chukchi_lora/tokenizer_config.json",
    # TTS runtime models; root files only, checkpoint-* dirs are excluded below.
    "models/checkpoints/tts/added_tokens.json",
    "models/checkpoints/tts/config.json",
    "models/checkpoints/tts/model.safetensors",
    "models/checkpoints/tts/preprocessor_config.json",
    "models/checkpoints/tts/tokenizer_config.json",
    "models/checkpoints/tts/vocab.json",
    "models/checkpoints/tts_top1h/added_tokens.json",
    "models/checkpoints/tts_top1h/config.json",
    "models/checkpoints/tts_top1h/model.safetensors",
    "models/checkpoints/tts_top1h/preprocessor_config.json",
    "models/checkpoints/tts_top1h/tokenizer_config.json",
    "models/checkpoints/tts_top1h/vocab.json",
]


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink"],
        default="hardlink",
        help="hardlink is fast and does not duplicate local disk blocks on the same filesystem.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove output before packaging.")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Sha256 for this pipeline stage."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: Path) -> list[Path]:
    """Iter files for this pipeline stage."""
    if path.is_file():
        return [path]
    return [file for file in sorted(path.rglob("*")) if file.is_file()]


def link_or_copy(source: Path, target: Path, mode: str) -> None:
    """Link or copy for this pipeline stage."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    source = source.resolve() if source.is_symlink() else source
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def hf_cache_dir(repo_id: str) -> Path:
    """Hf cache dir for this pipeline stage."""
    return HF_CACHE / f"models--{repo_id.replace('/', '--')}"


def latest_hf_snapshot(repo_id: str) -> Path:
    """Latest hf snapshot for this pipeline stage."""
    snapshots = hf_cache_dir(repo_id) / "snapshots"
    candidates = (
        [path for path in snapshots.iterdir() if path.is_dir()] if snapshots.exists() else []
    )
    if not candidates:
        raise FileNotFoundError(
            f"Missing local HF snapshot for {repo_id}. Download it first with "
            f"`huggingface-cli download {repo_id}` or by running the model once."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def package_hf_base_models(output: Path, mode: str) -> int:
    """Package hf base models for this pipeline stage."""
    copied = 0
    for repo_id, target_relative in HF_BASE_MODELS.items():
        snapshot = latest_hf_snapshot(repo_id)
        print(f"[hf] {repo_id} <- {snapshot}", flush=True)
        for source_file in iter_files(snapshot):
            target = output / target_relative / source_file.relative_to(snapshot)
            link_or_copy(source_file, target, mode)
            copied += 1
            print(f"[package] {target.relative_to(output)}", flush=True)
    return copied


def rewrite_packaged_server_config(output: Path) -> None:
    """Rewrite packaged server config for this pipeline stage."""
    path = output / "configs" / "server.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    qwen_path = HF_BASE_MODELS["Qwen/Qwen2.5-7B-Instruct"]
    mms_path = HF_BASE_MODELS["facebook/mms-tts-ckt"]

    config["llm"]["hf_news_base_model"] = qwen_path
    config["llm"]["hf_base_model"] = qwen_path
    config["tts"]["model_path"] = mms_path

    news_choices = config.get("model_options", {}).get("llm_news", {}).get("choices", {})
    if "hf_qwen" in news_choices:
        news_choices["hf_qwen"]["hf_news_base_model"] = qwen_path

    direct_choices = config.get("model_options", {}).get("direct_chukchi", {}).get("choices", {})
    if "hf_lora_qwen" in direct_choices:
        direct_choices["hf_lora_qwen"]["hf_base_model"] = qwen_path

    tts_choices = config.get("model_options", {}).get("tts", {}).get("choices", {})
    if "mms_baseline" in tts_choices:
        tts_choices["mms_baseline"]["model_path"] = mms_path

    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("[package] rewrote configs/server.yaml to local HF base paths", flush=True)


def write_readme(output: Path, total_bytes: int, file_count: int) -> None:
    """Write readme for this pipeline stage."""
    readme = f"""# Runtime-артефакты Chukchi News Voice

Этот бандл содержит тяжелые локальные артефакты, необходимые для запуска `make serve`.

Внутри:
- финальные MT-модели для `RU -> CKT` и `CKT -> RU`;
- LoRA-адаптер для прямой генерации чукотского текста;
- inference-директории TTS;
- базовые веса HF Qwen для русской новости и LoRA;
- базовые веса HF MMS TTS;
- JSONL-данные translation memory;
- runtime-конфиги.

Внутри нет:
- Python virtualenv;
- training checkpoint-ов и optimizer state;
- сырого аудио;
- ASR-сегментов и промежуточных файлов.

Статистика:
- файлов: {file_count}
- байт: {total_bytes}

Загрузка в Hugging Face:

```bash
python3 -m pip install -U huggingface_hub
huggingface-cli login
huggingface-cli repo create chukchi-news-runtime --type dataset --private
huggingface-cli upload YOUR_USER/chukchi-news-runtime artifacts/runtime_bundle . --repo-type dataset
```

Восстановление в свежем checkout:

```bash
python3 -m pip install -U huggingface_hub
huggingface-cli download YOUR_USER/chukchi-news-runtime --repo-type dataset --local-dir .
```

После этого:

```bash
make serve
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def create_manifest(output: Path) -> dict:
    """Create manifest for this pipeline stage."""
    files = [path for path in sorted(output.rglob("*")) if path.is_file()]
    entries = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(output)
        print(f"[manifest] {index}/{len(files)} {relative}", flush=True)
        entries.append(
            {
                "path": str(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "name": "chukchi-news-runtime",
        "root": str(output),
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def verify(output: Path) -> None:
    """Verify for this pipeline stage."""
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = []
    invalid = []
    for index, entry in enumerate(manifest["files"], start=1):
        path = output / entry["path"]
        print(f"[verify] {index}/{manifest['file_count']} {entry['path']}", flush=True)
        if not path.exists():
            missing.append(entry["path"])
            continue
        if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            invalid.append(entry["path"])
    result = {"missing": missing, "invalid": invalid}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing or invalid:
        raise SystemExit(1)


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.verify_only:
        verify(output)
        return
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    missing = []
    copied = 0
    for relative_name in RUNTIME_PATHS:
        source = ROOT / relative_name
        if not source.exists():
            missing.append(relative_name)
            continue
        for source_file in iter_files(source):
            target = output / source_file.relative_to(ROOT)
            link_or_copy(source_file, target, args.mode)
            copied += 1
            print(f"[package] {source_file.relative_to(ROOT)}", flush=True)
    if missing:
        print(json.dumps({"missing": missing}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    copied += package_hf_base_models(output, args.mode)
    rewrite_packaged_server_config(output)

    manifest = create_manifest(output)
    write_readme(output, manifest["total_bytes"], copied)
    manifest = create_manifest(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "files": manifest["file_count"],
                "size_bytes": manifest["total_bytes"],
                "size_gib": round(manifest["total_bytes"] / 1024**3, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
