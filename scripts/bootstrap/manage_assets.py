#!/usr/bin/env python3
"""Create or verify checksums for local assets that are not stored in Git."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "assets" / "audio_manifest.json"
CHUNK_SIZE = 8 * 1024 * 1024
SUPPORTED_AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(source: Path, output: Path) -> None:
    files = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    entries = []
    for index, path in enumerate(files, start=1):
        relative_path = path.relative_to(ROOT)
        print(f"[{index}/{len(files)}] hashing {relative_path}")
        entries.append(
            {
                "path": str(relative_path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "source": str(source.relative_to(ROOT)),
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {output}")


def verify_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = []
    invalid = []

    for index, entry in enumerate(manifest["files"], start=1):
        path = ROOT / entry["path"]
        print(f"[{index}/{manifest['file_count']}] verifying {entry['path']}")
        if not path.exists():
            missing.append(entry["path"])
            continue
        if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            invalid.append(entry["path"])

    result = {"missing": missing, "invalid": invalid}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing or invalid:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "verify"])
    parser.add_argument("--source", type=Path, default=ROOT / "audio")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if args.action == "create":
        create_manifest(source, manifest)
    else:
        verify_manifest(manifest)


if __name__ == "__main__":
    main()
