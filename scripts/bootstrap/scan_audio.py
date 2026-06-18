#!/usr/bin/env python3
"""Bootstrap and packaging helper for reproducible local project setup."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIO_ROOT = ROOT / "audio"
MANIFEST_PATH = ROOT / "data" / "manifests" / "audio_inventory.csv"
REPORT_PATH = ROOT / "reports" / "audio_inventory.json"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def probe_duration(audio_path: Path) -> float | None:
    """Probe duration for this pipeline stage."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def main() -> None:
    """Run the command-line workflow for this module."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_duration = 0.0
    broken = 0

    for audio_path in sorted(AUDIO_ROOT.rglob("*")):
        if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        duration = probe_duration(audio_path)
        if duration is None:
            broken += 1

        rows.append(
            {
                "path": str(audio_path.relative_to(ROOT)),
                "suffix": audio_path.suffix.lower(),
                "duration_sec": "" if duration is None else f"{duration:.2f}",
                "size_bytes": audio_path.stat().st_size,
            }
        )

        if duration is not None:
            total_duration += duration

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["path", "suffix", "duration_sec", "size_bytes"]
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "audio_root": str(AUDIO_ROOT.relative_to(ROOT)),
        "file_count": len(rows),
        "total_hours": round(total_duration / 3600, 2),
        "broken_or_unreadable_files": broken,
    }
    REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
