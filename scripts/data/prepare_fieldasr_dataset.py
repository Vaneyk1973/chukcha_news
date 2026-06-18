#!/usr/bin/env python3
"""Recover and validate exact FieldASR audio/transcript pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
MANIFEST_FIELDS = [
    "id",
    "audio_path",
    "text",
    "duration_sec",
    "source_audio_path",
    "transcript_path",
]


def normalize_text(text: str) -> str:
    """Normalize text for this pipeline stage."""
    text = text.lstrip("\ufeff")
    return re.sub(r"\s+", " ", text).strip()


def safe_member_path(root: Path, name: str) -> Path:
    """Safe member path for this pipeline stage."""
    target = (root / name).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"Archive member escapes destination: {name}")
    return target


def extract_archive(path: Path, destination: Path) -> None:
    """Extract archive for this pipeline stage."""
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = safe_member_path(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                target = safe_member_path(destination, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is not None:
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        return
    raise ValueError(f"Unsupported archive: {path}")


def download(url: str, destination: Path) -> Path:
    """Download for this pipeline stage."""
    destination.mkdir(parents=True, exist_ok=True)
    name = Path(urllib.parse.urlparse(url).path).name or hashlib.sha256(url.encode()).hexdigest()
    target = destination / name
    if target.exists() and target.stat().st_size > 0:
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "chukcha-news/0.1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(target)
    return target


def index_files(roots: list[Path], extensions: set[str] | None = None) -> dict[str, list[Path]]:
    """Index files for this pipeline stage."""
    index: dict[str, list[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or extensions and path.suffix.lower() not in extensions:
                continue
            for key in {path.name.casefold(), path.stem.casefold()}:
                index.setdefault(key, []).append(path)
    return index


def probe_duration(path: Path) -> float | None:
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
            str(path),
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


def duration_matches(actual: float, expected: float, config: dict) -> bool:
    """Duration matches for this pipeline stage."""
    tolerance = max(
        float(config["duration_tolerance_sec"]),
        expected * float(config["duration_tolerance_ratio"]),
    )
    return abs(actual - expected) <= tolerance


def normalize_audio(source: Path, target: Path, sample_rate: int) -> None:
    """Normalize audio for this pipeline stage."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.wav")
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(result.stderr[-1000:])
    temporary.replace(target)


def read_legacy_metadata(path: Path) -> list[dict]:
    """Read legacy metadata for this pipeline stage."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def read_metadata_archive(path: Path, member: str) -> list[dict]:
    """Read metadata archive for this pipeline stage."""
    with zipfile.ZipFile(path) as archive, archive.open(member) as raw_file:
        lines = (line.decode("utf-8-sig") for line in raw_file)
        rows = []
        for row in csv.DictReader(lines, delimiter="\t"):
            rows.append(
                {
                    "audiofile": row["path"],
                    "text": row["sentence"],
                    "duration": row["duration"],
                }
            )
        return rows


def read_split_metadata(source_config: dict, source_split: str) -> tuple[list[dict], str]:
    """Read split metadata for this pipeline stage."""
    archive_path = resolve_path(source_config["metadata_archive"])
    archive_members = {
        "train": "train_new.tsv",
        "dev": "dev_new.tsv",
        "test": "test_new.tsv",
    }
    if archive_path.exists():
        member = archive_members[source_split]
        return read_metadata_archive(archive_path, member), f"{archive_path}::{member}"
    metadata_path = resolve_path(source_config["metadata_root"]) / f"{source_split}.csv"
    return read_legacy_metadata(metadata_path), str(metadata_path)


def write_manifest(path: Path, rows: list[dict]) -> None:
    """Write manifest for this pipeline stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def relative_or_absolute(path: Path) -> str:
    """Relative or absolute for this pipeline stage."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fieldasr.yaml")
    parser.add_argument("--archive-url", action="append", default=[])
    parser.add_argument("--archive", action="append", type=Path, default=[])
    parser.add_argument("--audio-root", action="append", type=Path, default=[])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    config = load_yaml(args.config)
    source_config = config["source"]
    output_config = config["output"]
    validation = config["validation"]
    download_root = resolve_path(source_config["download_root"])

    urls = list(source_config.get("exact_archive_urls", []))
    urls.extend(args.archive_url)
    if os.environ.get("FIELDASR_ARCHIVE_URL"):
        urls.append(os.environ["FIELDASR_ARCHIVE_URL"])
    download_results = []
    for archive in args.archive:
        try:
            archive = archive.resolve()
            extract_root = download_root / "extracted" / archive.stem
            extract_archive(archive, extract_root)
            download_results.append(
                {"archive": str(archive), "status": "extracted", "path": str(extract_root)}
            )
        except Exception as error:
            download_results.append(
                {"archive": str(archive), "status": "failed", "error": str(error)}
            )
    if not args.skip_download:
        for url in dict.fromkeys(urls):
            try:
                archive = download(url, download_root / "archives")
                extract_root = download_root / "extracted" / archive.stem
                extract_archive(archive, extract_root)
                download_results.append({"url": url, "status": "downloaded", "path": str(archive)})
            except Exception as error:
                download_results.append({"url": url, "status": "failed", "error": str(error)})

    audio_roots = [resolve_path(path) for path in source_config["local_audio_roots"]]
    audio_roots.extend(path.resolve() for path in args.audio_root)
    audio_index = index_files(audio_roots, AUDIO_EXTENSIONS)
    transcript_root = resolve_path(source_config["transcript_root"])
    transcript_index = index_files([transcript_root])
    output_audio_root = resolve_path(output_config["audio_root"])
    manifest_root = resolve_path(output_config["manifest_root"])

    report = {
        "downloads": download_results,
        "audio_roots": [str(path) for path in audio_roots],
        "splits": {},
        "rejections": Counter(),
        "missing_audio": [],
        "missing_transcripts": [],
        "ambiguous_audio": [],
    }
    seen_audio: dict[str, str] = {}
    split_names = {"train": "train", "dev": "validation", "test": "test"}
    for source_split, output_split in split_names.items():
        accepted = []
        metadata_rows, metadata_source = read_split_metadata(source_config, source_split)
        for row in metadata_rows:
            audio_name = row["audiofile"].strip()
            audio_matches = audio_index.get(audio_name.casefold(), [])
            if not audio_matches:
                audio_matches = audio_index.get(Path(audio_name).stem.casefold(), [])
            if not audio_matches:
                report["rejections"]["missing_audio"] += 1
                report["missing_audio"].append(audio_name)
                continue
            if len(audio_matches) > 1:
                report["rejections"]["ambiguous_audio"] += 1
                report["ambiguous_audio"].append(
                    {"name": audio_name, "matches": [str(path) for path in audio_matches]}
                )
                continue
            source_audio = audio_matches[0]
            transcript_name = row.get("transcript", "").strip()
            transcript_matches = transcript_index.get(transcript_name.casefold(), [])
            transcript = transcript_matches[0] if transcript_matches else None
            if row.get("text"):
                text = normalize_text(row["text"])
            elif transcript is not None:
                text = normalize_text(transcript.read_text(encoding="utf-8-sig"))
            else:
                report["rejections"]["missing_transcript"] += 1
                report["missing_transcripts"].append(transcript_name)
                continue
            if not text:
                report["rejections"]["empty_transcript"] += 1
                continue
            actual_duration = probe_duration(source_audio)
            try:
                expected_duration = float(row["duration"])
            except ValueError:
                expected_duration = 0.0
            if actual_duration is None:
                report["rejections"]["unreadable_audio"] += 1
                continue
            if not duration_matches(actual_duration, expected_duration, validation):
                report["rejections"]["duration_mismatch"] += 1
                continue
            if (
                not float(validation["min_duration_sec"])
                <= actual_duration
                <= float(validation["max_duration_sec"])
            ):
                report["rejections"]["duration_out_of_range"] += 1
                continue
            audio_key = source_audio.resolve().as_posix()
            if audio_key in seen_audio:
                report["rejections"]["cross_split_duplicate"] += 1
                continue
            seen_audio[audio_key] = output_split
            item_id = hashlib.sha256(
                f"{output_split}\0{audio_name}\0{text}".encode("utf-8")
            ).hexdigest()[:20]
            target = output_audio_root / output_split / f"{item_id}.wav"
            if not target.exists():
                try:
                    normalize_audio(source_audio, target, int(validation["sample_rate"]))
                except RuntimeError:
                    report["rejections"]["normalization_failed"] += 1
                    continue
            normalized_duration = probe_duration(target)
            accepted.append(
                {
                    "id": item_id,
                    "audio_path": relative_or_absolute(target),
                    "text": text,
                    "duration_sec": f"{normalized_duration:.3f}",
                    "source_audio_path": relative_or_absolute(source_audio),
                    "transcript_path": ""
                    if transcript is None
                    else relative_or_absolute(transcript),
                }
            )
        write_manifest(manifest_root / f"{output_split}.csv", accepted)
        report["splits"][output_split] = {
            "metadata_rows": len(metadata_rows),
            "metadata_source": metadata_source,
            "accepted": len(accepted),
            "recovery_ratio": round(len(accepted) / len(metadata_rows), 4),
            "duration_hours": round(sum(float(row["duration_sec"]) for row in accepted) / 3600, 3),
        }

    report["rejections"] = dict(report["rejections"])
    report["training_ready"] = all(
        report["splits"][split]["recovery_ratio"] >= float(validation["min_recovery_ratio"])
        for split in split_names.values()
    )
    report_path = resolve_path(output_config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "downloads": download_results,
        "splits": report["splits"],
        "rejections": report["rejections"],
        "training_ready": report["training_ready"],
        "report_path": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    require_all = bool(validation["require_all_splits"]) and not args.allow_partial
    if require_all and not report["training_ready"]:
        raise SystemExit(
            "FieldASR data is not training-ready. Add an exact segmented audio archive or local "
            f"audio root, then rerun. See {report_path}."
        )


if __name__ == "__main__":
    main()
