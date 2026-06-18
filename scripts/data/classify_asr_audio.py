#!/usr/bin/env python3
"""Classify ASR segments into speech, music, or uncertain buckets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import resolve_path  # noqa: E402


DEFAULT_INPUT = ROOT / "data" / "interim" / "asr_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "asr_audio_classes"
DEFAULT_REPORT = ROOT / "reports" / "asr_audio_classification.json"
DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000

SPEECH_LABELS = {
    "speech",
    "conversation",
    "narration, monologue",
    "babbling",
}
MUSIC_LABELS = {
    "music",
    "musical instrument",
    "singing",
    "song",
    "background music",
    "theme music",
    "jingle (music)",
    "radio",
}
NOISE_LABELS = {
    "silence",
    "inside, small room",
    "outside, urban or manmade",
    "noise",
    "static",
    "white noise",
}
OUTPUT_FIELDS = [
    "audio_class",
    "audio_class_confidence",
    "audio_class_margin",
    "audio_top_labels",
    "audio_class_model",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--speech-threshold", type=float, default=0.35)
    parser.add_argument("--music-threshold", type=float, default=0.25)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def read_rows(path: Path, limit: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return rows[:limit] if limit else rows


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")


def read_audio(path: Path, sample_rate: int = SAMPLE_RATE):
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-1000:])

    import array

    samples = array.array("h")
    samples.frombytes(result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    import numpy as np

    return np.asarray(samples, dtype=np.float32) / 32768.0


def normalize_label(label: str) -> str:
    return label.strip().casefold()


def bucket_scores(predictions: list[dict]) -> dict[str, float]:
    scores = {"speech": 0.0, "music": 0.0, "noise": 0.0}
    for prediction in predictions:
        label = normalize_label(prediction["label"])
        score = float(prediction["score"])
        if label in SPEECH_LABELS:
            scores["speech"] = max(scores["speech"], score)
        if label in MUSIC_LABELS or "music" in label or "sing" in label:
            scores["music"] = max(scores["music"], score)
        if label in NOISE_LABELS or "noise" in label or "silence" in label:
            scores["noise"] = max(scores["noise"], score)
    return scores


def decide_class(
    predictions: list[dict],
    speech_threshold: float,
    music_threshold: float,
    margin_threshold: float,
) -> tuple[str, float, float]:
    scores = bucket_scores(predictions)
    speech_score = scores["speech"]
    music_score = max(scores["music"], scores["noise"])
    margin = abs(speech_score - music_score)

    if music_score >= music_threshold and music_score >= speech_score - margin_threshold:
        return "music", music_score, music_score - speech_score
    if speech_score >= speech_threshold and speech_score >= music_score + margin_threshold:
        return "speech", speech_score, speech_score - music_score
    return "uncertain", max(speech_score, music_score), margin


def load_model(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline
    except ImportError as error:
        raise RuntimeError("Install ASR dependencies with: make setup-asr") from error

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline_device = 0 if device == "cuda" else -1
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    return pipeline(
        "audio-classification",
        model=model,
        feature_extractor=extractor,
        device=pipeline_device,
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def classify_rows(args: argparse.Namespace) -> tuple[list[dict], Counter]:
    classifier = load_model(args.model, args.device)
    rows = read_rows(args.input, args.limit)
    classified = []
    failures = Counter()

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        audio_batch = []
        valid_rows = []
        for row in batch:
            try:
                audio_batch.append(read_audio(resolve_path(row["audio_path"])))
                valid_rows.append(row)
            except Exception as error:
                enriched = {
                    **row,
                    "audio_class": "uncertain",
                    "audio_class_confidence": "0.000000",
                    "audio_class_margin": "0.000000",
                    "audio_top_labels": json.dumps([{"error": str(error)}], ensure_ascii=False),
                    "audio_class_model": args.model,
                }
                classified.append(enriched)
                failures["decode_failed"] += 1

        if not audio_batch:
            continue
        predictions_batch = classifier(audio_batch, top_k=args.top_k)
        for row, predictions in zip(valid_rows, predictions_batch):
            decision, confidence, margin = decide_class(
                predictions,
                args.speech_threshold,
                args.music_threshold,
                args.margin_threshold,
            )
            classified.append(
                {
                    **row,
                    "audio_class": decision,
                    "audio_class_confidence": f"{confidence:.6f}",
                    "audio_class_margin": f"{margin:.6f}",
                    "audio_top_labels": json.dumps(predictions, ensure_ascii=False),
                    "audio_class_model": args.model,
                }
            )
        print(f"classified {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    return classified, failures


def main() -> None:
    args = parse_args()
    require_ffmpeg()
    classified, failures = classify_rows(args)
    fields = list(csv.DictReader(args.input.open("r", encoding="utf-8", newline="")).fieldnames or [])
    fields.extend(field for field in OUTPUT_FIELDS if field not in fields)

    by_class: dict[str, list[dict]] = {"speech": [], "music": [], "uncertain": []}
    for row in classified:
        by_class[row["audio_class"]].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "all.csv", classified, fields)
    for class_name, rows in by_class.items():
        write_csv(args.output_dir / f"{class_name}.csv", rows, fields)

    report = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "rows": len(classified),
        "counts": {name: len(rows) for name, rows in by_class.items()},
        "failures": dict(failures),
        "thresholds": {
            "speech": args.speech_threshold,
            "music": args.music_threshold,
            "margin": args.margin_threshold,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
