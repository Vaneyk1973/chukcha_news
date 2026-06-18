#!/usr/bin/env python3
"""Create speech segments and optional MMS pseudo-labels from long-form audio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")
SEGMENT_FIELDS = [
    "segment_id",
    "audio_path",
    "source_path",
    "start_sec",
    "end_sec",
    "duration_sec",
]
ASR_FIELDS = SEGMENT_FIELDS + ["transcript", "confidence", "label_source"]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable is missing: {name}")


def probe_duration(path: Path) -> float | None:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def parse_silences(stderr: str) -> list[tuple[float, float]]:
    silences: list[tuple[float, float]] = []
    current_start: float | None = None
    for line in stderr.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            current_start = float(start_match.group(1))
        end_match = SILENCE_END_RE.search(line)
        if end_match:
            end = float(end_match.group(1))
            silences.append((0.0 if current_start is None else current_start, end))
            current_start = None
    return silences


def speech_regions(
    duration: float,
    silences: list[tuple[float, float]],
    min_segment: float,
    max_segment: float,
    padding: float,
) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in silences:
        if silence_start > cursor:
            regions.append((cursor, min(silence_start, duration)))
        cursor = max(cursor, silence_end)
    if cursor < duration:
        regions.append((cursor, duration))

    return bound_regions(regions, duration, min_segment, max_segment, padding)


def bound_regions(
    regions: list[tuple[float, float]],
    duration: float,
    min_segment: float,
    max_segment: float,
    padding: float,
) -> list[tuple[float, float]]:
    bounded: list[tuple[float, float]] = []
    for start, end in regions:
        start = max(0.0, start - padding)
        end = min(duration, end + padding)
        while end - start > max_segment:
            next_end = start + max_segment
            if end - next_end < min_segment:
                next_end = end - min_segment
            bounded.append((start, next_end))
            start = next_end
        if end - start >= min_segment:
            bounded.append((start, end))
    return bounded


class SpeechDetector:
    def __init__(self, config: dict, sample_rate: int, method: str | None = None) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self.method = method or config["vad_method"]
        if self.method == "silero":
            try:
                from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
            except ImportError as error:
                raise RuntimeError(
                    "Silero VAD is required for production segmentation. "
                    "Install it with: python3 -m pip install -e '.[asr]'"
                ) from error
            self.get_speech_timestamps = get_speech_timestamps
            self.read_audio = read_audio
            self.model = load_silero_vad()
        elif self.method != "silence":
            raise ValueError(f"Unsupported VAD method: {self.method}")

    def detect(self, path: Path, duration: float) -> list[tuple[float, float]]:
        if self.method == "silero":
            audio = self.read_audio(str(path), sampling_rate=self.sample_rate)
            timestamps = self.get_speech_timestamps(
                audio,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=float(self.config["vad_threshold"]),
                min_speech_duration_ms=int(self.config["min_speech_ms"]),
                min_silence_duration_ms=int(self.config["min_silence_ms"]),
                return_seconds=True,
            )
            regions = [(float(item["start"]), float(item["end"])) for item in timestamps]
            return bound_regions(
                regions,
                duration,
                float(self.config["min_segment_sec"]),
                float(self.config["max_segment_sec"]),
                float(self.config["boundary_padding_sec"]),
            )

        result = run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                str(path),
                "-af",
                (
                    f"silencedetect=noise={self.config['silence_noise_db']}dB:"
                    f"d={self.config['min_silence_sec']}"
                ),
                "-f",
                "null",
                "-",
            ]
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg silence detection failed for {path}:\n{result.stderr[-1000:]}")
        return speech_regions(
            duration,
            parse_silences(result.stderr),
            float(self.config["min_segment_sec"]),
            float(self.config["max_segment_sec"]),
            float(self.config["boundary_padding_sec"]),
        )


def segment_id(source_relative: str, start: float, end: float) -> str:
    digest = hashlib.sha256(f"{source_relative}\0{start:.3f}\0{end:.3f}".encode()).hexdigest()
    return digest[:20]


def write_segment(source: Path, target: Path, start: float, end: float, sample_rate: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.wav")
    result = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{end - start:.3f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ]
    )
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg segmentation failed for {source}:\n{result.stderr[-1000:]}")
    temporary.replace(target)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    state = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                row = json.loads(line)
                state[row["segment_id"]] = row
    return state


def write_state(path: Path, state: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for segment_key in sorted(state):
            output.write(json.dumps(state[segment_key], ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_wav(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM WAV: {path}")
        frames = wav_file.readframes(wav_file.getnframes())
    import array

    samples = array.array("h")
    samples.frombytes(frames)
    return [sample / 32768.0 for sample in samples]


class MMSTranscriber:
    def __init__(self, model_name: str, language_code: str, device: str) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Wav2Vec2ForCTC
        except ImportError as error:
            raise RuntimeError("Install ASR dependencies with: python3 -m pip install -e '.[asr]'") from error

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but the installed PyTorch build has no CUDA support.")
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor.tokenizer.set_target_lang(language_code)
        self.model.load_adapter(language_code)
        self.model.to(device).eval()

    def transcribe(self, paths: list[Path], sample_rate: int) -> list[tuple[str, float]]:
        audio = [read_wav(path) for path in paths]
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
        ).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            predicted_ids = self.torch.argmax(logits, dim=-1)
            frame_confidence = logits.softmax(dim=-1).amax(dim=-1)
            input_lengths = self.torch.tensor([len(samples) for samples in audio], device=self.device)
            output_lengths = self.model._get_feat_extract_output_lengths(input_lengths).cpu().tolist()
            confidence = self.torch.stack(
                [scores[:length].mean() for scores, length in zip(frame_confidence, output_lengths)]
            )
        texts = self.processor.batch_decode(predicted_ids)
        return [(text.strip(), float(score)) for text, score in zip(texts, confidence.cpu())]


def prepare_segments(
    config: dict, limit: int | None, dry_run: bool, vad_method: str | None
) -> tuple[list[dict], dict]:
    data = config["data"]
    preprocessing = config["preprocessing"]
    raw_root = resolve_path(data["raw_audio_root"])
    segments_root = resolve_path(data["segments_root"])
    sample_rate = int(data["sample_rate"])
    sources = sorted(
        path for path in raw_root.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if limit is not None:
        sources = sources[:limit]
    detector = SpeechDetector(preprocessing, sample_rate, vad_method)

    rows: list[dict] = []
    broken_sources: list[str] = []
    created = skipped = 0
    for source_index, source in enumerate(sources, start=1):
        source_relative = str(source.relative_to(ROOT))
        duration = probe_duration(source)
        if duration is None:
            broken_sources.append(source_relative)
            continue
        regions = detector.detect(source, duration)
        print(f"[{source_index}/{len(sources)}] {source_relative}: {len(regions)} segments")
        for start, end in regions:
            key = segment_id(source_relative, start, end)
            target = segments_root / source.relative_to(raw_root).parent / f"{key}.wav"
            if not dry_run:
                if target.exists() and not preprocessing["overwrite"]:
                    skipped += 1
                else:
                    write_segment(source, target, start, end, sample_rate)
                    created += 1
            rows.append(
                {
                    "segment_id": key,
                    "audio_path": str(target.relative_to(ROOT)),
                    "source_path": source_relative,
                    "start_sec": f"{start:.3f}",
                    "end_sec": f"{end:.3f}",
                    "duration_sec": f"{end - start:.3f}",
                }
            )
    return rows, {
        "source_files": len(sources),
        "segments": len(rows),
        "segments_created": created,
        "segments_reused": skipped,
        "broken_sources": broken_sources,
        "vad_method": detector.method,
    }


def transcribe_segments(config: dict, rows: list[dict], limit: int | None) -> tuple[dict[str, dict], dict]:
    data = config["data"]
    transcription = config["transcription"]
    state_path = resolve_path(data["transcription_state"])
    state = load_state(state_path)
    pending = [row for row in rows if row["segment_id"] not in state]
    if limit is not None:
        pending = pending[:limit]
    transcriber = MMSTranscriber(
        config["model"]["base_model"],
        config["model"]["language_code"],
        transcription["device"],
    )
    batch_size = int(transcription["batch_size"])
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        paths = [resolve_path(row["audio_path"]) for row in batch]
        results = transcriber.transcribe(paths, int(data["sample_rate"]))
        for row, (text, confidence) in zip(batch, results):
            state[row["segment_id"]] = {
                "segment_id": row["segment_id"],
                "transcript": text,
                "confidence": round(confidence, 6),
                "label_source": config["model"]["base_model"],
            }
        write_state(state_path, state)
        print(f"transcribed {min(start + len(batch), len(pending))}/{len(pending)}")
    return state, {"already_transcribed": len(rows) - len(pending), "transcribed_now": len(pending)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/asr.yaml")
    parser.add_argument("--stage", choices=["segment", "transcribe", "all"], default="all")
    parser.add_argument("--limit", type=int, help="Limit source files during segmentation.")
    parser.add_argument("--transcribe-limit", type=int, help="Limit new segments transcribed.")
    parser.add_argument("--vad", choices=["silero", "silence"], help="Override configured VAD.")
    parser.add_argument("--dry-run", action="store_true", help="Detect boundaries without writing WAVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_binary("ffmpeg")
    require_binary("ffprobe")
    config = load_yaml(args.config)
    data = config["data"]
    segment_manifest = resolve_path(data["segment_manifest"])

    if args.stage in {"segment", "all"}:
        rows, report = prepare_segments(config, args.limit, args.dry_run, args.vad)
        if not args.dry_run:
            write_csv(segment_manifest, rows, SEGMENT_FIELDS)
    else:
        with segment_manifest.open("r", encoding="utf-8", newline="") as input_file:
            rows = list(csv.DictReader(input_file))
        report = {"source_files": None, "segments": len(rows)}

    state = load_state(resolve_path(data["transcription_state"]))
    if args.stage in {"transcribe", "all"} and not args.dry_run:
        state, transcription_report = transcribe_segments(config, rows, args.transcribe_limit)
        report.update(transcription_report)

    min_confidence = float(config["transcription"]["min_confidence"])
    train_rows = []
    for row in rows:
        label = state.get(row["segment_id"])
        if label and label["transcript"] and float(label["confidence"]) >= min_confidence:
            train_rows.append({**row, **label})
    if not args.dry_run:
        write_csv(resolve_path(data["audio_manifest"]), train_rows, ASR_FIELDS)

    report.update(
        {
            "stage": args.stage,
            "dry_run": args.dry_run,
            "train_ready_segments": len(train_rows),
            "segment_manifest": str(segment_manifest),
            "audio_manifest": str(resolve_path(data["audio_manifest"])),
        }
    )
    report_path = ROOT / "reports" / "asr_preprocessing.json"
    if not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
