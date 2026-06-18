#!/usr/bin/env python3
"""Evaluate TTS with ASR round-trip CER and listening samples."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402
from scripts.data.prepare_asr import MMSTranscriber  # noqa: E402
from scripts.data.select_tts_pseudolabels import character_error_rate  # noqa: E402


def configure_transformers_output() -> None:
    try:
        from transformers.utils import logging
    except ImportError:
        return
    logging.set_verbosity_error()
    logging.disable_progress_bar()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tts.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", choices=["baseline", "finetuned", "both"], default="both")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Legacy shortcut. Used for both TTS and ASR unless the specific device args are set.",
    )
    parser.add_argument("--tts-device", choices=["auto", "cpu", "cuda"], help="Device for synthesis.")
    parser.add_argument(
        "--asr-device",
        choices=["auto", "cpu", "cuda"],
        help="Device for ASR round-trip scoring. Defaults to CPU to avoid CUDA OOM after synthesis.",
    )
    parser.add_argument("--asr-batch-size", type=int, help="Number of synthetic files per ASR batch.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


class TTSSynthesizer:
    def __init__(self, model_path: str | Path, device: str) -> None:
        import torch
        from transformers import VitsModel, VitsTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = device
        self.tokenizer = VitsTokenizer.from_pretrained(model_path)
        self.model = VitsModel.from_pretrained(model_path).to(device).eval()
        self.sample_rate = int(self.model.config.sampling_rate)

    def synthesize(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs)
        return output.waveform[0].detach().cpu().numpy()

    def close(self) -> None:
        del self.model
        del self.tokenizer
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_specs(config: dict, requested: str) -> list[tuple[str, str | Path]]:
    specs: list[tuple[str, str | Path]] = []
    if requested in {"baseline", "both"}:
        specs.append(("baseline", config["model"]["base_model"]))
    finetuned = resolve_path(config["training"]["output_dir"])
    if requested in {"finetuned", "both"} and (finetuned / "config.json").exists():
        specs.append(("finetuned", finetuned))
    return specs


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"samples": 0, "mean_cer": None}
    cers = [float(row["cer"]) for row in rows]
    durations = [float(row["synthetic_duration_sec"]) for row in rows]
    return {
        "samples": len(rows),
        "mean_cer": round(sum(cers) / len(cers), 4),
        "median_cer": round(sorted(cers)[len(cers) // 2], 4),
        "mean_synthetic_duration_sec": round(sum(durations) / len(durations), 3),
    }


def transcribe_in_batches(
    model_name: str,
    paths: list[Path],
    sample_rate: int,
    config: dict,
    device: str,
    batch_size: int,
) -> list[tuple[str, float]]:
    transcriber = MMSTranscriber(
        config["pseudo_labeling"]["transcriber"],
        config["model"]["language_code"],
        device,
    )
    results: list[tuple[str, float]] = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        results.extend(transcriber.transcribe(batch, sample_rate))
        print(
            f"[eval-tts:{model_name}] ASR {min(start + len(batch), len(paths))}/{len(paths)}",
            flush=True,
        )
    del transcriber
    clear_cuda_cache()
    return results


def main() -> None:
    configure_transformers_output()
    args = parse_args()
    config = load_yaml(args.config)
    eval_config = config["evaluation"]
    tts_device = args.tts_device or args.device or eval_config.get("tts_device", "auto")
    asr_device = args.asr_device or args.device or eval_config.get("asr_device", "cpu")
    asr_batch_size = args.asr_batch_size or int(eval_config.get("asr_batch_size", 2))
    if asr_batch_size < 1:
        raise ValueError("--asr-batch-size must be >= 1")

    rows = read_jsonl(resolve_path(config["data"]["manifest"]))
    sample_count = args.limit or int(eval_config["samples"])
    sample = random.Random(int(eval_config["seed"])).sample(rows, min(sample_count, len(rows)))

    specs = model_specs(config, args.models)
    if not specs:
        raise RuntimeError("No TTS models selected. Train a model or evaluate baseline.")

    output_dir = resolve_path(eval_config["output_dir"])

    eval_rows = []
    for model_name, model_path in specs:
        synthesizer = TTSSynthesizer(model_path, tts_device)
        generated_paths = []
        generated_meta = []
        for index, row in enumerate(sample, start=1):
            audio = synthesizer.synthesize(row["text"])
            target = output_dir / model_name / f"{row['segment_id']}.wav"
            write_wav(target, audio, synthesizer.sample_rate)
            generated_paths.append(target)
            generated_meta.append((row, len(audio) / synthesizer.sample_rate))
            print(f"[eval-tts:{model_name}] synth {index}/{len(sample)}", flush=True)

        synthesizer.close()
        del synthesizer
        clear_cuda_cache()

        transcriptions = transcribe_in_batches(
            model_name,
            generated_paths,
            int(config["data"]["sample_rate"]),
            config,
            asr_device,
            asr_batch_size,
        )
        for (row, duration), path, (asr_text, confidence) in zip(
            generated_meta, generated_paths, transcriptions
        ):
            cer = character_error_rate(row["text"], asr_text)
            eval_rows.append(
                {
                    "model": model_name,
                    "segment_id": row["segment_id"],
                    "text": row["text"],
                    "synthetic_audio_path": str(path.relative_to(ROOT)),
                    "original_audio_path": row["audio_path"],
                    "original_duration_sec": f"{float(row['duration_sec']):.3f}",
                    "synthetic_duration_sec": f"{duration:.3f}",
                    "roundtrip_asr": asr_text,
                    "roundtrip_asr_confidence": f"{confidence:.6f}",
                    "cer": f"{cer:.6f}",
                }
            )

    samples_csv = resolve_path(eval_config["samples_csv"])
    samples_csv.parent.mkdir(parents=True, exist_ok=True)
    with samples_csv.open("w", encoding="utf-8", newline="") as output:
        fieldnames = list(eval_rows[0].keys()) if eval_rows else []
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)

    by_model = {
        model_name: summarize([row for row in eval_rows if row["model"] == model_name])
        for model_name, _ in specs
    }
    report = {
        "manifest": str(resolve_path(config["data"]["manifest"])),
        "output_dir": str(output_dir),
        "samples_csv": str(samples_csv),
        "tts_device": tts_device,
        "asr_device": asr_device,
        "asr_batch_size": asr_batch_size,
        "models": by_model,
    }
    report_path = resolve_path(eval_config["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
