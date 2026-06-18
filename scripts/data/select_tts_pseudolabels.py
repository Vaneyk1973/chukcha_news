#!/usr/bin/env python3
"""Select high-confidence ASR pseudo-labels for a bootstrap TTS corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "tts.yaml"


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate for this pipeline stage."""
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    if not reference:
        return 1.0 if hypothesis else 0.0

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1] / len(reference)


def rejection_reasons(row: dict, config: dict) -> list[str]:
    """Rejection reasons for this pipeline stage."""
    reasons = []
    duration = float(row.get("duration_sec", 0.0))
    text = str(row.get("text", "")).strip()
    confidence = float(row.get("confidence", 0.0))

    if not config["min_duration_sec"] <= duration <= config["max_duration_sec"]:
        reasons.append("duration")
    if confidence < config["min_confidence"]:
        reasons.append("confidence")
    if not text:
        reasons.append("empty_text")

    chars_per_sec = len(text) / duration if duration > 0 else 0.0
    if not config["min_chars_per_sec"] <= chars_per_sec <= config["max_chars_per_sec"]:
        reasons.append("chars_per_sec")

    if config["require_single_speaker"] and row.get("single_speaker") is not True:
        reasons.append("multiple_or_unknown_speakers")
    if config["require_no_music"] and row.get("has_music") is not False:
        reasons.append("music_or_unknown_audio")

    second_text = str(row.get("second_text", "")).strip()
    if second_text:
        disagreement = character_error_rate(text, second_text)
        row["asr_disagreement_cer"] = round(disagreement, 4)
        if disagreement > config["max_asr_disagreement_cer"]:
            reasons.append("asr_disagreement")

    return reasons


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--accepted", type=Path)
    parser.add_argument("--rejected", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pseudo_config = config["pseudo_labeling"]
    data_config = config["data"]

    input_path = args.input or ROOT / data_config["pseudo_label_manifest"]
    accepted_path = args.accepted or ROOT / data_config["manifest"]
    rejected_path = args.rejected or ROOT / data_config["rejected_manifest"]
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    accepted = []
    rejected = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            row = json.loads(line)
            reasons = rejection_reasons(row, pseudo_config)
            if reasons:
                row["rejection_reasons"] = reasons
                rejected.append(row)
            else:
                accepted.append(row)

    with accepted_path.open("w", encoding="utf-8") as output_file:
        for row in accepted:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with rejected_path.open("w", encoding="utf-8") as output_file:
        for row in rejected:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
