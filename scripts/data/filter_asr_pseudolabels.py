#!/usr/bin/env python3
"""Filter ASR pseudo-labels for Russian speech and likely foreign songs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "interim" / "asr_manifest.csv"
DEFAULT_AUDIO_CLASSES = ROOT / "data" / "interim" / "asr_audio_classes" / "all.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "asr_text_filtered"
DEFAULT_REPORT = ROOT / "reports" / "asr_text_filtering.json"

CHUKCHI_SPECIFIC = set("ӃӄӇӈԒԓ")
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁёӃӄӇӈԒԓ']+")
RUSSIAN_STOPWORDS = {
    "а",
    "без",
    "будет",
    "бы",
    "был",
    "была",
    "были",
    "вам",
    "вас",
    "весь",
    "все",
    "вы",
    "где",
    "для",
    "до",
    "его",
    "если",
    "есть",
    "еще",
    "же",
    "за",
    "и",
    "из",
    "или",
    "как",
    "когда",
    "который",
    "мы",
    "на",
    "не",
    "но",
    "о",
    "от",
    "по",
    "при",
    "с",
    "сегодня",
    "так",
    "то",
    "у",
    "что",
    "это",
    "этот",
    "я",
}
RUSSIAN_RADIO_WORDS = {
    "внимание",
    "говорит",
    "говорите",
    "готовится",
    "господа",
    "дамы",
    "интервью",
    "новости",
    "передача",
    "программа",
    "программы",
    "радио",
    "расскажу",
    "русский",
    "слушайте",
    "эфир",
    "язык",
}
FOREIGN_SONG_LABELS = {"singing", "song", "music"}
RUSSIAN_FUZZY_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"нов[аэо]ч?т",
        r"нав[ао]ч?т",
        r"ради[оёе]",
        r"на\s*ради[оёе]",
        r"п[уо]рг[ао]",
        r"пр[ао]грам",
        r"г[оа]сп[оа]д",
        r"инт[еэ]р[еэ]сн",
        r"с[еэ]годн",
    )
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audio-classes", type=Path, default=DEFAULT_AUDIO_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-chukchi-specific-ratio", type=float, default=0.015)
    parser.add_argument("--russian-score-threshold", type=float, default=2.0)
    parser.add_argument("--hard-russian-score-threshold", type=float, default=4.0)
    parser.add_argument("--foreign-song-score-threshold", type=float, default=0.35)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def chukchi_specific_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char in CHUKCHI_SPECIFIC for char in letters) / len(letters)


def russian_score(text: str) -> float:
    words = tokens(text)
    if not words:
        return 0.0
    stopword_hits = sum(word in RUSSIAN_STOPWORDS for word in words)
    radio_hits = sum(word in RUSSIAN_RADIO_WORDS for word in words)
    latin_hits = sum(any("a" <= char <= "z" for char in word) for word in words)
    fuzzy_hits = sum(1 for pattern in RUSSIAN_FUZZY_PATTERNS if pattern.search(normalize_text(text)))
    return stopword_hits * 0.5 + radio_hits * 1.5 + latin_hits + fuzzy_hits


def load_audio_classes(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return {row["segment_id"]: row for row in csv.DictReader(input_file)}


def top_label_score(row: dict, wanted: set[str]) -> float:
    if not row or not row.get("audio_top_labels"):
        return 0.0
    try:
        labels = json.loads(row["audio_top_labels"])
    except json.JSONDecodeError:
        return 0.0
    score = 0.0
    for item in labels:
        label = str(item.get("label", "")).casefold()
        if any(want in label for want in wanted):
            score = max(score, float(item.get("score", 0.0)))
    return score


def rejection_reasons(
    row: dict,
    audio_row: dict | None,
    min_chukchi_specific_ratio: float,
    russian_score_threshold: float,
    hard_russian_score_threshold: float,
    foreign_song_score_threshold: float,
) -> list[str]:
    text = row.get("transcript", "")
    reasons = []
    chukchi_ratio = chukchi_specific_ratio(text)
    ru_score = russian_score(text)
    song_score = top_label_score(audio_row or {}, FOREIGN_SONG_LABELS)

    if (
        ru_score >= hard_russian_score_threshold
        or ru_score >= russian_score_threshold
        and chukchi_ratio < min_chukchi_specific_ratio
    ):
        reasons.append("russian_text")
    if song_score >= foreign_song_score_threshold and chukchi_ratio < min_chukchi_specific_ratio:
        reasons.append("likely_foreign_song")
    return reasons


def read_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return rows[:limit] if limit else rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input, args.limit)
    audio_classes = load_audio_classes(args.audio_classes)

    kept = []
    rejected = []
    reason_counts = Counter()
    for row in rows:
        audio_row = audio_classes.get(row["segment_id"])
        reasons = rejection_reasons(
            row,
            audio_row,
            args.min_chukchi_specific_ratio,
            args.russian_score_threshold,
            args.hard_russian_score_threshold,
            args.foreign_song_score_threshold,
        )
        enriched = {
            **row,
            "text_filter_reasons": ",".join(reasons),
            "russian_score": f"{russian_score(row.get('transcript', '')):.3f}",
            "chukchi_specific_ratio": f"{chukchi_specific_ratio(row.get('transcript', '')):.6f}",
            "foreign_song_score": f"{top_label_score(audio_row or {}, FOREIGN_SONG_LABELS):.6f}",
        }
        if audio_row:
            enriched["audio_class"] = audio_row.get("audio_class", "")
            enriched["audio_top_labels"] = audio_row.get("audio_top_labels", "")
        if reasons:
            rejected.append(enriched)
            reason_counts.update(reasons)
        else:
            kept.append(enriched)

    base_fields = list(csv.DictReader(args.input.open("r", encoding="utf-8", newline="")).fieldnames or [])
    extra_fields = [
        "text_filter_reasons",
        "russian_score",
        "chukchi_specific_ratio",
        "foreign_song_score",
        "audio_class",
        "audio_top_labels",
    ]
    fieldnames = base_fields + [field for field in extra_fields if field not in base_fields]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "kept.csv", kept, fieldnames)
    write_csv(args.output_dir / "rejected.csv", rejected, fieldnames)
    write_csv(args.output_dir / "all.csv", kept + rejected, fieldnames)

    report = {
        "input": str(args.input),
        "audio_classes": str(args.audio_classes) if args.audio_classes.exists() else None,
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "kept": len(kept),
        "rejected": len(rejected),
        "rejection_reasons": dict(reason_counts),
        "thresholds": {
            "min_chukchi_specific_ratio": args.min_chukchi_specific_ratio,
            "russian_score": args.russian_score_threshold,
            "hard_russian_score": args.hard_russian_score_threshold,
            "foreign_song_score": args.foreign_song_score_threshold,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
