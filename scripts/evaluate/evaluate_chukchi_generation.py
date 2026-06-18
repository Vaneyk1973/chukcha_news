#!/usr/bin/env python3
"""Ensemble evaluator for generated Chukchi text.

This is not a replacement for a native speaker. It is a failure-localization
tool: semantic drift, non-Chukchi form, lexicon miss, morphology-ish anomaly,
or plain generation junk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from chukcha_news.config import load_yaml, resolve_path  # noqa: E402
from chukcha_news.mt.metrics import normalize_chukchi_detokenization  # noqa: E402
from chukcha_news.mt.modeling import (  # noqa: E402
    configure_tokenizer,
    ensure_language_token,
    ensure_vocabulary_tokens,
    generation_kwargs,
    prefer_max_new_tokens,
)
from chukcha_news.mt.text_lm import CharNgramLM, generic_cyrillic_score  # noqa: E402


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁёӃӄӇӈԒԓ']+")
RU_TOKEN_RE = re.compile(r"[а-яё]+", re.IGNORECASE)
HAN_RE = re.compile(r"[\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
CHUKCHI_SPECIFIC = set("ӃӄӇӈԒԓ")
RUSSIAN_STOPS = {
    "и",
    "в",
    "во",
    "на",
    "по",
    "с",
    "со",
    "к",
    "ко",
    "о",
    "об",
    "от",
    "до",
    "для",
    "что",
    "это",
    "как",
    "не",
    "за",
    "из",
    "у",
    "а",
    "но",
    "или",
    "мы",
    "они",
    "он",
    "она",
    "будет",
    "были",
    "после",
}


@dataclass
class EvalInput:
    """Document the state and behavior for the `EvalInput` component."""

    sample_id: str
    chukchi_text: str
    source_ru: str = ""
    source_path: str = ""


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chukchi_eval.yaml")
    parser.add_argument("--input", type=Path, help="CSV/JSONL/TXT with generated Chukchi texts")
    parser.add_argument("--text", help="Single generated Chukchi text to evaluate")
    parser.add_argument("--source-ru", default="", help="Russian source/reference for one text")
    parser.add_argument("--server-dir", type=Path, help="Local server output dir to scan")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-backtranslation", action="store_true")
    parser.add_argument("--retrain-lm", action="store_true")
    return parser.parse_args()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp for this pipeline stage."""
    return min(max(value, low), high)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safe float for this pipeline stage."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def tokens(text: str) -> list[str]:
    """Tokens for this pipeline stage."""
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def russian_tokens(text: str) -> list[str]:
    """Russian tokens for this pipeline stage."""
    return [token.casefold() for token in RU_TOKEN_RE.findall(text)]


def read_jsonl(path: Path) -> list[dict]:
    """Read jsonl for this pipeline stage."""
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_inputs(args: argparse.Namespace, config: dict) -> list[EvalInput]:
    """Read inputs for this pipeline stage."""
    if args.text:
        return [EvalInput("inline", args.text.strip(), args.source_ru.strip(), "inline")]

    input_path = args.input
    if input_path:
        path = resolve_path(input_path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            raw_rows = read_jsonl(path)
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as input_file:
                raw_rows = list(csv.DictReader(input_file))
        else:
            raw_rows = [
                {"id": str(index), "chukchi_text": line}
                for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                if line.strip()
            ]
        rows = []
        for index, row in enumerate(raw_rows, 1):
            text = (
                row.get("chukchi_text")
                or row.get("text")
                or row.get("generated")
                or row.get("target_text")
                or row.get("transcript")
                or ""
            ).strip()
            if not text:
                continue
            rows.append(
                EvalInput(
                    sample_id=str(row.get("id") or row.get("sample_id") or index),
                    chukchi_text=text,
                    source_ru=(
                        row.get("source_ru") or row.get("news_ru") or row.get("source_text") or ""
                    ).strip(),
                    source_path=str(path),
                )
            )
        return rows

    server_dir = resolve_path(
        args.server_dir or config["input"].get("server_dir") or "outputs/server"
    )
    rows = []
    for ckt_path in sorted(server_dir.glob("*/news_ckt.txt")):
        ru_path = ckt_path.with_name("news_ru.txt")
        prompt_path = ckt_path.with_name("prompt.txt")
        source_ru = ""
        if ru_path.exists():
            source_ru = ru_path.read_text(encoding="utf-8").strip()
        elif prompt_path.exists():
            source_ru = prompt_path.read_text(encoding="utf-8").strip()
        rows.append(
            EvalInput(
                sample_id=ckt_path.parent.name,
                chukchi_text=ckt_path.read_text(encoding="utf-8").strip(),
                source_ru=source_ru,
                source_path=str(ckt_path),
            )
        )
    return rows


def load_text_lm(config: dict, retrain: bool) -> CharNgramLM:
    """Load text lm for this pipeline stage."""
    resources = config["resources"]
    model_path = resolve_path(resources["text_lm_model"])
    corpus_path = resolve_path(resources["trusted_corpus"])
    if model_path.exists() and not retrain:
        return CharNgramLM.load(model_path)
    texts = corpus_path.read_text(encoding="utf-8").splitlines()
    lm = CharNgramLM.train(texts, order=4)
    lm.save(model_path)
    return lm


def iter_corpus_texts(config: dict) -> list[str]:
    """Iter corpus texts for this pipeline stage."""
    resources = config["resources"]
    texts: list[str] = []
    trusted = resolve_path(resources["trusted_corpus"])
    if trusted.exists():
        texts.extend(
            line.strip()
            for line in trusted.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    mt_train = resolve_path(resources["mt_ru_ckt_train"])
    if mt_train.exists():
        for row in read_jsonl(mt_train):
            text = (row.get("target_text") or "").strip()
            if text:
                texts.append(text)

    asr_clean = resolve_path(resources["asr_clean"])
    if asr_clean.exists():
        with asr_clean.open("r", encoding="utf-8", newline="") as input_file:
            for row in csv.DictReader(input_file):
                text = (row.get("transcript") or row.get("text") or "").strip()
                if text:
                    texts.append(text)
    return texts


@dataclass
class CorpusStats:
    """Document the state and behavior for the `CorpusStats` component."""

    lexicon: set[str]
    char4: set[str]
    suffixes: Counter[str]
    word_lengths: list[int]


def build_corpus_stats(texts: list[str]) -> CorpusStats:
    """Build corpus stats for this pipeline stage."""
    lexicon: set[str] = set()
    char4: set[str] = set()
    suffixes: Counter[str] = Counter()
    word_lengths = []
    for text in texts:
        for token in tokens(text):
            if len(token) < 2:
                continue
            lexicon.add(token)
            word_lengths.append(len(token))
            padded = f"~{token}#"
            for index in range(max(len(padded) - 3, 0)):
                char4.add(padded[index : index + 4])
            for size in range(2, min(6, len(token) + 1)):
                suffixes[token[-size:]] += 1
    return CorpusStats(lexicon, char4, suffixes, word_lengths)


class BackTranslator:
    """Document the state and behavior for the `BackTranslator` component."""

    def __init__(self, config: dict) -> None:
        """Implement the `__init__` protocol hook for this object."""
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install MT dependencies with `make setup-mt`.") from error

        self.torch = torch
        self.config = config
        bt_config = config["backtranslation"]
        mt_config = load_yaml(bt_config["mt_config"])
        self.mt_config = mt_config
        self.direction = mt_config["directions"][bt_config.get("direction", "ckt_ru")]
        model_path = resolve_path(
            bt_config.get("model_path") or Path(self.direction["output_dir"]) / "final"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        prefer_max_new_tokens(self.model)
        ensure_vocabulary_tokens(
            self.tokenizer, self.model, mt_config["tokenizer"]["additional_tokens"]
        )
        ensure_language_token(
            self.tokenizer,
            self.model,
            self.direction["source_language"],
            self.direction["initialize_source_language_from"],
        )
        configure_tokenizer(self.tokenizer, self.direction)
        device_config = str(bt_config.get("device", "auto"))
        self.device = (
            "cuda" if device_config == "auto" and torch.cuda.is_available() else device_config
        )
        if self.device == "auto":
            self.device = "cpu"
        self.model.to(self.device).eval()

    def translate(self, text: str) -> str:
        """Translate for this pipeline stage."""
        bt_config = self.config["backtranslation"]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        generation = {
            "num_beams": int(bt_config.get("num_beams", 4)),
            "max_new_tokens": int(bt_config.get("max_new_tokens", 160)),
        }
        generation.update(generation_kwargs(self.tokenizer, self.direction))
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def semantic_similarity(source_ru: str, backtranslation_ru: str) -> float | None:
    """Semantic similarity for this pipeline stage."""
    if not source_ru.strip() or not backtranslation_ru.strip():
        return None
    import difflib

    source = russian_tokens(source_ru)
    back = russian_tokens(backtranslation_ru)
    if not source or not back:
        return 0.0
    source_set = set(source)
    back_set = set(back)
    jaccard = len(source_set & back_set) / max(len(source_set | back_set), 1)
    seq = difflib.SequenceMatcher(None, " ".join(source), " ".join(back)).ratio()
    return clamp(0.55 * seq + 0.45 * jaccard)


def chukchi_specific_ratio(text: str) -> float:
    """Chukchi specific ratio for this pipeline stage."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char in CHUKCHI_SPECIFIC for char in letters) / len(letters)


def repetition_stats(text_tokens: list[str]) -> tuple[float, float]:
    """Repetition stats for this pipeline stage."""
    if not text_tokens:
        return 1.0, 0.0
    counts = Counter(text_tokens)
    max_share = max(counts.values()) / len(text_tokens)
    distinct_ratio = len(counts) / len(text_tokens)
    return max_share, distinct_ratio


def lexicon_score(text_tokens: list[str], stats: CorpusStats) -> tuple[float, float, list[str]]:
    """Lexicon score for this pipeline stage."""
    if not text_tokens:
        return 0.0, 0.0, []
    known = [token for token in text_tokens if token in stats.lexicon]
    unknown = [token for token in text_tokens if token not in stats.lexicon]
    token_coverage = len(known) / len(text_tokens)
    char_hits = 0
    char_total = 0
    for token in text_tokens:
        padded = f"~{token}#"
        grams = [padded[index : index + 4] for index in range(max(len(padded) - 3, 0))]
        char_hits += sum(gram in stats.char4 for gram in grams)
        char_total += len(grams)
    char_coverage = char_hits / max(char_total, 1)
    return token_coverage, char_coverage, unknown[:12]


def percentile(values: list[int], q: float) -> float:
    """Percentile for this pipeline stage."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return float(ordered[index])


def morph_score(text_tokens: list[str], stats: CorpusStats) -> tuple[float, dict[str, float]]:
    """Morph score for this pipeline stage."""
    if not text_tokens:
        return 0.0, {"known_suffix_ratio": 0.0, "length_in_range_ratio": 0.0}
    suffix_hits = 0
    for token in text_tokens:
        candidates = [token[-size:] for size in range(2, min(6, len(token) + 1))]
        if any(stats.suffixes.get(candidate, 0) >= 5 for candidate in candidates):
            suffix_hits += 1
    known_suffix_ratio = suffix_hits / len(text_tokens)
    p05 = percentile(stats.word_lengths, 0.05)
    p95 = percentile(stats.word_lengths, 0.95)
    length_hits = sum(p05 <= len(token) <= p95 for token in text_tokens)
    length_in_range_ratio = length_hits / len(text_tokens)
    score = clamp(0.65 * known_suffix_ratio + 0.35 * length_in_range_ratio)
    return score, {
        "known_suffix_ratio": known_suffix_ratio,
        "length_in_range_ratio": length_in_range_ratio,
        "corpus_word_len_p05": p05,
        "corpus_word_len_p95": p95,
    }


def hygiene_score(
    text: str, text_tokens: list[str], thresholds: dict
) -> tuple[float, list[str], dict[str, float]]:
    """Hygiene score for this pipeline stage."""
    chars = max(len(text), 1)
    latin_ratio = len(LATIN_RE.findall(text)) / chars
    han_ratio = len(HAN_RE.findall(text)) / chars
    ru_stop_count = sum(1 for token in text_tokens if token in RUSSIAN_STOPS)
    ru_stop_ratio = ru_stop_count / max(len(text_tokens), 1)
    repeated_share, distinct_ratio = repetition_stats(text_tokens)
    run_penalty = 1.0 if re.search(r"(.)\1{8,}", text.casefold()) else 0.0

    penalties = [
        clamp(latin_ratio / max(float(thresholds["max_latin_ratio"]), 1e-6)),
        1.0 if han_ratio > float(thresholds["max_han_ratio"]) else 0.0,
        clamp(ru_stop_ratio / max(float(thresholds["max_russian_stopword_ratio"]), 1e-6)),
        clamp(
            (repeated_share - 0.12)
            / max(float(thresholds["max_repeated_token_share"]) - 0.12, 1e-6)
        ),
        run_penalty,
    ]
    score = clamp(1.0 - sum(penalties) / len(penalties))
    flags = []
    if latin_ratio > float(thresholds["max_latin_ratio"]):
        flags.append("latin_leak")
    if han_ratio > float(thresholds["max_han_ratio"]):
        flags.append("han_leak")
    if ru_stop_ratio > float(thresholds["max_russian_stopword_ratio"]):
        flags.append("russian_stopwords")
    if repeated_share > float(thresholds["max_repeated_token_share"]):
        flags.append("repetition")
    if run_penalty:
        flags.append("char_run")
    return (
        score,
        flags,
        {
            "latin_ratio": latin_ratio,
            "han_ratio": han_ratio,
            "russian_stopword_ratio": ru_stop_ratio,
            "max_repeated_token_share": repeated_share,
            "distinct_token_ratio": distinct_ratio,
        },
    )


def weighted_score(parts: dict[str, float | None]) -> float:
    """Weighted score for this pipeline stage."""
    weights = {
        "semantic": 0.35,
        "form": 0.25,
        "lexicon": 0.15,
        "morphology": 0.15,
        "hygiene": 0.10,
    }
    active = {key: value for key, value in parts.items() if value is not None}
    total_weight = sum(weights[key] for key in active)
    if total_weight <= 0:
        return 0.0
    return sum(float(active[key]) * weights[key] for key in active) / total_weight


def verdict(
    overall: float, parts: dict[str, float | None], flags: list[str], thresholds: dict
) -> str:
    """Verdict for this pipeline stage."""
    critical = set(flags) & {"han_leak", "repetition", "char_run"}
    if parts.get("semantic") is not None and float(parts["semantic"] or 0) < float(
        thresholds["min_semantic_score"]
    ):
        critical.add("semantic_drift")
    if float(parts.get("hygiene") or 0) < float(thresholds["min_hygiene_score"]):
        critical.add("dirty_output")
    if critical:
        return "reject"
    if overall >= float(thresholds["pass"]):
        return "pass"
    if overall >= float(thresholds["weak"]):
        return "weak"
    return "reject"


def failure_layers(parts: dict[str, float | None], flags: list[str], thresholds: dict) -> list[str]:
    """Failure layers for this pipeline stage."""
    layers = []
    if parts.get("semantic") is not None and float(parts["semantic"] or 0) < float(
        thresholds["min_semantic_score"]
    ):
        layers.append("semantic")
    if float(parts.get("form") or 0) < 0.45:
        layers.append("chukchi_form")
    if float(parts.get("lexicon") or 0) < float(thresholds["min_lexicon_coverage"]):
        layers.append("lexicon")
    if float(parts.get("morphology") or 0) < float(thresholds["min_morph_score"]):
        layers.append("morphology")
    if float(parts.get("hygiene") or 0) < float(thresholds["min_hygiene_score"]) or flags:
        layers.append("hygiene")
    return layers


def score_one(
    row: EvalInput,
    lm: CharNgramLM,
    stats: CorpusStats,
    thresholds: dict,
    translator: BackTranslator | None,
) -> dict:
    """Score one for this pipeline stage."""
    text = normalize_chukchi_detokenization(row.chukchi_text.strip())
    text_tokens = tokens(text)
    chukchi_lm_score = lm.average_log_probability(text)
    generic_score = generic_cyrillic_score(text)
    lm_margin = chukchi_lm_score - generic_score
    form_score = clamp(
        (lm_margin - float(thresholds["chukchi_lm_reject_margin"]))
        / max(
            float(thresholds["chukchi_lm_keep_margin"])
            - float(thresholds["chukchi_lm_reject_margin"]),
            1e-6,
        )
    )
    token_coverage, char_coverage, unknown_tokens = lexicon_score(text_tokens, stats)
    lex_score = clamp(0.45 * token_coverage + 0.55 * char_coverage)
    morph, morph_details = morph_score(text_tokens, stats)
    hygiene, flags, hygiene_details = hygiene_score(text, text_tokens, thresholds)
    back_ru = translator.translate(text) if translator else ""
    semantic = semantic_similarity(row.source_ru, back_ru)
    parts: dict[str, float | None] = {
        "semantic": semantic,
        "form": form_score,
        "lexicon": lex_score,
        "morphology": morph,
        "hygiene": hygiene,
    }
    overall = weighted_score(parts)
    layers = failure_layers(parts, flags, thresholds)
    sample_verdict = verdict(overall, parts, flags, thresholds)

    return {
        "id": row.sample_id,
        "verdict": sample_verdict,
        "overall_score": round(overall, 4),
        "semantic_score": "" if semantic is None else round(semantic, 4),
        "form_score": round(form_score, 4),
        "lexicon_score": round(lex_score, 4),
        "morphology_score": round(morph, 4),
        "hygiene_score": round(hygiene, 4),
        "failure_layers": ",".join(layers),
        "flags": ",".join(flags),
        "source_ru": row.source_ru,
        "backtranslation_ru": back_ru,
        "chukchi_text": text,
        "source_path": row.source_path,
        "chars": len(text),
        "tokens": len(text_tokens),
        "chukchi_specific_ratio": round(chukchi_specific_ratio(text), 4),
        "chukchi_lm_score": round(chukchi_lm_score, 6) if math.isfinite(chukchi_lm_score) else "",
        "generic_cyrillic_score": round(generic_score, 6) if math.isfinite(generic_score) else "",
        "chukchi_lm_margin": round(lm_margin, 6) if math.isfinite(lm_margin) else "",
        "lexicon_token_coverage": round(token_coverage, 4),
        "char4_coverage": round(char_coverage, 4),
        "unknown_tokens_sample": " ".join(unknown_tokens),
        **{key: round(value, 4) for key, value in morph_details.items()},
        **{key: round(value, 4) for key, value in hygiene_details.items()},
    }


def summarize(rows: list[dict]) -> dict:
    """Summarize for this pipeline stage."""
    if not rows:
        return {}
    score_keys = [
        "overall_score",
        "form_score",
        "lexicon_score",
        "morphology_score",
        "hygiene_score",
    ]
    semantic_values = [
        safe_float(row["semantic_score"], -1.0) for row in rows if row["semantic_score"] != ""
    ]
    summary = {
        "samples": len(rows),
        "verdict_counts": dict(Counter(row["verdict"] for row in rows)),
        "failure_layer_counts": dict(
            Counter(
                layer for row in rows for layer in str(row["failure_layers"]).split(",") if layer
            )
        ),
    }
    for key in score_keys:
        values = [safe_float(row[key]) for row in rows]
        summary[f"mean_{key}"] = round(statistics.mean(values), 4)
        summary[f"median_{key}"] = round(statistics.median(values), 4)
    if semantic_values:
        summary["mean_semantic_score"] = round(statistics.mean(semantic_values), 4)
        summary["median_semantic_score"] = round(statistics.median(semantic_values), 4)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write csv for this pipeline stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the command-line workflow for this module."""
    args = parse_args()
    config = load_yaml(args.config)
    rows = read_inputs(args, config)
    limit = args.limit if args.limit is not None else config["input"].get("limit")
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError("No generated Chukchi texts found to evaluate.")

    lm = load_text_lm(config, args.retrain_lm)
    stats = build_corpus_stats(iter_corpus_texts(config))
    translator = None
    if bool(config["backtranslation"].get("enabled", True)) and not args.skip_backtranslation:
        translator = BackTranslator(config)

    scored = []
    for index, row in enumerate(rows, 1):
        scored.append(score_one(row, lm, stats, config["thresholds"], translator))
        print(
            f"[eval-chukchi] {index}/{len(rows)} {row.sample_id}: {scored[-1]['verdict']}",
            flush=True,
        )

    samples_csv = resolve_path(config["output"]["samples_csv"])
    write_csv(samples_csv, scored)
    report = {
        "input_count": len(rows),
        "samples_csv": str(samples_csv),
        "backtranslation_enabled": translator is not None,
        "summary": summarize(scored),
        "thresholds": config["thresholds"],
    }
    report_path = resolve_path(config["output"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
