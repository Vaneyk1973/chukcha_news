#!/usr/bin/env python3
"""Local browser server: prompt -> Russian news -> Chukchi translation -> speech."""

from __future__ import annotations

import argparse
import copy
import csv
import contextlib
import gc
import json
import mimetypes
import os
import re
import signal
import sys
import threading
import time
import uuid
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "web" / "server"
sys.path.insert(0, str(ROOT))
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
from chukcha_news.mt.quality import (  # noqa: E402
    has_repetition_collapse,
    mt_generation_quality_args,
)


PIPELINE_LOCK = threading.Lock()
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
BAD_LLM_PHRASES = (
    "не могу",
    "извин",
    "as an ai",
    "i cannot",
    "无法",
    "不能",
    "抱歉",
)
WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")
_TRANSLATION_MEMORY: list[dict] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/server.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--mock-llm", action="store_true")
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    with contextlib.suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def model_options_payload(config: dict) -> dict:
    groups = {}
    for group, group_config in config.get("model_options", {}).items():
        groups[group] = {
            "default": group_config.get("default"),
            "choices": [
                {
                    "key": key,
                    "label": value.get("label", key),
                    "description": value.get("description", ""),
                }
                for key, value in group_config.get("choices", {}).items()
            ],
        }
    return groups


def apply_model_selection(config: dict, selection: dict | None) -> dict:
    selected = copy.deepcopy(config)
    selection = selection or {}
    options = selected.get("model_options", {})

    def choice(group: str) -> tuple[str, dict]:
        group_config = options.get(group, {})
        choices = group_config.get("choices", {})
        key = str(selection.get(group) or group_config.get("default") or "")
        if key not in choices:
            raise ValueError(f"Unknown {group} model choice: {key}")
        return key, choices[key]

    if "llm_news" in options:
        _, news = choice("llm_news")
        selected["llm"].update(
            {
                key: value
                for key, value in news.items()
                if key not in {"label", "description"}
            }
        )

    if "direct_chukchi" in options:
        _, direct = choice("direct_chukchi")
        selected["llm"].update(
            {
                key: value
                for key, value in direct.items()
                if key not in {"label", "description"}
            }
        )

    if "mt_ru_ckt" in options:
        _, mt = choice("mt_ru_ckt")
        selected["mt"]["config"] = mt.get("config", selected["mt"]["config"])
        selected["mt"]["model_path"] = mt.get("model_path", selected["mt"]["model_path"])
        if "translation_memory_files" in mt:
            selected["mt"].setdefault("translation_memory", {})["files"] = mt[
                "translation_memory_files"
            ]

    if "mt_ckt_ru" in options:
        _, back = choice("mt_ckt_ru")
        selected["mt"]["backtranslation_model_path"] = back.get(
            "model_path", selected["mt"]["backtranslation_model_path"]
        )
        selected["mt"]["backtranslation_config"] = back.get(
            "config", selected["mt"].get("backtranslation_config", selected["mt"]["config"])
        )

    if "tts" in options:
        _, tts = choice("tts")
        selected["tts"].update(
            {
                key: value
                for key, value in tts.items()
                if key not in {"label", "description"}
            }
        )
    return selected


def public_output_url(path: Path) -> str:
    return "/" + str(path.relative_to(ROOT)).replace(os.sep, "/")


class HFNewsRuntime:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.base_model = str(config["hf_news_base_model"])
        self.max_new_tokens = int(config.get("hf_news_max_new_tokens", 180))

    def __enter__(self) -> "HFNewsRuntime":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("Install LLM dependencies with: python3 -m pip install -e '.[llm]'") from error

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization_config = None
        if bool(self.config.get("hf_news_load_in_4bit", True)):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for name in ("model", "tokenizer"):
            if hasattr(self, name):
                delattr(self, name)
        clear_cuda()

    def generate_news_stream(self, user_prompt: str, news_style: str) -> Iterator[str]:
        system_prompt = (
            "Ты редактор местной радиостанции. По пользовательскому промпту напиши готовый "
            "текст новости строго на русском языке. Не переводи промпт дословно: осмысли "
            "тему и преврати ее в короткий выпуск новости. Используй только кириллицу, "
            "цифры и обычную пунктуацию. Запрещены китайский, английский, markdown, списки, "
            "заголовки, дисклеймеры, извинения и фразы о невозможности выполнить задачу. "
            f"Стиль: {news_style}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.25,
                top_p=0.8,
                repetition_penalty=1.08,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        for index, word in enumerate(text.split(" ")):
            if word:
                yield word if index == 0 else f" {word}"


class HFLoraRuntime:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.base_model = str(config["hf_base_model"])
        self.adapter = resolve_path(config["hf_lora_adapter"])
        self.max_new_tokens = int(config.get("hf_max_new_tokens", 140))

    def __enter__(self) -> "HFLoraRuntime":
        if not self.adapter.exists():
            raise RuntimeError(
                f"LoRA adapter is missing: {self.adapter}. Run `make train-llm-chukchi` first."
            )
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("Install LLM dependencies with: python3 -m pip install -e '.[llm]'") from error

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization_config = None
        if bool(self.config.get("hf_load_in_4bit", True)):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter).eval()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for name in ("model", "tokenizer"):
            if hasattr(self, name):
                delattr(self, name)
        clear_cuda()

    def generate_chukchi_stream(self, user_prompt: str, direct_style: str) -> Iterator[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты пишешь только на чукотском языке кириллицей. "
                    "Не используй русский, английский, китайский, markdown и пояснения."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Напиши короткий чукотский текст новости по теме: {user_prompt}. "
                    f"Стиль: {direct_style}."
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.45,
                top_p=0.8,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        for index, word in enumerate(text.split(" ")):
            if word:
                yield word if index == 0 else f" {word}"


def mock_news(prompt: str) -> str:
    topic = prompt.strip().rstrip(".") or "важном событии в регионе"
    return (
        f"Сегодня стало известно о {topic}. Представители местных служб сообщили, "
        "что ситуация находится под наблюдением. Жителям рекомендуют следить за "
        "официальными сообщениями и учитывать обновления в течение дня."
    )


def mock_news_stream(prompt: str) -> Iterator[str]:
    words = mock_news(prompt).split(" ")
    for index, word in enumerate(words):
        yield word if index == 0 else f" {word}"
        time.sleep(0.035)


def mock_chukchi_text(prompt: str) -> str:
    return (
        "Игыр Чукоткак нымытваԓьыт нэнаԓгыӄинэт. "
        "Оравэтԓьат ынӄэн нымытваԓьыт эпы нэнъэӈэӈэтынэт. "
        "Мынгыргыԓьыт нэнаԓгыӄинэт."
    )


def mock_chukchi_stream(prompt: str) -> Iterator[str]:
    words = mock_chukchi_text(prompt).split(" ")
    for index, word in enumerate(words):
        yield word if index == 0 else f" {word}"
        time.sleep(0.035)


def generate_news_stream(prompt: str, config: dict, mock_llm: bool) -> Iterator[str]:
    if mock_llm:
        yield from mock_news_stream(prompt)
        return
    llm_config = config["llm"]
    if llm_config.get("news_backend") != "hf_qwen":
        raise RuntimeError(f"Unsupported news LLM backend: {llm_config.get('news_backend')}")
    with HFNewsRuntime(llm_config) as runtime:
        yield from runtime.generate_news_stream(prompt, str(llm_config["news_style"]))


def generate_chukchi_stream(prompt: str, config: dict, mock_llm: bool) -> Iterator[str]:
    if mock_llm:
        yield from mock_chukchi_stream(prompt)
        return
    llm_config = config["llm"]
    backend = llm_config.get("direct_chukchi_backend", "hf_lora")
    if backend == "hf_lora":
        with HFLoraRuntime(llm_config) as runtime:
            yield from runtime.generate_chukchi_stream(
                prompt, str(llm_config["direct_chukchi_style"])
            )
        return
    raise RuntimeError(f"Unsupported direct Chukchi backend: {backend}")


def clean_news_text(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^\s*(текст новости|новость|заголовок)\s*:\s*", "", text, flags=re.I)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*•\d.)]+\s*", "", line).strip()
        if line:
            lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def validate_russian_news(text: str) -> None:
    lowered = text.lower()
    cyrillic = len(CYRILLIC_RE.findall(text))
    han = len(HAN_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    if len(text) < 60:
        raise RuntimeError("LLM generated a news text that is too short. Try a more concrete prompt.")
    if han:
        raise RuntimeError("LLM generated non-Russian text. Try again or switch to another local LLM.")
    if any(phrase in lowered for phrase in BAD_LLM_PHRASES):
        raise RuntimeError("LLM refused instead of writing a news item. Rephrase the prompt and try again.")
    if cyrillic < 40 or latin > max(20, cyrillic * 0.15):
        raise RuntimeError("LLM output is not clean Russian news text. Try again.")


def validate_chukchi_text(text: str, source: str = "MT") -> None:
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < 20:
        raise RuntimeError(f"{source} generated an empty or too-short Chukchi text.")
    if has_repetition_collapse(text):
        raise RuntimeError(
            f"{source} generated a repeated-token collapse. This prompt is not safe for the server; "
            "try a shorter, simpler news prompt."
        )
    if HAN_RE.search(text) or LATIN_RE.search(text):
        raise RuntimeError(f"{source} generated non-Chukchi mixed-script text.")


def normalize_ru_for_match(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def char_ngrams(text: str, size: int = 3) -> set[str]:
    text = normalize_ru_for_match(text)
    if len(text) <= size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def sentence_similarity(left: str, right: str) -> float:
    left_words = set(normalize_ru_for_match(left).split())
    right_words = set(normalize_ru_for_match(right).split())
    word_score = jaccard(left_words, right_words)
    char_score = jaccard(char_ngrams(left), char_ngrams(right))
    return 0.55 * char_score + 0.45 * word_score


def load_translation_memory(config: dict) -> list[dict]:
    global _TRANSLATION_MEMORY
    cache_key = tuple(config["mt"].get("translation_memory", {}).get("files", []))
    if (
        _TRANSLATION_MEMORY is not None
        and _TRANSLATION_MEMORY
        and _TRANSLATION_MEMORY[0].get("cache_key") == cache_key
    ):
        return _TRANSLATION_MEMORY
    memory_config = config["mt"].get("translation_memory", {})
    limit = int(memory_config.get("max_candidates", 60000))
    rows = []
    seen = set()
    for file_name in memory_config.get("files", []):
        path = resolve_path(file_name)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                source = str(row.get("source_text", "")).strip()
                target = str(row.get("target_text", "")).strip()
                if not source or not target:
                    continue
                key = f"{source}\0{target}".casefold()
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "cache_key": cache_key,
                        "source": source,
                        "target": target,
                        "normalized": normalize_ru_for_match(source),
                        "chars": char_ngrams(source),
                        "words": set(normalize_ru_for_match(source).split()),
                    }
                )
                if len(rows) >= limit:
                    break
        if len(rows) >= limit:
            break
    _TRANSLATION_MEMORY = rows
    return rows


def translation_memory_lookup(sentence: str, config: dict) -> dict | None:
    memory_config = config["mt"].get("translation_memory", {})
    if not bool(memory_config.get("enabled", False)):
        return None
    min_similarity = float(memory_config.get("min_similarity", 0.62))
    query_words = set(normalize_ru_for_match(sentence).split())
    query_chars = char_ngrams(sentence)
    best = None
    best_score = 0.0
    for row in load_translation_memory(config):
        score = 0.55 * jaccard(query_chars, row["chars"]) + 0.45 * jaccard(query_words, row["words"])
        if score > best_score:
            best_score = score
            best = row
    if best and best_score >= min_similarity:
        return {"score": best_score, **best}
    return None


def split_russian_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        if len(cleaned) > 220:
            sentences.extend(piece.strip() for piece in re.split(r"[;:]\s+|,\s+(?=а|но|и|кроме|старшие|молодые)", cleaned) if piece.strip())
        else:
            sentences.append(cleaned)
    return sentences or [text.strip()]


def translate_with_mt(text: str, config: dict, direction_key: str, model_path_key: str, max_new_tokens_key: str) -> str:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config_key = "backtranslation_config" if direction_key == "ckt_ru" else "config"
    mt_config = load_yaml(config["mt"].get(config_key, config["mt"]["config"]))
    direction = mt_config["directions"][direction_key]
    model_path = resolve_path(config["mt"].get(model_path_key) or Path(direction["output_dir"]) / "final")

    device = str(config["mt"].get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    prefer_max_new_tokens(model)
    ensure_vocabulary_tokens(tokenizer, model, mt_config["tokenizer"]["additional_tokens"])
    if direction_key == "ru_ckt":
        ensure_language_token(
            tokenizer,
            model,
            direction["target_language"],
            direction["initialize_target_language_from"],
        )
    else:
        ensure_language_token(
            tokenizer,
            model,
            direction["source_language"],
            direction["initialize_source_language_from"],
        )
    configure_tokenizer(tokenizer, direction)
    model.to(device).eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    generate_args = {
        "num_beams": int(config["mt"].get("num_beams", mt_config["training"]["generation_num_beams"])),
        "max_new_tokens": int(
            config["mt"].get(max_new_tokens_key, mt_config["training"]["max_target_length"])
        ),
    }
    if direction_key == "ru_ckt":
        generate_args.update(
            {
                "no_repeat_ngram_size": int(config["mt"].get("no_repeat_ngram_size", 0)),
                "repetition_penalty": float(config["mt"].get("repetition_penalty", 1.0)),
            }
        )
        generate_args.update(mt_generation_quality_args(direction_key))
    generate_args.update(generation_kwargs(tokenizer, direction))
    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_args)
    translated = tokenizer.decode(generated[0], skip_special_tokens=True)
    if direction["target_column"] == "ckt":
        translated = normalize_chukchi_detokenization(translated)
    del model
    del tokenizer
    clear_cuda()
    return translated


def translate_ru_to_chukchi(text: str, config: dict) -> str:
    translated_sentences = []
    for sentence in split_russian_sentences(text):
        match = translation_memory_lookup(sentence, config)
        if match:
            translated_sentences.append(normalize_chukchi_detokenization(match["target"]))
        else:
            translated_sentences.append(
                translate_with_mt(
                    sentence,
                    config,
                    str(config["mt"].get("direction", "ru_ckt")),
                    "model_path",
                    "max_new_tokens",
                )
            )
    translated = " ".join(translated_sentences)
    validate_chukchi_text(translated, "MT")
    return translated


def write_translation_memory_matches(text: str, config: dict, output_path: Path) -> None:
    rows = []
    for sentence in split_russian_sentences(text):
        match = translation_memory_lookup(sentence, config)
        rows.append(
            {
                "sentence": sentence,
                "used_memory": bool(match),
                "similarity": f"{match['score']:.4f}" if match else "",
                "matched_ru": match["source"] if match else "",
                "matched_ckt": match["target"] if match else "",
            }
        )
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def translate_chukchi_to_ru(text: str, config: dict) -> str:
    return translate_with_mt(
        text,
        config,
        str(config["mt"].get("backtranslation_direction", "ckt_ru")),
        "backtranslation_model_path",
        "backtranslation_max_new_tokens",
    )


def synthesize_chukchi(text: str, config: dict, output_path: Path) -> tuple[Path, float]:
    import torch
    from transformers.utils import logging as transformers_logging
    from transformers import VitsModel, VitsTokenizer

    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    model_path = config["tts"].get("model_path") or load_yaml(config["tts"]["config"])["model"][
        "base_model"
    ]
    device = str(config["tts"].get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = VitsTokenizer.from_pretrained(model_path)
    model = VitsModel.from_pretrained(model_path).to(device).eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**inputs)
    audio = output.waveform[0].detach().cpu().numpy()
    sample_rate = int(model.config.sampling_rate)
    write_wav(output_path, audio, sample_rate)
    duration = len(audio) / sample_rate
    del output
    del model
    del tokenizer
    clear_cuda()
    return output_path, duration


def run_translated_pipeline(prompt: str, config: dict, mock_llm: bool, output_dir: Path) -> Iterator[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    yield {"stage": "news", "status": "running", "message": "Генерируем русскую новость"}
    news_chunks = []
    for chunk in generate_news_stream(prompt, config, mock_llm):
        news_chunks.append(chunk)
        yield {"stage": "news", "status": "partial", "text": "".join(news_chunks)}
    news_text = clean_news_text("".join(news_chunks))
    if not news_text:
        raise RuntimeError("Local LLM returned an empty news text.")
    validate_russian_news(news_text)
    yield {"stage": "news", "status": "partial", "text": news_text}
    (output_dir / "news_ru.txt").write_text(news_text, encoding="utf-8")
    yield {"stage": "news", "status": "done", "text": news_text}

    yield {"stage": "translation", "status": "running", "message": "Переводим на чукотский"}
    write_translation_memory_matches(news_text, config, output_dir / "translation_memory.csv")
    chukchi_text = translate_ru_to_chukchi(news_text, config)
    (output_dir / "news_ckt.txt").write_text(chukchi_text, encoding="utf-8")
    yield {"stage": "translation", "status": "done", "text": chukchi_text}

    yield {"stage": "backtranslation", "status": "running", "message": "Переводим чукотский обратно на русский"}
    backtranslation_text = translate_chukchi_to_ru(chukchi_text, config)
    (output_dir / "news_back_ru.txt").write_text(backtranslation_text, encoding="utf-8")
    yield {"stage": "backtranslation", "status": "done", "text": backtranslation_text}

    yield {"stage": "tts", "status": "running", "message": "Озвучиваем baseline MMS TTS"}
    audio_path, duration = synthesize_chukchi(chukchi_text, config, output_dir / "news_ckt.wav")
    yield {
        "stage": "tts",
        "status": "done",
        "audio_url": public_output_url(audio_path),
        "duration_sec": round(duration, 3),
    }


def run_direct_chukchi_pipeline(
    prompt: str, config: dict, mock_llm: bool, output_dir: Path
) -> Iterator[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    yield {
        "stage": "translation",
        "status": "running",
        "message": "LLM пишет чукотский текст напрямую",
    }
    chunks = []
    for chunk in generate_chukchi_stream(prompt, config, mock_llm):
        chunks.append(chunk)
        yield {"stage": "translation", "status": "partial", "text": "".join(chunks)}
    chukchi_text = clean_news_text("".join(chunks))
    validate_chukchi_text(chukchi_text, "Direct Chukchi LLM")
    (output_dir / "news_ckt.txt").write_text(chukchi_text, encoding="utf-8")
    yield {"stage": "translation", "status": "done", "text": chukchi_text}

    yield {"stage": "backtranslation", "status": "running", "message": "Переводим чукотский обратно на русский"}
    backtranslation_text = translate_chukchi_to_ru(chukchi_text, config)
    (output_dir / "news_back_ru.txt").write_text(backtranslation_text, encoding="utf-8")
    yield {"stage": "backtranslation", "status": "done", "text": backtranslation_text}

    yield {"stage": "tts", "status": "running", "message": "Озвучиваем baseline MMS TTS"}
    audio_path, duration = synthesize_chukchi(chukchi_text, config, output_dir / "news_ckt.wav")
    yield {
        "stage": "tts",
        "status": "done",
        "audio_url": public_output_url(audio_path),
        "duration_sec": round(duration, 3),
    }


def run_pipeline(prompt: str, mode: str, config: dict, mock_llm: bool) -> Iterator[dict]:
    run_id = uuid.uuid4().hex[:12]
    output_dir = resolve_path(config["output"]["dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "mode.txt").write_text(mode, encoding="utf-8")
    (output_dir / "model_selection.json").write_text(
        json.dumps(
            {
                "llm": config["llm"].get("model"),
                "direct_chukchi_backend": config["llm"].get("direct_chukchi_backend"),
                "mt_config": config["mt"].get("config"),
                "mt_model_path": config["mt"].get("model_path"),
                "backtranslation_config": config["mt"].get("backtranslation_config", config["mt"].get("config")),
                "backtranslation_model_path": config["mt"].get("backtranslation_model_path"),
                "tts_model_path": config["tts"].get("model_path"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if mode == "translated":
        yield from run_translated_pipeline(prompt, config, mock_llm, output_dir)
        yield {"stage": "complete", "status": "done", "run_id": run_id}
    elif mode == "direct_chukchi":
        yield {"stage": "news", "status": "skipped", "text": "Direct Chukchi mode: русский этап пропущен."}
        yield from run_direct_chukchi_pipeline(prompt, config, mock_llm, output_dir)
        yield {"stage": "complete", "status": "done", "run_id": run_id}
    else:
        raise RuntimeError(f"Unsupported server mode: {mode}")


class ServerHandler(SimpleHTTPRequestHandler):
    server_version = "ChukchaNews/0.1"

    def do_GET(self) -> None:
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if self.path == "/api/options":
            self.send_json(200, model_options_payload(self.server.config))
            return
        if self.path == "/":
            self.path = "/index.html"
        if self.path.startswith("/outputs/server/"):
            return self.serve_file(ROOT / self.path.lstrip("/"))
        return self.serve_file(STATIC_DIR / self.path.lstrip("/"))

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = str(payload.get("prompt", "")).strip()
            mode = str(payload.get("mode", "translated")).strip()
            model_selection = payload.get("models") or {}
            if not prompt:
                raise ValueError("Prompt is empty.")
            run_config = apply_model_selection(self.server.config, model_selection)
        except Exception as error:
            self.send_json(400, {"error": str(error)})
            return

        if not PIPELINE_LOCK.acquire(blocking=False):
            self.send_json(409, {"error": "Pipeline is already running. Wait for it to finish."})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for event in run_pipeline(prompt, mode, run_config, self.server.mock_llm):
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
        except Exception as error:
            self.wfile.write(
                (json.dumps({"stage": "error", "status": "error", "error": str(error)}, ensure_ascii=False) + "\n").encode(
                    "utf-8"
                )
            )
            self.wfile.flush()
        finally:
            PIPELINE_LOCK.release()

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as input_file:
            self.wfile.write(input_file.read())

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        print(f"[server] {self.address_string()} - {format % args}")


class LocalNewsServer(ThreadingHTTPServer):
    config: dict
    mock_llm: bool


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    host = args.host or str(config["server"]["host"])
    port = args.port or int(config["server"]["port"])
    server = LocalNewsServer((host, port), ServerHandler)
    server.config = config
    server.mock_llm = args.mock_llm

    def stop(signum, frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    print(f"Server: http://{host}:{port}")
    print("LLM mode:", "mock" if args.mock_llm else f"hf:{config['llm']['hf_news_base_model']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
