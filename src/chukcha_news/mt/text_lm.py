"""Reusable machine-translation helper module for Chukchi News Voice."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁёӃӄӇӈԒԓ']+")
CYRILLIC_ALPHABET = set(" абвгдеёжзийклмнопрстуфхцчшщъыьэюяӄӈԓ'-")


def normalize_text(text: str) -> str:
    """Normalize text for this pipeline stage."""
    text = text.casefold().replace("Ӄ".casefold(), "ӄ")
    text = text.replace("Ӈ".casefold(), "ӈ").replace("Ԓ".casefold(), "ԓ")
    tokens = TOKEN_RE.findall(text)
    return " ".join(tokens)


def char_ngrams(text: str, order: int) -> list[str]:
    """Char ngrams for this pipeline stage."""
    padded = f"{'~' * (order - 1)}{text}#"
    return [padded[index : index + order] for index in range(len(padded) - order + 1)]


class CharNgramLM:
    """Document the state and behavior for the `CharNgramLM` component."""

    def __init__(
        self,
        order: int,
        counts: Counter[str],
        context_counts: Counter[str],
        vocabulary: set[str],
        alpha: float = 0.1,
    ) -> None:
        """Implement the `__init__` protocol hook for this object."""
        self.order = order
        self.counts = counts
        self.context_counts = context_counts
        self.vocabulary = vocabulary
        self.alpha = alpha

    @classmethod
    def train(cls, texts: list[str], order: int = 4, alpha: float = 0.1) -> "CharNgramLM":
        """Train for this pipeline stage."""
        counts: Counter[str] = Counter()
        context_counts: Counter[str] = Counter()
        vocabulary = set()
        for raw_text in texts:
            text = normalize_text(raw_text)
            if not text:
                continue
            vocabulary.update(text)
            for gram in char_ngrams(text, order):
                counts[gram] += 1
                context_counts[gram[:-1]] += 1
        vocabulary.add("#")
        return cls(
            order=order,
            counts=counts,
            context_counts=context_counts,
            vocabulary=vocabulary,
            alpha=alpha,
        )

    def average_log_probability(self, text: str) -> float:
        """Average log probability for this pipeline stage."""
        normalized = normalize_text(text)
        if not normalized:
            return float("-inf")
        grams = char_ngrams(normalized, self.order)
        vocab_size = max(len(self.vocabulary), 1)
        total = 0.0
        for gram in grams:
            context = gram[:-1]
            total += math.log(
                (self.counts[gram] + self.alpha)
                / (self.context_counts[context] + self.alpha * vocab_size)
            )
        return total / len(grams)

    def to_dict(self) -> dict:
        """To dict for this pipeline stage."""
        return {
            "order": self.order,
            "alpha": self.alpha,
            "counts": dict(self.counts),
            "context_counts": dict(self.context_counts),
            "vocabulary": sorted(self.vocabulary),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "CharNgramLM":
        """From dict for this pipeline stage."""
        return cls(
            order=int(value["order"]),
            alpha=float(value["alpha"]),
            counts=Counter(value["counts"]),
            context_counts=Counter(value["context_counts"]),
            vocabulary=set(value["vocabulary"]),
        )

    def save(self, path: Path) -> None:
        """Save for this pipeline stage."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharNgramLM":
        """Load for this pipeline stage."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def generic_cyrillic_score(text: str) -> float:
    """Generic cyrillic score for this pipeline stage."""
    normalized = normalize_text(text)
    if not normalized:
        return float("-inf")
    unknown = sum(char not in CYRILLIC_ALPHABET for char in normalized)
    return -2.0 - unknown / max(len(normalized), 1)
