"""Reusable machine-translation helper module for Chukchi News Voice."""

from __future__ import annotations

import re


CHUKCHI_LETTERS = "ӃӄӇӈԒԓ"
_SPACE_AFTER_ADDED_CHUKCHI_LETTER = re.compile(rf"(?<=[{CHUKCHI_LETTERS}]) (?=[а-яёьъ])")


def normalize_chukchi_detokenization(text: str) -> str:
    """Normalize chukchi detokenization for this pipeline stage."""
    previous = text
    while True:
        current = _SPACE_AFTER_ADDED_CHUKCHI_LETTER.sub("", previous)
        if current == previous:
            return current
        previous = current


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate for this pipeline stage."""
    reference = reference.strip()
    hypothesis = hypothesis.strip()
    if not reference:
        return 0.0 if not hypothesis else 1.0

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


def mean_character_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Mean character error rate for this pipeline stage."""
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have equal length")
    if not references:
        return 0.0
    return sum(character_error_rate(ref, hyp) for ref, hyp in zip(references, hypotheses)) / len(
        references
    )
