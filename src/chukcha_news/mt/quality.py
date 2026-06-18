from __future__ import annotations

import re
from collections import Counter


TEXT_TOKEN_RE = re.compile(r"[А-Яа-яЁёӃӄӇӈԒԓA-Za-z']+")


def repetition_features(text: str) -> dict[str, float]:
    tokens = [token.casefold() for token in TEXT_TOKEN_RE.findall(text)]
    if not tokens:
        return {
            "token_count": 0,
            "max_token_share": 0.0,
            "longest_token_run": 0,
            "distinct_token_ratio": 0.0,
            "char_run": 0,
        }

    counts = Counter(tokens)
    longest_run = 1
    current_run = 1
    for previous, current in zip(tokens, tokens[1:]):
        if previous == current:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    compact = re.sub(r"\s+", "", text.casefold())
    char_run = 0
    if compact:
        run = 1
        for previous, current in zip(compact, compact[1:]):
            if previous == current:
                run += 1
                char_run = max(char_run, run)
            else:
                run = 1

    return {
        "token_count": len(tokens),
        "max_token_share": max(counts.values()) / len(tokens),
        "longest_token_run": longest_run,
        "distinct_token_ratio": len(counts) / len(tokens),
        "char_run": char_run,
    }


def has_repetition_collapse(text: str) -> bool:
    features = repetition_features(text)
    token_count = int(features["token_count"])
    if token_count >= 4 and features["longest_token_run"] >= 3:
        return True
    if token_count >= 8 and features["max_token_share"] > 0.28:
        return True
    if token_count >= 8 and features["distinct_token_ratio"] < 0.38:
        return True

    compact = re.sub(r"\s+", "", text.casefold())
    if len(compact) >= 30:
        if features["char_run"] >= 8:
            return True
        for size in range(2, 9):
            for start in range(0, min(size, len(compact))):
                chunk = compact[start : start + size]
                if chunk and chunk * 8 in compact:
                    return True
    return False


def mt_generation_quality_args(direction_key: str) -> dict[str, float | int]:
    if direction_key != "ru_ckt":
        return {}
    return {
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.45,
        "encoder_no_repeat_ngram_size": 3,
    }
