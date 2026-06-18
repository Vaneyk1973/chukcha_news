"""Reusable machine-translation helper module for Chukchi News Voice."""

from __future__ import annotations

from typing import Any


def token_id(tokenizer: Any, token: str) -> int | None:
    """Token id for this pipeline stage."""
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or value == tokenizer.unk_token_id:
        return None
    return int(value)


def ensure_language_token(
    tokenizer: Any, model: Any | None, language: str, initialize_from: str
) -> int:
    """Ensure language token for this pipeline stage."""
    existing = token_id(tokenizer, language)
    if existing is not None:
        return existing

    source_id = token_id(tokenizer, initialize_from)
    if source_id is None:
        raise ValueError(
            f"Tokenizer does not contain initialization language token: {initialize_from}"
        )
    added = tokenizer.add_special_tokens({"additional_special_tokens": [language]})
    if added != 1:
        raise RuntimeError(f"Failed to add language token: {language}")
    language_id = token_id(tokenizer, language)
    if language_id is None:
        raise RuntimeError(f"Added language token cannot be resolved: {language}")

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings().weight.data
        embeddings[language_id].copy_(embeddings[source_id])
        output_embeddings = model.get_output_embeddings()
        if (
            output_embeddings is not None
            and output_embeddings.weight.data_ptr() != embeddings.data_ptr()
        ):
            output_embeddings.weight.data[language_id].copy_(
                output_embeddings.weight.data[source_id]
            )
    return language_id


def ensure_vocabulary_tokens(
    tokenizer: Any, model: Any | None, token_mapping: dict[str, str]
) -> list[int]:
    """Ensure vocabulary tokens for this pipeline stage."""
    missing = [token for token in token_mapping if token_id(tokenizer, token) is None]
    if not missing:
        return [token_id(tokenizer, token) for token in token_mapping]
    source_ids = {}
    for token in missing:
        source_id = token_id(tokenizer, token_mapping[token])
        if source_id is None:
            raise ValueError(
                f"Tokenizer does not contain initialization token: {token_mapping[token]}"
            )
        source_ids[token] = source_id
    tokenizer.add_tokens(missing)
    added_ids = [token_id(tokenizer, token) for token in missing]
    if any(value is None for value in added_ids):
        raise RuntimeError("One or more vocabulary tokens could not be added")

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings()
        for token, added_id in zip(missing, added_ids):
            source_id = source_ids[token]
            embeddings[added_id].copy_(embeddings[source_id])
            if (
                output_embeddings is not None
                and output_embeddings.weight.data_ptr() != embeddings.data_ptr()
            ):
                output_embeddings.weight.data[added_id].copy_(
                    output_embeddings.weight.data[source_id]
                )
    return [token_id(tokenizer, token) for token in token_mapping]


def configure_tokenizer(tokenizer: Any, direction: dict) -> None:
    """Configure tokenizer for this pipeline stage."""
    source_language = direction["source_language"]
    target_language = direction["target_language"]
    for language in (source_language, target_language):
        if token_id(tokenizer, language) is None:
            raise ValueError(f"Tokenizer is missing required language token: {language}")
    tokenizer.src_lang = source_language
    tokenizer.tgt_lang = target_language


def generation_kwargs(tokenizer: Any, direction: dict) -> dict[str, int]:
    """Generation kwargs for this pipeline stage."""
    target_id = token_id(tokenizer, direction["target_language"])
    if target_id is None:
        raise ValueError(f"Tokenizer is missing target language: {direction['target_language']}")
    return {"forced_bos_token_id": target_id}


def prefer_max_new_tokens(model: Any) -> None:
    """Prefer max new tokens for this pipeline stage."""
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "max_length"):
        generation_config.max_length = None
