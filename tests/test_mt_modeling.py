"""Regression tests for the Chukchi News Voice pipeline."""

from chukcha_news.mt.modeling import (
    configure_tokenizer,
    ensure_language_token,
    ensure_vocabulary_tokens,
    generation_kwargs,
    prefer_max_new_tokens,
)


class FakeTokenizer:
    """Document the state and behavior for the `FakeTokenizer` component."""

    unk_token_id = 0

    def __init__(self) -> None:
        """Exercise the `__init__` behavior and guard against regressions."""
        self.tokens = {"<unk>": 0, "rus_Cyrl": 1, "kir_Cyrl": 2, "к": 3, "л": 4}

    def __len__(self) -> int:
        """Exercise the `__len__` behavior and guard against regressions."""
        return len(self.tokens)

    def convert_tokens_to_ids(self, token: str) -> int:
        """Exercise the `convert_tokens_to_ids` behavior and guard against regressions."""
        return self.tokens.get(token, self.unk_token_id)

    def add_special_tokens(self, values: dict) -> int:
        """Exercise the `add_special_tokens` behavior and guard against regressions."""
        token = values["additional_special_tokens"][0]
        self.tokens[token] = len(self.tokens)
        return 1

    def add_tokens(self, values: list[str]) -> int:
        """Exercise the `add_tokens` behavior and guard against regressions."""
        for token in values:
            self.tokens[token] = len(self.tokens)
        return len(values)


class FakeModel:
    """Document the state and behavior for the `FakeModel` component."""

    class GenerationConfig:
        """Document the state and behavior for the `GenerationConfig` component."""

        max_length = 200

    generation_config = GenerationConfig()


def test_add_and_configure_chukchi_language_token() -> None:
    """
    Exercise the `test_add_and_configure_chukchi_language_token` behavior and guard against regressions.
    """
    tokenizer = FakeTokenizer()
    token = ensure_language_token(tokenizer, None, "ckt_Cyrl", "kir_Cyrl")
    direction = {"source_language": "rus_Cyrl", "target_language": "ckt_Cyrl"}
    configure_tokenizer(tokenizer, direction)
    assert token == 5
    assert tokenizer.src_lang == "rus_Cyrl"
    assert tokenizer.tgt_lang == "ckt_Cyrl"
    assert generation_kwargs(tokenizer, direction) == {"forced_bos_token_id": 5}


def test_add_chukchi_alphabet_tokens() -> None:
    """
    Exercise the `test_add_chukchi_alphabet_tokens` behavior and guard against regressions.
    """
    tokenizer = FakeTokenizer()
    ids = ensure_vocabulary_tokens(tokenizer, None, {"ӄ": "к", "ԓ": "л"})
    assert ids == [5, 6]


def test_prefer_max_new_tokens_clears_default_max_length() -> None:
    """
    Exercise the `test_prefer_max_new_tokens_clears_default_max_length` behavior and guard against regressions.
    """
    model = FakeModel()
    prefer_max_new_tokens(model)
    assert model.generation_config.max_length is None
