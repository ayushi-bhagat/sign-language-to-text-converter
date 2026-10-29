"""Unit tests for TextComposer."""

import pytest
from sign_language.config.schema import TextConfig
from sign_language.text.composer import TextComposer


@pytest.fixture
def text_config() -> TextConfig:
    return TextConfig(
        separator=" ",
        max_length=50,
        auto_capitalise=False,
        strip_underscores=True,
        enable_word_suggestions=False,
        enable_text_to_speech=False,
    )


@pytest.mark.unit
def test_composer_append_and_formatting(text_config: TextConfig) -> None:
    """Verify appending words and underscore stripping for display."""
    composer = TextComposer(text_config)
    composer.append("HELLO")
    assert composer.formatted == "HELLO"

    composer.append("THANK_YOU")
    assert composer.formatted == "HELLO THANK YOU"
    assert composer.raw_text == "HELLO THANK_YOU"


@pytest.mark.unit
def test_composer_neutral_emits_nothing(text_config: TextConfig) -> None:
    """Verify that _NEUTRAL is strictly ignored and emits no text."""
    composer = TextComposer(text_config)
    composer.append("_NEUTRAL")
    assert composer.formatted == ""

    composer.append("HELLO")
    composer.append("_NEUTRAL")
    assert composer.formatted == "HELLO"


@pytest.mark.unit
def test_composer_backspace_removes_word(text_config: TextConfig) -> None:
    """Verify word-based backspace."""
    composer = TextComposer(text_config)
    composer.append("HELLO")
    composer.append("WORLD")
    assert composer.formatted == "HELLO WORLD"

    composer.backspace()
    assert composer.formatted == "HELLO"

    composer.backspace()
    assert composer.formatted == ""

    # Backspacing on empty buffer should be safe
    composer.backspace()
    assert composer.formatted == ""


@pytest.mark.unit
def test_composer_clear(text_config: TextConfig) -> None:
    """Verify clear empties buffer."""
    composer = TextComposer(text_config)
    composer.append("HELLO")
    composer.append("THANK_YOU")
    assert composer.clear() == ""
    assert composer.formatted == ""


@pytest.mark.unit
def test_composer_max_length_truncation(text_config: TextConfig) -> None:
    """Verify FIFO truncation when buffer exceeds max_length."""
    # text_config max_length is 50
    composer = TextComposer(text_config)
    for _ in range(10):
        composer.append("LONGWORD")

    assert len(composer.formatted) <= 50
