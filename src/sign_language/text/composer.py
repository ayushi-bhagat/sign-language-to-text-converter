"""Text composition engine for accumulating recognised sign labels into sentences.

Handles:
    - Appending committed sign labels with configurable separator
    - Filtering out rejection/control classes (e.g. _NEUTRAL) so they never emit text
    - Word-level backspacing (removing the last committed sign)
    - Enforcing maximum buffer length limits (FIFO trimming)
    - Formatting text for display (stripping underscores: THANK_YOU -> THANK YOU)
"""

from __future__ import annotations

import logging
from sign_language.config.schema import TextConfig

logger = logging.getLogger(__name__)


class TextComposer:
    """Accumulates and formats recognized sign language labels into text.

    Args:
        config: Central TextConfig configuration.
        neutral_class: The rejection class label that emits no text (default '_NEUTRAL').
    """

    def __init__(self, config: TextConfig, neutral_class: str = "_NEUTRAL") -> None:
        self._cfg = config
        self._neutral_class = neutral_class
        self._tokens: list[str] = []

    def append(self, label: str) -> str:
        """Append a committed sign label to the text buffer.

        Args:
            label: Canonical sign label string (e.g. 'HELLO', 'THANK_YOU').

        Returns:
            str: Updated formatted display text.
        """
        # Rejection class (_NEUTRAL) must never produce text output
        if label == self._neutral_class or not label:
            return self.formatted

        self._tokens.append(label)
        self._enforce_max_length()
        return self.formatted

    def backspace(self) -> str:
        """Remove the last committed word/sign from the buffer.

        Returns:
            str: Updated formatted display text.
        """
        if self._tokens:
            removed = self._tokens.pop()
            logger.debug("TextComposer backspace removed token: %s", removed)
        return self.formatted

    def clear(self) -> str:
        """Clear all accumulated text from the buffer.

        Returns:
            str: Empty string.
        """
        self._tokens.clear()
        return ""

    def add_space(self) -> str:
        """Insert an explicit space token."""
        self._tokens.append("")
        self._enforce_max_length()
        return self.formatted

    @property
    def raw_tokens(self) -> list[str]:
        """Return the immutable sequence of accumulated raw sign tokens."""
        return list(self._tokens)

    @property
    def raw_text(self) -> str:
        """Return concatenated text using the configured separator."""
        return self._cfg.separator.join(self._tokens)

    @property
    def formatted(self) -> str:
        """Return text formatted for display according to configuration flags."""
        text = self.raw_text
        if self._cfg.strip_underscores:
            text = text.replace("_", " ")

        if self._cfg.auto_capitalise:
            # Capitalize first letter of each sentence
            text = ". ".join(s.capitalize() for s in text.split(". "))

        return text

    def _enforce_max_length(self) -> None:
        """Trim oldest tokens from the front if accumulated text exceeds max_length."""
        max_len = self._cfg.max_length
        while self._tokens and len(self.formatted) > max_len:
            self._tokens.pop(0)
