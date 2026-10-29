"""Prediction stabilization using rolling-window voting, debounce cooldown, and neutral repeat gating."""

from __future__ import annotations

from collections import Counter, deque
from enum import Enum
import logging

from sign_language.config.schema import InferenceConfig
from sign_language.core.types import Prediction

logger = logging.getLogger(__name__)


class StabilizerState(Enum):
    """Operational states of the prediction stabilizer state machine."""
    IDLE = "idle"
    ACCUMULATING = "accumulating"
    COMMITTED = "committed"
    COOLDOWN = "cooldown"
    AWAITING_NEUTRAL = "awaiting_neutral"


class PredictionStabilizer:
    """Filters noisy per-frame predictions into discrete, intentional word commits.

    Implements four sequential defense guards:
        1. Confidence Threshold Gate: Low-confidence frames are discarded.
        2. Rolling-Window Voting: 7 of the last 10 votes must agree on the sign.
        3. Debounce Cooldown: Minimum duration before the next commit is allowed.
        4. Neutral Gate: The hand must return to neutral before repeating the same sign.
    """

    def __init__(self, inference_config: InferenceConfig, neutral_class: str = "_NEUTRAL") -> None:
        self._cfg = inference_config
        self._stab_cfg = inference_config.stabilisation
        self._neutral_class = neutral_class

        self._window: deque[str] = deque(maxlen=self._stab_cfg.window_size)
        self._state = StabilizerState.IDLE
        self._last_commit_time: float = -100.0
        self._last_seen_time: float = 0.0
        self._last_committed_label: str | None = None
        self._progress: float = 0.0

    def reset(self) -> None:
        """Reset stabilizer internal buffer and state machine to IDLE."""
        self._window.clear()
        self._state = StabilizerState.IDLE
        self._last_committed_label = None
        self._progress = 0.0
        self._last_commit_time = -100.0

    @property
    def state(self) -> StabilizerState:
        """Return current state name."""
        return self._state

    @property
    def progress(self) -> float:
        """Return progress fraction toward commit in [0.0, 1.0]."""
        return self._progress

    def update(self, prediction: Prediction | None, now: float) -> str | None:
        """Feed a single frame prediction into the stabilizer state machine.

        Args:
            prediction: Prediction object from model, or None if no hand detected.
            now: Current monotonic timestamp in seconds.

        Returns:
            str | None: The committed sign label if a commit occurred at this exact frame, else None.
        """
        # Guard 1: Stale buffer ageing when hand is absent or low confidence
        if (
            prediction is None
            or prediction.confidence < self._cfg.confidence_threshold
        ):
            if now - self._last_seen_time > self._stab_cfg.max_buffer_age_seconds:
                self._window.clear()
                self._progress = 0.0
                if self._state not in (StabilizerState.COOLDOWN, StabilizerState.AWAITING_NEUTRAL):
                    self._state = StabilizerState.IDLE
            return None

        self._last_seen_time = now

        # Add valid prediction to rolling window
        self._window.append(prediction.label)

        # Count votes in the rolling window
        counts = Counter(self._window)
        top_label, top_votes = counts.most_common(1)[0]
        self._progress = min(1.0, float(top_votes) / float(self._stab_cfg.min_votes_in_window))

        # Check if cooldown is active
        in_cooldown = (now - self._last_commit_time) < self._stab_cfg.cooldown_seconds
        if in_cooldown:
            self._state = StabilizerState.COOLDOWN
            return None

        # Check if vote threshold is satisfied (e.g. 7 out of 10 votes)
        if top_votes >= self._stab_cfg.min_votes_in_window:
            # Rejection / Neutral handling
            if top_label == self._neutral_class:
                # Hand passed through neutral: clear repeat gate
                self._last_committed_label = None
                self._window.clear()
                self._state = StabilizerState.IDLE
                self._progress = 0.0
                return None

            # Real sign candidate: check repeat gate
            if (
                self._stab_cfg.require_neutral_between_repeats
                and top_label == self._last_committed_label
            ):
                self._state = StabilizerState.AWAITING_NEUTRAL
                return None

            # COMMIT EVENT!
            self._last_committed_label = top_label
            self._last_commit_time = now
            self._window.clear()
            self._state = StabilizerState.COMMITTED
            self._progress = 1.0

            logger.info("Stabilizer COMMITTED sign: '%s' at %.2fs", top_label, now)
            return top_label

        # Accumulating votes
        self._state = StabilizerState.ACCUMULATING
        return None
