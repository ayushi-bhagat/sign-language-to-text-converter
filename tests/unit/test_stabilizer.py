"""Unit tests for the prediction stabilizer state machine."""

import numpy as np
import pytest

from sign_language.config.schema import InferenceConfig, StabilisationConfig
from sign_language.core.types import Prediction
from sign_language.inference.stabilizer import PredictionStabilizer, StabilizerState


@pytest.fixture
def stabilizer_config() -> InferenceConfig:
    stab_cfg = StabilisationConfig(
        enabled=True,
        window_size=10,
        min_votes_in_window=7,
        smoothing="majority_vote",
        ema_alpha=0.4,
        cooldown_seconds=1.2,
        require_neutral_between_repeats=True,
        max_buffer_age_seconds=2.0,
    )
    return InferenceConfig(
        confidence_threshold=0.80,
        stabilisation=stab_cfg,
    )


def _make_pred(label: str, conf: float = 0.9) -> Prediction:
    return Prediction(
        label=label,
        confidence=conf,
        probabilities=np.array([conf]),
        latency_ms=1.0,
    )


@pytest.mark.unit
def test_stabilizer_commits_at_threshold(stabilizer_config: InferenceConfig) -> None:
    """Verify that 7 confident frames out of 10 triggers exactly one commit."""
    stabilizer = PredictionStabilizer(stabilizer_config)
    commits = []

    # Stream 10 frames of HELLO at 0.9 confidence
    for i in range(10):
        t = 1.0 + (i * 0.05)  # 50 ms apart
        res = stabilizer.update(_make_pred("HELLO", 0.9), now=t)
        if res is not None:
            commits.append(res)
            # Right at commit moment, state is COMMITTED
            assert stabilizer.state == StabilizerState.COMMITTED

    assert len(commits) == 1
    assert commits[0] == "HELLO"
    # After commit, while frames arrive within 1.2s cooldown, state is COOLDOWN
    assert stabilizer.state == StabilizerState.COOLDOWN


@pytest.mark.unit
def test_stabilizer_votes_with_noisy_frame(stabilizer_config: InferenceConfig) -> None:
    """Verify that a single dropped or noisy frame does not reset voting progress."""
    stabilizer = PredictionStabilizer(stabilizer_config)
    commits = []

    # 3x HELLO, 1x NO (noise), 4x HELLO = 7 HELLO votes in window of 8 frames
    stream = ["HELLO"] * 3 + ["NO"] + ["HELLO"] * 4
    for i, label in enumerate(stream):
        t = 1.0 + (i * 0.05)
        res = stabilizer.update(_make_pred(label, 0.9), now=t)
        if res is not None:
            commits.append(res)

    # Must still commit HELLO because window-based count reached 7!
    assert len(commits) == 1
    assert commits[0] == "HELLO"


@pytest.mark.unit
def test_stabilizer_low_confidence_discarded(stabilizer_config: InferenceConfig) -> None:
    """Verify predictions below confidence_threshold (0.80) are discarded."""
    stabilizer = PredictionStabilizer(stabilizer_config)
    commits = []

    for i in range(15):
        t = 1.0 + (i * 0.05)
        res = stabilizer.update(_make_pred("HELLO", 0.65), now=t)
        if res is not None:
            commits.append(res)

    assert len(commits) == 0
    assert stabilizer.progress == 0.0


@pytest.mark.unit
def test_stabilizer_repeat_gate_requires_neutral(stabilizer_config: InferenceConfig) -> None:
    """Verify that repeating the same sign requires passing through neutral."""
    stabilizer = PredictionStabilizer(stabilizer_config)
    commits = []

    # 1. First commit of HELLO
    for i in range(7):
        t = 1.0 + (i * 0.05)
        res = stabilizer.update(_make_pred("HELLO", 0.95), now=t)
        if res is not None:
            commits.append(res)
    assert len(commits) == 1

    # 2. Continue holding HELLO for 20 frames past cooldown
    # Cooldown is 1.2s, let's advance past 3.0s
    for i in range(15):
        t = 3.0 + (i * 0.05)
        res = stabilizer.update(_make_pred("HELLO", 0.95), now=t)
        if res is not None:
            commits.append(res)

    # Should NOT commit a second HELLO because hand did not pass through neutral
    assert len(commits) == 1
    assert stabilizer.state == StabilizerState.AWAITING_NEUTRAL

    # 3. Pass through _NEUTRAL for 7 frames
    for i in range(7):
        t = 5.0 + (i * 0.05)
        res = stabilizer.update(_make_pred("_NEUTRAL", 0.95), now=t)
        # _NEUTRAL should never commit text
        assert res is None

    # 4. Now sign HELLO again past cooldown
    for i in range(7):
        t = 7.0 + (i * 0.05)
        res = stabilizer.update(_make_pred("HELLO", 0.95), now=t)
        if res is not None:
            commits.append(res)

    # Now second HELLO successfully committed!
    assert len(commits) == 2
    assert commits == ["HELLO", "HELLO"]
