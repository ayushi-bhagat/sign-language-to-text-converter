"""Unit tests for SignPredictor."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from sign_language.config.loader import get_config
from sign_language.inference.predictor import SignPredictor
from sign_language.utils.exceptions import FeatureVersionMismatchError, ModelLoadError, PredictionError


@pytest.mark.unit
def test_predictor_unloaded_raises_error() -> None:
    """Predicting before loading artifacts raises PredictionError."""
    config = get_config()
    predictor = SignPredictor(config)
    assert not predictor.is_loaded

    with pytest.raises(PredictionError, match="not loaded"):
        predictor.predict(np.zeros(63, dtype=np.float32))


@pytest.mark.unit
def test_predictor_missing_dir_raises_load_error() -> None:
    """Loading from a non-existent directory raises ModelLoadError."""
    config = get_config()
    predictor = SignPredictor(config)

    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(ModelLoadError, match="Missing model artifacts"):
            predictor.load(Path(empty_dir))


@pytest.mark.unit
def test_predictor_predict_valid_shape() -> None:
    """Verify that predictor loads current trained model and predicts with latency."""
    config = get_config()
    predictor = SignPredictor(config)
    predictor.load()
    assert predictor.is_loaded
    assert len(predictor.classes) >= 2

    # Valid 63-dim input
    dummy_feat = np.zeros(63, dtype=np.float32)
    pred = predictor.predict(dummy_feat)
    assert pred.label in predictor.classes
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.latency_ms >= 0.0
    assert len(pred.probabilities) == len(predictor.classes)

    # Invalid dimension input
    with pytest.raises(PredictionError, match="Invalid feature vector shape"):
        predictor.predict(np.zeros(42, dtype=np.float32))
