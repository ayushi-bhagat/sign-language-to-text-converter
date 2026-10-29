"""Integration test: Train/Serve feature parity.

Guarantees that offline training and real-time inference use the exact same feature
extraction code and produce bit-identical feature vectors for any HandLandmarks input.
"""

import numpy as np
import pytest

from sign_language.config.loader import get_config
from sign_language.core.types import HandLandmarks
from sign_language.features.extractor import LandmarkExtractor


@pytest.mark.integration
def test_train_serve_feature_identity() -> None:
    """Verify that feature vectors extracted offline and online are bit-identical."""
    cfg = get_config()

    # Create two independent extractor instances (simulating training and inference workers)
    offline_extractor = LandmarkExtractor(cfg.features)
    online_extractor = LandmarkExtractor(cfg.features)

    # Deterministic test landmarks
    rng = np.random.default_rng(12345)
    test_points = rng.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)

    landmarks_right = HandLandmarks(
        points=test_points,
        handedness="Right",
        detection_confidence=0.92,
    )

    landmarks_left = HandLandmarks(
        points=test_points,
        handedness="Left",
        detection_confidence=0.88,
    )

    # 1. Right hand parity
    vec_offline_r = offline_extractor.extract(landmarks_right)
    vec_online_r = online_extractor.extract(landmarks_right)
    np.testing.assert_array_equal(
        vec_offline_r,
        vec_online_r,
        err_msg="Train/Serve skew detected on Right hand features!",
    )

    # 2. Left hand parity
    vec_offline_l = offline_extractor.extract(landmarks_left)
    vec_online_l = online_extractor.extract(landmarks_left)
    np.testing.assert_array_equal(
        vec_offline_l,
        vec_online_l,
        err_msg="Train/Serve skew detected on Left hand features!",
    )

    # 3. Shape and version
    assert vec_online_r.shape == (cfg.features.base_dim,)
    assert offline_extractor.feature_set_version == online_extractor.feature_set_version
