"""Online sign classification predictor with feature version gating and latency tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any
import joblib
import numpy as np

from sign_language.config.schema import AppConfig
from sign_language.core.types import Prediction
from sign_language.features.registry import verify_feature_version
from sign_language.utils.exceptions import FeatureVersionMismatchError, ModelLoadError, PredictionError
from sign_language.utils.paths import resolve

logger = logging.getLogger(__name__)


class SignPredictor:
    """Loads trained Pipeline artifact and performs low-latency inference on feature vectors.

    Args:
        config: Central application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._models_dir = resolve(config.paths.models) / config.model.version
        self._pipeline = None
        self._label_encoder = None
        self._metadata: dict[str, Any] = {}
        self._classes: list[str] = []
        self._expected_dim = config.features.base_dim

    def load(self, model_dir: Path | str | None = None) -> None:
        """Load trained sklearn Pipeline, LabelEncoder, and metadata.

        Args:
            model_dir: Optional custom model directory. Defaults to models/{model.version}.

        Raises:
            ModelLoadError: If artifact files are missing or unreadable.
            FeatureVersionMismatchError: If trained feature version differs from current config.
        """
        dir_path = Path(model_dir) if model_dir else self._models_dir
        clf_file = dir_path / self._cfg.model.classifier_file
        le_file = dir_path / self._cfg.model.label_encoder_file
        meta_file = dir_path / self._cfg.model.metadata_file

        if not (clf_file.exists() and le_file.exists() and meta_file.exists()):
            raise ModelLoadError(
                f"Missing model artifacts in {dir_path}. Ensure {clf_file.name}, "
                f"{le_file.name}, and {meta_file.name} are present. Run scripts/train_model.py."
            )

        try:
            self._pipeline = joblib.load(str(clf_file))
            self._label_encoder = joblib.load(str(le_file))
            with meta_file.open("r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model artifacts from {dir_path}: {exc}") from exc

        # Feature version match enforcement
        if self._cfg.model.enforce_feature_version_match:
            loaded_ver = self._metadata.get("feature_set_version", "")
            current_ver = self._cfg.features.feature_set_version
            verify_feature_version(loaded_ver, current_ver)

        self._classes = list(self._label_encoder.classes_)
        logger.info(
            "SignPredictor loaded successfully: model_version=%s, winner=%s, classes=%s",
            self._metadata.get("model_version", "v1"),
            self._metadata.get("winner", "unknown"),
            self._classes,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if predictor has loaded valid artifacts."""
        return self._pipeline is not None and self._label_encoder is not None

    @property
    def classes(self) -> list[str]:
        """Return the list of class labels recognised by the model."""
        return list(self._classes)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model metadata dictionary."""
        return dict(self._metadata)

    def predict(self, feature_vector: np.ndarray) -> Prediction:
        """Predict class label and posterior probabilities for a single feature vector.

        Args:
            feature_vector: 1D float32 array of shape (dim,).

        Returns:
            Prediction: Predicted label, confidence, full probability distribution, and latency.

        Raises:
            PredictionError: If model is not loaded or feature shape is invalid.
        """
        if not self.is_loaded:
            raise PredictionError("SignPredictor is not loaded. Call load() before predict().")

        if feature_vector.ndim != 1 or len(feature_vector) != self._expected_dim:
            raise PredictionError(
                f"Invalid feature vector shape: expected (63,), got {feature_vector.shape}"
            )

        start = time.perf_counter()
        X_2d = feature_vector.reshape(1, -1)

        try:
            if hasattr(self._pipeline, "predict_proba"):
                probs = self._pipeline.predict_proba(X_2d)[0]
                best_idx = int(np.argmax(probs))
                confidence = float(probs[best_idx])
            else:
                pred_label = self._pipeline.predict(X_2d)[0]
                best_idx = int(pred_label)
                confidence = 1.0
                probs = np.zeros(len(self._classes))
                probs[best_idx] = 1.0

            label = self._label_encoder.inverse_transform([best_idx])[0]
        except Exception as exc:
            raise PredictionError(f"Model prediction failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return Prediction(
            label=str(label),
            confidence=confidence,
            probabilities=probs,
            latency_ms=elapsed_ms,
        )
