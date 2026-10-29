"""Real-time sign recognition pipeline orchestrating capture, detection, extraction, prediction, stabilization, and text composition."""

from __future__ import annotations

import logging
import time
from sign_language.config.schema import AppConfig
from sign_language.core.camera import CameraStream
from sign_language.core.hand_detector import HandDetector
from sign_language.core.types import FrameResult, HandLandmarks, Prediction
from sign_language.features.extractor import LandmarkExtractor
from sign_language.inference.predictor import SignPredictor
from sign_language.inference.stabilizer import PredictionStabilizer
from sign_language.text.composer import TextComposer
from sign_language.utils.exceptions import InferenceError

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """Wires end-to-end hardware capture, computer vision, ML inference, and text buffers.

    Args:
        config: Central application configuration.
        predictor: Pre-initialized (or loaded) SignPredictor instance.
    """

    def __init__(self, config: AppConfig, predictor: SignPredictor | None = None) -> None:
        self._cfg = config

        cam_cfg = config.camera
        self._camera = CameraStream(
            index=cam_cfg.index,
            width=cam_cfg.width,
            height=cam_cfg.height,
            target_fps=cam_cfg.target_fps,
            backend=cam_cfg.backend,
            warmup_frames=cam_cfg.warmup_frames,
            read_timeout_ms=cam_cfg.read_timeout_ms,
            flip_horizontal=cam_cfg.flip_horizontal,
        )

        det_cfg = config.hand_detection
        self._detector = HandDetector(
            max_num_hands=det_cfg.max_num_hands,
            min_detection_confidence=det_cfg.min_detection_confidence,
            min_tracking_confidence=det_cfg.min_tracking_confidence,
            model_complexity=det_cfg.model_complexity,
            static_image_mode=det_cfg.static_image_mode,
            multi_hand_policy=det_cfg.multi_hand_policy,
        )

        self._extractor = LandmarkExtractor(config.features)

        if predictor is not None:
            self._predictor = predictor
        else:
            self._predictor = SignPredictor(config)
            try:
                self._predictor.load()
            except Exception as exc:
                logger.warning("SignPredictor could not load model at startup: %s", exc)

        self._stabilizer = PredictionStabilizer(
            inference_config=config.inference,
            neutral_class=config.dataset.neutral_class,
        )
        self._composer = TextComposer(
            config=config.text,
            neutral_class=config.dataset.neutral_class,
        )

        self._is_running = False

    @property
    def predictor(self) -> SignPredictor:
        return self._predictor

    @property
    def composer(self) -> TextComposer:
        return self._composer

    @property
    def camera(self) -> CameraStream:
        return self._camera

    @property
    def detector(self) -> HandDetector:
        return self._detector

    def start(self) -> None:
        """Open camera and prepare runtime pipeline."""
        if self._is_running:
            return
        self._camera.open()
        self._is_running = True
        logger.info("RealtimePipeline started.")

    def step(self) -> FrameResult:
        """Process a single frame through the complete pipeline.

        Returns:
            FrameResult: Rich output payload consumed by user interfaces.
        """
        if not self._is_running or not self._camera.is_open():
            raise InferenceError("Pipeline is not running. Call start() before step().")

        now = time.monotonic()
        frame_raw = self._camera.read(flip=False)
        warnings: list[str] = []

        all_hands = self._detector.detect_all(frame_raw)
        if len(all_hands) > 1:
            warnings.append(
                f"Multiple hands detected ({len(all_hands)}) — single-hand vocabulary active"
            )

        primary_hand = self._detector.detect(frame_raw)
        prediction: Prediction | None = None
        committed_sign: str | None = None

        if primary_hand is not None:
            if self._predictor.is_loaded:
                feature_vec = self._extractor.extract(primary_hand)
                prediction = self._predictor.predict(feature_vec)
                committed_sign = self._stabilizer.update(prediction, now=now)
            else:
                warnings.append("No trained model loaded — predictions inactive")
                self._stabilizer.update(None, now=now)
        else:
            # Hand absent
            self._stabilizer.update(None, now=now)

        if committed_sign is not None:
            self._composer.append(committed_sign)

        return FrameResult(
            frame_bgr=frame_raw,
            landmarks=primary_hand,
            prediction=prediction,
            committed_label=committed_sign,
            current_text=self._composer.formatted,
            fps=self._camera.measured_fps,
            stabilizer_state=self._stabilizer.state.value,
            stabilizer_progress=self._stabilizer.progress,
            warnings=warnings,
        )

    def stop(self) -> None:
        """Release camera and MediaPipe resources deterministically."""
        if not self._is_running:
            return
        self._camera.release()
        self._detector.close()
        self._is_running = False
        logger.info("RealtimePipeline stopped.")

    def __enter__(self) -> "RealtimePipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
