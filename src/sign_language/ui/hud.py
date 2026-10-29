"""OpenCV Head-Up Display (HUD) overlay renderer for real-time sign recognition."""

from __future__ import annotations

import cv2
import numpy as np

from sign_language.config.schema import UIConfig
from sign_language.core.hand_detector import HandDetector
from sign_language.core.types import FrameResult, HandLandmarks
from sign_language.ui.theme import Theme


class HUD:
    """Renders informative and accessible telemetry overlays onto video frames."""

    def __init__(self, ui_config: UIConfig, detector: HandDetector) -> None:
        self._cfg = ui_config
        self._theme = Theme(ui_config)
        self._detector = detector

    def draw(self, result: FrameResult) -> np.ndarray:
        """Render complete HUD overlay onto a mirrored display copy of the frame.

        Args:
            result: FrameResult containing image buffer, landmarks, predictions, and status.

        Returns:
            np.ndarray: Fully annotated BGR image buffer ready for display.
        """
        # Mirror frame horizontally for user comfort (selfie/mirror view)
        frame = cv2.flip(result.frame_bgr, 1)
        h, w = frame.shape[:2]

        # 1. Draw hand landmark skeleton if detected
        if self._cfg.show_landmarks and result.landmarks is not None:
            # Map raw-frame coordinates to mirrored display space (x_disp = 1.0 - x_raw)
            pts_disp = result.landmarks.points.copy()
            pts_disp[:, 0] = 1.0 - pts_disp[:, 0]
            disp_lm = HandLandmarks(
                points=pts_disp,
                handedness=result.landmarks.handedness,
                detection_confidence=result.landmarks.detection_confidence,
            )
            frame = self._detector.draw_landmarks(frame, disp_lm, color_scheme="accent")

        # 2. Top Status Bar
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 22, 28), -1)
        cv2.line(frame, (0, 65), (w, 65), (50, 54, 66), 1)

        # FPS counter
        if self._cfg.show_fps:
            fps_text = f"FPS: {result.fps:4.1f}"
            cv2.putText(
                frame,
                fps_text,
                (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                self._theme.success_bgr,
                2,
                cv2.LINE_AA,
            )

        # Prediction and Confidence
        if result.prediction is not None:
            label_text = f"Sign: {result.prediction.label}"
            conf_pct = result.prediction.confidence * 100.0
            conf_text = f"{conf_pct:4.1f}%"

            cv2.putText(
                frame,
                label_text,
                (200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                self._theme.text_primary_bgr,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                conf_text,
                (w - 110, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                self._theme.accent_bgr,
                2,
                cv2.LINE_AA,
            )

            # Mini confidence bar
            if self._cfg.show_confidence_bar:
                bar_x = w - 240
                bar_y = 26
                bar_w = 110
                bar_h = 16
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 54, 66), -1)
                fill_w = int(bar_w * min(1.0, result.prediction.confidence))
                fill_color = self._theme.success_bgr if result.prediction.confidence >= 0.8 else self._theme.warning_bgr
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
        else:
            cv2.putText(
                frame,
                "No hand in view",
                (200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                self._theme.text_muted_bgr,
                2,
                cv2.LINE_AA,
            )

        # 3. Active Warning Banner
        warning_y_offset = 65
        if result.warnings:
            for warning in result.warnings:
                cv2.rectangle(frame, (0, warning_y_offset), (w, warning_y_offset + 30), (25, 60, 200), -1)
                cv2.putText(
                    frame,
                    f"! {warning}",
                    (16, warning_y_offset + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                warning_y_offset += 30

        # 4. Stabilizer Commit Progress Bar (just above footer)
        prog_bar_y = h - 85
        cv2.rectangle(frame, (0, prog_bar_y), (w, prog_bar_y + 8), (35, 38, 48), -1)
        fill_prog = int(w * result.stabilizer_progress)
        if fill_prog > 0:
            prog_color = self._theme.success_bgr if result.stabilizer_progress >= 1.0 else self._theme.accent_bgr
            cv2.rectangle(frame, (0, prog_bar_y), (fill_prog, prog_bar_y + 8), prog_color, -1)

        # 5. Bottom Text Bar
        cv2.rectangle(frame, (0, h - 75), (w, h), (18, 20, 26), -1)
        cv2.line(frame, (0, h - 75), (w, h - 75), (50, 54, 66), 1)

        # Composed Text
        composed_text = result.current_text if result.current_text else "[Composed text will appear here]"
        text_color = self._theme.text_primary_bgr if result.current_text else self._theme.text_muted_bgr
        cv2.putText(
            frame,
            composed_text,
            (16, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            text_color,
            2,
            cv2.LINE_AA,
        )

        # Instructions hint (right footer)
        hint = "Q: quit | C: clear | B: backspace"
        tw, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0], None
        cv2.putText(
            frame,
            hint,
            (w - tw - 16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self._theme.text_muted_bgr,
            1,
            cv2.LINE_AA,
        )

        return frame
