"""Modern CustomTkinter desktop interface for real-time sign language recognition."""

from __future__ import annotations

import logging
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import customtkinter as ctk
except ImportError:
    ctk = None

from sign_language.config.schema import AppConfig
from sign_language.core.types import FrameResult, HandLandmarks
from sign_language.inference.pipeline import RealtimePipeline
from sign_language.ui.theme import Theme

logger = logging.getLogger(__name__)


class DesktopApp:
    """Primary desktop user interface built with CustomTkinter.

    Args:
        config: Central application configuration.
        pipeline: RealtimePipeline instance.
    """

    def __init__(self, config: AppConfig, pipeline: RealtimePipeline | None = None) -> None:
        if ctk is None:
            raise ImportError(
                "CustomTkinter is not installed. Run 'pip install customtkinter' "
                "or run scripts/run_realtime.py for OpenCV HUD mode."
            )

        self._cfg = config
        self._theme = Theme(config.ui)
        self._pipeline = pipeline if pipeline is not None else RealtimePipeline(config)

        # Set CustomTkinter appearance
        ctk.set_appearance_mode(config.ui.theme)
        ctk.set_default_color_theme("blue")

        self._root = ctk.CTk()
        self._root.title(config.ui.window_title)
        self._root.geometry(f"{config.ui.window_width}x{config.ui.window_height}")
        self._root.minsize(1000, 640)
        self._root.configure(fg_color=self._theme.bg_hex)

        self._is_running = False
        self._current_photo: ImageTk.PhotoImage | None = None  # Reference to prevent GC

        self._build_ui()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        """Construct the four-quadrant screen layout."""
        # ── 1. Top Header Bar ──────────────────────────────────────────────
        self.header = ctk.CTkFrame(self._root, height=54, fg_color=self._theme.surface_hex, corner_radius=0)
        self.header.pack(side="top", fill="x", padx=0, pady=0)

        self.title_label = ctk.CTkLabel(
            self.header,
            text=self._cfg.project.name.upper(),
            font=ctk.CTkFont(family=self._theme.font_family, size=16, weight="bold"),
            text_color=self._theme.text_primary_hex,
        )
        self.title_label.pack(side="left", padx=20, pady=12)

        self.fps_label = ctk.CTkLabel(
            self.header,
            text="FPS: --",
            font=ctk.CTkFont(family=self._theme.font_family, size=13),
            text_color=self._theme.success_hex,
        )
        self.fps_label.pack(side="right", padx=20, pady=12)

        model_status_text = f"● Model {self._cfg.model.version} loaded" if self._pipeline.predictor.is_loaded else "○ No Model Loaded"
        model_status_color = self._theme.success_hex if self._pipeline.predictor.is_loaded else self._theme.warning_hex
        self.model_status_label = ctk.CTkLabel(
            self.header,
            text=model_status_text,
            font=ctk.CTkFont(family=self._theme.font_family, size=13),
            text_color=model_status_color,
        )
        self.model_status_label.pack(side="right", padx=15, pady=12)

        # ── Middle Work Area: Video Feed (Left) & Telemetry Panel (Right) ──
        self.main_content = ctk.CTkFrame(self._root, fg_color="transparent")
        self.main_content.pack(side="top", fill="both", expand=True, padx=16, pady=12)

        # Left: Video Card
        self.video_card = ctk.CTkFrame(self.main_content, fg_color=self._theme.surface_hex, corner_radius=10)
        self.video_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_label = ctk.CTkLabel(self.video_card, text="Camera stopped\nClick 'Start Camera' below")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Right: Prediction & Controls Card
        self.side_panel = ctk.CTkFrame(self.main_content, width=380, fg_color=self._theme.surface_hex, corner_radius=10)
        self.side_panel.pack(side="right", fill="y", padx=(10, 0))
        self.side_panel.pack_propagate(False)

        # Prediction display
        self.pred_header = ctk.CTkLabel(
            self.side_panel,
            text="RECOGNISED SIGN",
            font=ctk.CTkFont(family=self._theme.font_family, size=11, weight="bold"),
            text_color=self._theme.text_muted_hex,
        )
        self.pred_header.pack(anchor="w", padx=20, pady=(20, 5))

        self.pred_label = ctk.CTkLabel(
            self.side_panel,
            text="--",
            font=ctk.CTkFont(family=self._theme.font_family, size=40, weight="bold"),
            text_color=self._theme.accent_hex,
        )
        self.pred_label.pack(anchor="w", padx=20, pady=(0, 15))

        # Confidence bar
        self.conf_header = ctk.CTkLabel(
            self.side_panel,
            text="CONFIDENCE",
            font=ctk.CTkFont(family=self._theme.font_family, size=11, weight="bold"),
            text_color=self._theme.text_muted_hex,
        )
        self.conf_header.pack(anchor="w", padx=20, pady=(5, 5))

        self.conf_bar = ctk.CTkProgressBar(self.side_panel, height=12, corner_radius=6)
        self.conf_bar.pack(fill="x", padx=20, pady=(0, 5))
        self.conf_bar.set(0.0)

        self.conf_text = ctk.CTkLabel(
            self.side_panel,
            text="0%",
            font=ctk.CTkFont(family=self._theme.font_family, size=12),
            text_color=self._theme.text_muted_hex,
        )
        self.conf_text.pack(anchor="e", padx=20, pady=(0, 15))

        # Stabilizer progress
        self.stab_header = ctk.CTkLabel(
            self.side_panel,
            text="STABILISATION PROGRESS",
            font=ctk.CTkFont(family=self._theme.font_family, size=11, weight="bold"),
            text_color=self._theme.text_muted_hex,
        )
        self.stab_header.pack(anchor="w", padx=20, pady=(5, 5))

        self.stab_bar = ctk.CTkProgressBar(
            self.side_panel,
            height=8,
            corner_radius=4,
            progress_color=self._theme.success_hex,
        )
        self.stab_bar.pack(fill="x", padx=20, pady=(0, 5))
        self.stab_bar.set(0.0)

        self.stab_status = ctk.CTkLabel(
            self.side_panel,
            text="State: Idle",
            font=ctk.CTkFont(family=self._theme.font_family, size=12),
            text_color=self._theme.text_muted_hex,
        )
        self.stab_status.pack(anchor="w", padx=20, pady=(0, 25))

        # Control Buttons
        self.btn_toggle = ctk.CTkButton(
            self.side_panel,
            text="Start Camera",
            command=self._toggle_camera,
            font=ctk.CTkFont(family=self._theme.font_family, size=14, weight="bold"),
            fg_color=self._theme.accent_hex,
            height=38,
        )
        self.btn_toggle.pack(fill="x", padx=20, pady=6)

        self.btn_space = ctk.CTkButton(
            self.side_panel,
            text="Space",
            command=self._on_space,
            fg_color="transparent",
            border_width=1,
            border_color=self._theme.text_muted_hex,
            height=32,
        )
        self.btn_space.pack(fill="x", padx=20, pady=4)

        self.btn_backspace = ctk.CTkButton(
            self.side_panel,
            text="Backspace Word",
            command=self._on_backspace,
            fg_color="transparent",
            border_width=1,
            border_color=self._theme.text_muted_hex,
            height=32,
        )
        self.btn_backspace.pack(fill="x", padx=20, pady=4)

        self.btn_clear = ctk.CTkButton(
            self.side_panel,
            text="Clear All",
            command=self._on_clear,
            fg_color="transparent",
            text_color=self._theme.danger_hex,
            border_width=1,
            border_color=self._theme.danger_hex,
            height=32,
        )
        self.btn_clear.pack(fill="x", padx=20, pady=4)

        # ── 3. Bottom Text Box ─────────────────────────────────────────────
        self.bottom_panel = ctk.CTkFrame(self._root, height=130, fg_color=self._theme.surface_hex, corner_radius=10)
        self.bottom_panel.pack(side="bottom", fill="x", padx=16, pady=(0, 16))
        self.bottom_panel.pack_propagate(False)

        self.text_header = ctk.CTkLabel(
            self.bottom_panel,
            text="COMPOSED TEXT",
            font=ctk.CTkFont(family=self._theme.font_family, size=11, weight="bold"),
            text_color=self._theme.text_muted_hex,
        )
        self.text_header.pack(anchor="w", padx=20, pady=(10, 2))

        self.text_display = ctk.CTkLabel(
            self.bottom_panel,
            text="[Composed text will appear here]",
            font=ctk.CTkFont(family=self._theme.font_family, size=22),
            text_color=self._theme.text_muted_hex,
            anchor="w",
            wraplength=1100,
        )
        self.text_display.pack(anchor="w", fill="x", padx=20, pady=(0, 10))

    def _toggle_camera(self) -> None:
        """Start or stop live camera stream."""
        if not self._is_running:
            try:
                self._pipeline.start()
                self._is_running = True
                self.btn_toggle.configure(text="Stop Camera", fg_color=self._theme.danger_hex)
                self._schedule_next_frame()
            except Exception as exc:
                logger.error("Could not start camera: %s", exc)
                self.video_label.configure(text=f"Failed to open camera:\n{exc}")
        else:
            self._is_running = False
            self._pipeline.stop()
            self.btn_toggle.configure(text="Start Camera", fg_color=self._theme.accent_hex)
            self.video_label.configure(text="Camera stopped\nClick 'Start Camera' below", image=None)
            self.fps_label.configure(text="FPS: --")

    def _schedule_next_frame(self) -> None:
        """Process one frame and schedule next tick on Tkinter main thread."""
        if not self._is_running:
            return

        try:
            result: FrameResult = self._pipeline.step()
            self._render_frame(result)
        except Exception as exc:
            logger.error("Error during pipeline step: %s", exc)

        # Schedule next tick at ~30 FPS (every 25-30ms)
        self._root.after(25, self._schedule_next_frame)

    def _render_frame(self, result: FrameResult) -> None:
        """Update video label and telemetry indicators from FrameResult."""
        frame = cv2.flip(result.frame_bgr, 1)

        # Draw skeleton on mirrored frame
        if self._cfg.ui.show_landmarks and result.landmarks is not None:
            pts_disp = result.landmarks.points.copy()
            pts_disp[:, 0] = 1.0 - pts_disp[:, 0]
            disp_lm = HandLandmarks(
                points=pts_disp,
                handedness=result.landmarks.handedness,
                detection_confidence=result.landmarks.detection_confidence,
            )
            frame = self._pipeline.detector.draw_landmarks(frame, disp_lm, color_scheme="accent")

        # Convert to PIL Image and resize to fit video container
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Resize keeping aspect ratio
        target_w = self._cfg.ui.video_target_width
        aspect = frame.shape[0] / frame.shape[1]
        target_h = int(target_w * aspect)
        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        self._current_photo = ImageTk.PhotoImage(image=pil_img)
        self.video_label.configure(image=self._current_photo, text="")

        # Update FPS
        self.fps_label.configure(text=f"FPS: {result.fps:4.1f}")

        # Update Prediction & Confidence
        if result.prediction is not None:
            self.pred_label.configure(text=result.prediction.label)
            conf_pct = result.prediction.confidence
            self.conf_bar.set(conf_pct)
            self.conf_text.configure(text=f"{conf_pct * 100:.1f}%")
        else:
            self.pred_label.configure(text="--")
            self.conf_bar.set(0.0)
            self.conf_text.configure(text="0%")

        # Update Stabilizer status & progress
        self.stab_bar.set(result.stabilizer_progress)
        self.stab_status.configure(text=f"State: {result.stabilizer_state.capitalize()}")

        # Update Composed Text
        if result.current_text:
            self.text_display.configure(text=result.current_text, text_color=self._theme.text_primary_hex)
        else:
            self.text_display.configure(text="[Composed text will appear here]", text_color=self._theme.text_muted_hex)

    def _on_space(self) -> None:
        self._pipeline.composer.add_space()
        self.text_display.configure(text=self._pipeline.composer.formatted)

    def _on_backspace(self) -> None:
        self._pipeline.composer.backspace()
        self.text_display.configure(text=self._pipeline.composer.formatted or "[Composed text will appear here]")

    def _on_clear(self) -> None:
        self._pipeline.composer.clear()
        self.text_display.configure(text="[Composed text will appear here]", text_color=self._theme.text_muted_hex)

    def _on_close(self) -> None:
        """Deterministically release hardware resources on window exit."""
        self._is_running = False
        try:
            self._pipeline.stop()
        except Exception:
            pass
        self._root.destroy()

    def run(self) -> None:
        """Start the CustomTkinter UI main loop."""
        self._root.mainloop()
