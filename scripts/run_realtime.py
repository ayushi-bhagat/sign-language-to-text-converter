"""Standalone real-time sign language recognition with OpenCV HUD.

Usage:
    .venv\\Scripts\\python.exe scripts\\run_realtime.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sign_language.config.loader import get_config
from sign_language.inference.pipeline import RealtimePipeline
from sign_language.ui.hud import HUD
from sign_language.utils.logging_setup import configure_logging


def main() -> None:
    configure_logging()
    logger = logging.getLogger("run_realtime")
    cfg = get_config()

    print("\n" + "=" * 65)
    print("  REAL-TIME SIGN LANGUAGE RECOGNITION (HUD MODE)")
    print("=" * 65)
    print(f"  Confidence Threshold : {cfg.inference.confidence_threshold * 100:.0f}%")
    print(f"  Stabilisation Window : {cfg.inference.stabilisation.min_votes_in_window} of {cfg.inference.stabilisation.window_size} votes")
    print(f"  Cooldown             : {cfg.inference.stabilisation.cooldown_seconds}s")
    print(f"  Neutral Repeat Gate  : {'Active' if cfg.inference.stabilisation.require_neutral_between_repeats else 'Disabled'}")
    print("-" * 65)
    print("  Controls:")
    print("    'Q' or ESC : Quit")
    print("    'C'        : Clear composed text")
    print("    'B'        : Backspace last word")
    print("    ' ' (Space): Insert space")
    print("=" * 65 + "\n")

    pipeline = RealtimePipeline(cfg)
    hud = HUD(cfg.ui, pipeline.detector)

    window_name = cfg.ui.window_title
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, cfg.ui.window_width, cfg.ui.window_height)

    try:
        pipeline.start()
        while True:
            result = pipeline.step()
            annotated_frame = hud.draw(result)

            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            elif key in (ord("c"), ord("C")):
                pipeline.composer.clear()
            elif key in (ord("b"), ord("B"), 8):  # B or Backspace key
                pipeline.composer.backspace()
            elif key == ord(" "):
                pipeline.composer.add_space()

    except KeyboardInterrupt:
        logger.info("Real-time loop interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\nPipeline stopped and camera released cleanly.\n")


if __name__ == "__main__":
    main()
