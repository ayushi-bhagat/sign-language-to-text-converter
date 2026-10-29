"""Interactive data collection script.

Usage
-----
Collect samples for all 13 classes (starting from the first incomplete class):

    .venv\\Scripts\\python.exe scripts\\collect_data.py

Collect samples for a specific class only:

    .venv\\Scripts\\python.exe scripts\\collect_data.py --label HELLO

Show dataset status without collecting:

    .venv\\Scripts\\python.exe scripts\\collect_data.py --status

Workflow per class
------------------
1. A live camera window opens.
2. A 3-second countdown appears — get your hand into frame.
3. Hold the sign. Frames are captured at 8 fps (not 30 fps) to reduce redundancy.
4. The validator silently rejects frames that are:
   - Too close or too far (hand bbox fraction out of range)
   - Out of coordinate bounds (hand partly off-screen)
   - Nearly identical to the previous accepted frame (cosine similarity >= 0.9995)
5. The progress bar fills as accepted samples accumulate.
6. When the session cap (90 samples) is hit, or you press Q/ESC, the session ends.
7. Samples are saved immediately to data/features/.
8. You are prompted to collect another session or move to the next class.

Tips for good data quality
--------------------------
- Vary your hand slightly between frames (small rotations, different distances)
- Collect 4+ separate sessions per class on different days / lighting conditions
- The more varied your sessions, the better the trained model generalises
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

# Ensure src/ is on sys.path when running as a standalone script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Reconfigure stdout/stderr for UTF-8 on Windows terminals if supported
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from sign_language.config.loader import get_config
from sign_language.core.camera import CameraStream
from sign_language.core.hand_detector import HandDetector
from sign_language.data_collection.collector import SessionCollector
from sign_language.data_collection.persistence import DatasetReader, DatasetWriter
from sign_language.data_collection.validator import FrameValidator
from sign_language.features.extractor import LandmarkExtractor
from sign_language.utils.logging_setup import configure_logging
from sign_language.utils.paths import ensure_dir, resolve
from sign_language.utils.seeding import set_global_seed


def _print_status(cfg, reader: DatasetReader) -> None:
    """Print a table showing current sample and session counts per class."""
    class_counts = reader.class_counts()
    session_counts = reader.session_counts()

    target = cfg.dataset.samples_per_class_target
    minimum = cfg.dataset.samples_per_class_minimum
    session_min = cfg.dataset.sessions_per_class_minimum

    print("\n" + "=" * 65)
    print(f"  DATASET STATUS   target={target}  minimum={minimum}  sessions≥{session_min}")
    print("=" * 65)
    print(f"  {'Class':<16} {'Samples':>8} {'Sessions':>9} {'Progress':>12}  Status")
    print("  " + "-" * 61)

    for label in cfg.dataset.classes:
        n_samples = class_counts.get(label, 0)
        n_sessions = len(session_counts.get(label, {}))
        pct = min(100.0, 100.0 * n_samples / target) if target > 0 else 0.0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))

        if n_samples >= target and n_sessions >= session_min:
            status = "✅ COMPLETE"
        elif n_samples >= minimum and n_sessions >= session_min:
            status = "⚠️  AT MINIMUM"
        elif n_samples > 0:
            status = "🔄 IN PROGRESS"
        else:
            status = "⬜ NOT STARTED"

        print(f"  {label:<16} {n_samples:>8} {n_sessions:>9}  {bar}  {status}")

    total = sum(class_counts.values())
    grand_target = target * len(cfg.dataset.classes)
    print("  " + "-" * 61)
    print(f"  {'TOTAL':<16} {total:>8}{'':>9}   (target: {grand_target})")
    print("=" * 65 + "\n")


def _collect_for_label(label: str, cfg, collector: SessionCollector, writer: DatasetWriter) -> int:
    """Run one or more sessions for a given label. Returns total new samples."""
    total_new = 0

    while True:
        print(f"\n  ─── Collecting: {label} ───")
        print("  Press Q or ESC during capture to stop early.")
        input("  Press ENTER to start the countdown, or Ctrl+C to skip this class...")

        samples = collector.collect_session(label)
        n = writer.write_samples(samples)
        total_new += n
        print(f"  ✅ Session complete: {n} new samples saved (total this run: {total_new})")

        another = input("  Collect another session for this class? [y/N] ").strip().lower()
        if another != "y":
            break

    return total_new


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive data collection for Sign Language to Text"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Collect only this class label (default: all incomplete classes)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print dataset status and exit without collecting",
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)
    cfg = get_config()
    set_global_seed(cfg.reproducibility.random_seed)

    features_dir = resolve(cfg.paths.data_features)
    ensure_dir(features_dir)
    ensure_dir(resolve(cfg.paths.data_raw))

    reader = DatasetReader(features_dir)
    writer = DatasetWriter(features_dir)

    # ── Status only ──────────────────────────────────────────────────────────
    if args.status:
        _print_status(cfg, reader)
        return

    # ── Determine which labels to collect ────────────────────────────────────
    if args.label:
        if args.label not in cfg.dataset.classes:
            print(f"ERROR: '{args.label}' is not in the configured vocabulary.")
            print(f"Valid labels: {cfg.dataset.classes}")
            sys.exit(1)
        labels_to_collect = [args.label]
    else:
        # Collect all classes that haven't reached the target yet
        class_counts = reader.class_counts()
        target = cfg.dataset.samples_per_class_target
        labels_to_collect = [
            label for label in cfg.dataset.classes
            if class_counts.get(label, 0) < target
        ]

    if not labels_to_collect:
        print("\n✅ All classes have reached the target sample count!")
        _print_status(cfg, reader)
        return

    print(f"\n📋 Classes to collect: {labels_to_collect}")
    _print_status(cfg, reader)

    # ── Open hardware (camera + detector) once for the whole run ─────────────
    extractor = LandmarkExtractor(cfg.features)
    validator = FrameValidator(cfg.dataset.validation)

    cam_cfg = cfg.camera
    det_cfg = cfg.hand_detection

    with (
        CameraStream(
            index=cam_cfg.index,
            width=cam_cfg.width,
            height=cam_cfg.height,
            target_fps=cam_cfg.target_fps,
            backend=cam_cfg.backend,
            warmup_frames=cam_cfg.warmup_frames,
            read_timeout_ms=cam_cfg.read_timeout_ms,
        ) as camera,
        HandDetector(
            max_num_hands=det_cfg.max_num_hands,
            min_detection_confidence=det_cfg.min_detection_confidence,
            min_tracking_confidence=det_cfg.min_tracking_confidence,
            model_complexity=det_cfg.model_complexity,
            static_image_mode=det_cfg.static_image_mode,
            multi_hand_policy=det_cfg.multi_hand_policy,
        ) as detector,
    ):
        collector = SessionCollector(
            config=cfg,
            extractor=extractor,
            validator=validator,
            camera=camera,
            detector=detector,
        )

        total_collected = 0
        for label in labels_to_collect:
            try:
                n = _collect_for_label(label, cfg, collector, writer)
                total_collected += n
                print(f"\n  Current dataset status after collecting {label}:")
                _print_status(cfg, reader)
            except KeyboardInterrupt:
                print(f"\n  Interrupted. Saved {total_collected} samples so far.")
                break

    cv2.destroyAllWindows()
    print(f"\n✅ Collection run complete. Total new samples this run: {total_collected}")
    _print_status(cfg, reader)


if __name__ == "__main__":
    main()
