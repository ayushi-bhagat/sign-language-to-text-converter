"""Application entry point for Sign Language to Text conversion.

Usage:
    # Run primary desktop application:
    .venv\\Scripts\\python.exe app.py

    # Run lightweight OpenCV HUD mode:
    .venv\\Scripts\\python.exe scripts\\run_realtime.py
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent
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
from sign_language.utils.logging_setup import configure_logging
from sign_language.utils.seeding import set_global_seed


def main() -> None:
    configure_logging()
    logger = logging.getLogger("app")
    cfg = get_config()
    set_global_seed(cfg.reproducibility.random_seed)

    logger.info("Starting %s (v%s)...", cfg.project.name, cfg.project.version)

    mode = cfg.ui.mode.lower().strip()
    if mode == "desktop":
        try:
            from sign_language.ui.desktop_app import DesktopApp
            app = DesktopApp(cfg)
            app.run()
        except ImportError as exc:
            logger.warning("CustomTkinter unavailable (%s); falling back to HUD mode.", exc)
            import scripts.run_realtime as realtime_hud
            realtime_hud.main()
    else:
        import scripts.run_realtime as realtime_hud
        realtime_hud.main()


if __name__ == "__main__":
    main()
