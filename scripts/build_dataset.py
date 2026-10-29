"""Script to assemble raw feature arrays into consolidated training matrices.

Usage:
    # Build complete dataset (enforces full 13-class vocabulary and minimums):
    .venv\\Scripts\\python.exe scripts\\build_dataset.py

    # Build available data in partial mode (for development / testing):
    .venv\\Scripts\\python.exe scripts\\build_dataset.py --allow-partial

    # Generate and display dataset integrity report:
    .venv\\Scripts\\python.exe scripts\\build_dataset.py --report
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

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
from sign_language.preprocessing.dataset_builder import DatasetBuilder
from sign_language.utils.exceptions import DatasetError
from sign_language.utils.logging_setup import configure_logging
from sign_language.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate training dataset.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow building partial dataset if some classes/sessions are still pending collection.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed dataset build report.",
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger("build_dataset")
    cfg = get_config()
    set_global_seed(cfg.reproducibility.random_seed)

    builder = DatasetBuilder(cfg)
    try:
        report = builder.build(allow_partial=args.allow_partial)
    except DatasetError as exc:
        print(f"\n❌ Dataset build failed: {exc}")
        print("\nTip: Pass '--allow-partial' to build matrices from currently collected classes for testing.\n")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  DATASET BUILD SUCCESSFUL")
    print("=" * 65)
    print(f"  Total samples       : {report.total_samples}")
    print(f"  Classes present     : {report.num_classes} ({', '.join(report.classes)})")
    print(f"  Feature dimension   : {report.feature_dim}")
    print(f"  Feature set version : {report.feature_set_version}")
    print(f"  Imbalance ratio     : {report.imbalance_ratio:.2f}")
    print("  " + "-" * 61)
    print(f"  {'Class':<16} {'Samples':>10} {'Sessions':>10}")
    print("  " + "-" * 61)
    for cls in report.classes:
        s_cnt = report.sample_counts.get(cls, 0)
        ses_cnt = report.session_counts.get(cls, 0)
        print(f"  {cls:<16} {s_cnt:>10} {ses_cnt:>10}")
    print("=" * 65)
    print("Consolidated arrays written to: data/processed/\n")


if __name__ == "__main__":
    main()
