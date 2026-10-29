"""Script to train and tune sign language classifiers using session-grouped CV.

Usage:
    # Full training and tuning of all 6 candidate models:
    .venv\\Scripts\\python.exe scripts\\train_model.py

    # Quick train mode (skips hyperparameter search, fits default candidates):
    .venv\\Scripts\\python.exe scripts\\train_model.py --quick

    # Train a specific candidate only (e.g. svm_rbf, random_forest):
    .venv\\Scripts\\python.exe scripts\\train_model.py --candidate svm_rbf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import numpy as np

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
from sign_language.training.trainer import Trainer
from sign_language.utils.exceptions import DatasetError, TrainingError
from sign_language.utils.logging_setup import configure_logging
from sign_language.utils.paths import resolve
from sign_language.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate sign classification models.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick training without exhaustive hyperparameter grid search.",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Train only this candidate model (e.g. svm_rbf, random_forest, knn, mlp).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=True,
        help="Allow training on currently available classes even if full vocabulary is incomplete.",
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger("train_model")
    cfg = get_config()
    set_global_seed(cfg.reproducibility.random_seed)

    processed_dir = resolve(cfg.paths.data_processed)
    x_file = processed_dir / "X.npy"
    y_file = processed_dir / "y.npy"
    groups_file = processed_dir / "groups.npy"

    # If processed matrices don't exist yet, build them
    if not (x_file.exists() and y_file.exists() and groups_file.exists()):
        print("\nProcessed matrices not found. Building dataset from features...")
        builder = DatasetBuilder(cfg)
        try:
            builder.build(allow_partial=args.allow_partial)
        except DatasetError as exc:
            print(f"❌ Failed to build dataset: {exc}")
            sys.exit(1)

    X = np.load(str(x_file), allow_pickle=True)
    y = np.load(str(y_file), allow_pickle=True)
    groups = np.load(str(groups_file), allow_pickle=True)

    print("\n" + "=" * 70)
    print("  MODEL TRAINING & SELECTION")
    print("=" * 70)
    print(f"  Dataset Shape    : {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Classes          : {len(np.unique(y))} {list(np.unique(y))}")
    print(f"  Unique Sessions  : {len(np.unique(groups))}")
    print(f"  Split Strategy   : {cfg.split.strategy} ({cfg.split.cv_folds}-fold CV)")
    print(f"  Primary Metric   : {cfg.training.primary_metric}")
    print("-" * 70)

    candidate_subset = [args.candidate] if args.candidate else None
    trainer = Trainer(cfg)

    try:
        report = trainer.train(
            X,
            y,
            groups,
            quick_mode=args.quick,
            candidate_subset=candidate_subset,
        )
    except TrainingError as exc:
        print(f"\n❌ Training failed: {exc}\n")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"  WINNING MODEL: {report.winner_name.upper()}")
    print("=" * 70)
    print(f"  Grouped CV Macro-F1 : {report.cv_macro_f1:.4f}")
    print(f"  Test Accuracy       : {report.test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Test Macro-F1       : {report.test_metrics['macro_f1']:.4f}")
    print(f"  Test Macro-Recall   : {report.test_metrics['macro_recall']:.4f}")
    if report.winner_params:
        print(f"  Best Parameters     : {report.winner_params}")

    if report.candidate_comparison:
        print("\n  Candidate Model Comparison:")
        print(f"  {'Model':<22} {'CV Macro-F1':>12} {'Fit Time (s)':>14}")
        print("  " + "-" * 50)
        for model_name, stats in report.candidate_comparison.items():
            print(f"  {model_name:<22} {stats['cv_macro_f1']:>12.4f} {stats['fit_time_seconds']:>14.2f}")

    if report.leakage_gap:
        print("\n  Data Leakage Experiment:")
        print(f"  Grouped (Honest) CV : {report.leakage_gap.get('grouped_macro_f1', 0):.4f}")
        print(f"  Random (Leaky) CV   : {report.leakage_gap.get('random_macro_f1', 0):.4f}")
        print(f"  Leakage Gap (Inflation): +{report.leakage_gap.get('leakage_gap', 0):.4f}")

    print("\n  Per-Class Performance on Test Set:")
    print(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 60)
    for cls_name, metrics in report.per_class_report.items():
        if isinstance(metrics, dict) and "precision" in metrics:
            print(
                f"  {cls_name:<16} "
                f"{metrics['precision']:>10.3f} "
                f"{metrics['recall']:>10.3f} "
                f"{metrics['f1-score']:>10.3f} "
                f"{int(metrics['support']):>10d}"
            )
    print("=" * 70)
    print("Artifacts saved to: models/v1/\n")


if __name__ == "__main__":
    main()
