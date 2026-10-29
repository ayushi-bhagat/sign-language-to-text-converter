"""Script to evaluate a trained sign classification model on held-out test data.

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluate_model.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import joblib
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
from sign_language.features.registry import verify_feature_version
from sign_language.preprocessing.splitter import grouped_split
from sign_language.training.evaluation import (
    compute_metrics,
    compute_per_class_report,
    save_confusion_matrix_figure,
)
from sign_language.utils.exceptions import ModelLoadError
from sign_language.utils.logging_setup import configure_logging
from sign_language.utils.paths import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained sign language model.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to model artifacts directory (defaults to models/v1).",
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger("evaluate_model")
    cfg = get_config()

    models_dir = Path(args.model_dir) if args.model_dir else resolve(cfg.paths.models) / cfg.model.version
    clf_path = models_dir / cfg.model.classifier_file
    le_path = models_dir / cfg.model.label_encoder_file
    meta_path = models_dir / cfg.model.metadata_file

    if not (clf_path.exists() and le_path.exists() and meta_path.exists()):
        raise ModelLoadError(
            f"Model artifacts not found in {models_dir}. Run scripts/train_model.py first."
        )

    pipeline = joblib.load(str(clf_path))
    label_encoder = joblib.load(str(le_path))
    with meta_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    # Feature version enforcement
    if cfg.model.enforce_feature_version_match:
        verify_feature_version(
            metadata.get("feature_set_version", ""),
            cfg.features.feature_set_version,
        )

    processed_dir = resolve(cfg.paths.data_processed)
    x_file = processed_dir / "X.npy"
    y_file = processed_dir / "y.npy"
    groups_file = processed_dir / "groups.npy"

    if not (x_file.exists() and y_file.exists() and groups_file.exists()):
        raise FileNotFoundError(f"Processed dataset matrices not found in {processed_dir}.")

    X = np.load(str(x_file), allow_pickle=True)
    y = np.load(str(y_file), allow_pickle=True)
    groups = np.load(str(groups_file), allow_pickle=True)

    y_encoded = label_encoder.transform(y)

    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        _, _, test_idx = grouped_split(
            X,
            y_encoded,
            groups,
            test_size=cfg.split.test_size,
            val_size=0.0,
            seed=cfg.reproducibility.random_seed,
        )
        X_eval, y_eval = X[test_idx], y_encoded[test_idx]
        eval_label = f"Held-out test set ({len(test_idx)} samples across {len(np.unique(groups[test_idx]))} sessions)"
    else:
        X_eval, y_eval = X, y_encoded
        eval_label = f"Full dataset ({len(X)} samples — fewer than 3 sessions available)"

    y_pred = pipeline.predict(X_eval)
    metrics = compute_metrics(y_eval, y_pred)
    class_names = list(label_encoder.classes_)
    per_class = compute_per_class_report(y_eval, y_pred, target_names=class_names)

    # Save confusion matrix plot
    cm_path = resolve(cfg.paths.figures) / "confusion_matrix.png"
    save_confusion_matrix_figure(y_eval, y_pred, target_names=class_names, output_path=cm_path)

    print("\n" + "=" * 65)
    print("  MODEL EVALUATION REPORT")
    print("=" * 65)
    print(f"  Model Version       : {metadata.get('model_version', 'unknown')}")
    print(f"  Algorithm           : {metadata.get('winner', 'unknown')}")
    print(f"  Feature Set Version : {metadata.get('feature_set_version', 'unknown')}")
    print(f"  Evaluation Protocol : {eval_label}")
    print("-" * 65)
    print(f"  Accuracy            : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro-F1            : {metrics['macro_f1']:.4f}")
    print(f"  Macro-Precision     : {metrics['macro_precision']:.4f}")
    print(f"  Macro-Recall        : {metrics['macro_recall']:.4f}")
    print(f"  Balanced Accuracy   : {metrics['balanced_accuracy'] * 100:.2f}%")
    print("-" * 65)
    print("  Per-Class Metrics:")
    print(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 60)
    for cls_name, cls_metrics in per_class.items():
        if isinstance(cls_metrics, dict) and "precision" in cls_metrics:
            print(
                f"  {cls_name:<16} "
                f"{cls_metrics['precision']:>10.3f} "
                f"{cls_metrics['recall']:>10.3f} "
                f"{cls_metrics['f1-score']:>10.3f} "
                f"{int(cls_metrics['support']):>10d}"
            )
    print("=" * 65)
    print(f"Confusion matrix image saved to: {cm_path}\n")


if __name__ == "__main__":
    main()
