"""Model evaluation metrics, confusion matrix plotting, and leakage analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from sign_language.preprocessing.splitter import get_stratified_group_kfold
from sign_language.utils.paths import ensure_dir

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate core classification metrics.

    Args:
        y_true: Ground truth labels (encoded integer or string).
        y_pred: Predicted labels.

    Returns:
        dict[str, float]: accuracy, macro_f1, macro_precision, macro_recall, balanced_accuracy.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def compute_per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict[str, Any]:
    """Generate per-class precision, recall, and F1 metrics.

    Args:
        y_true: Ground truth integer class indices.
        y_pred: Predicted integer class indices.
        target_names: Ordered list of string class names.

    Returns:
        dict[str, Any]: Full dictionary report from sklearn.classification_report.
    """
    labels = list(range(len(target_names)))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    output_path: Path,
) -> None:
    """Render and save confusion matrix heatmap figure.

    Args:
        y_true: Ground truth integer class indices.
        y_pred: Predicted integer class indices.
        target_names: Class labels for axes.
        output_path: Destination path for PNG image.
    """
    ensure_dir(output_path.parent)

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        labels = list(range(len(target_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

        fig, ax = plt.subplots(figsize=(max(8, len(target_names) * 0.8), max(6, len(target_names) * 0.7)))
        sns.heatmap(
            cm_norm,
            annot=cm,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            ax=ax,
            cbar=True,
        )
        ax.set_title("Confusion Matrix (Counts annotated, Color normalized by row)")
        ax.set_xlabel("Predicted Sign")
        ax.set_ylabel("True Sign")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=200)
        plt.close(fig)
        logger.info("Confusion matrix figure saved to %s", output_path)

    except Exception as exc:
        logger.warning("Could not generate confusion matrix plot: %s", exc)


def compare_split_leakage(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Quantify dataset leakage by comparing grouped vs naive random cross-validation.

    Args:
        pipeline: Candidate model pipeline.
        X: Feature matrix.
        y: Target label indices.
        groups: Session IDs.
        cv_folds: Number of folds.
        seed: Random seed.

    Returns:
        dict[str, float]: grouped_macro_f1, random_macro_f1, leakage_gap
    """
    unique_groups = len(np.unique(groups))
    effective_folds = min(cv_folds, unique_groups)
    if effective_folds < 2:
        effective_folds = 2

    # 1. Grouped cross-validation (honest, session-disjoint)
    grouped_cv = get_stratified_group_kfold(
        X, y, groups, n_splits=effective_folds, shuffle=True, seed=seed
    )
    grouped_scores = cross_val_score(
        pipeline, X, y, cv=grouped_cv, scoring="f1_macro", n_jobs=-1
    )
    grouped_macro_f1 = float(np.mean(grouped_scores))

    # 2. Naive random cross-validation (leaky, consecutive frames scattered across train & test)
    random_cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    random_scores = cross_val_score(
        pipeline, X, y, cv=random_cv, scoring="f1_macro", n_jobs=-1
    )
    random_macro_f1 = float(np.mean(random_scores))

    gap = random_macro_f1 - grouped_macro_f1

    logger.info(
        "Leakage experiment: Grouped macro-F1 = %.4f | Naive Random macro-F1 = %.4f | Gap = %.4f",
        grouped_macro_f1,
        random_macro_f1,
        gap,
    )

    return {
        "grouped_macro_f1": round(grouped_macro_f1, 4),
        "random_macro_f1": round(random_macro_f1, 4),
        "leakage_gap": round(gap, 4),
    }
