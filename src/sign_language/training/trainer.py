"""Training orchestrator for candidate model comparison, tuning, and artifact serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sign_language.config.schema import AppConfig
from sign_language.preprocessing.splitter import grouped_split
from sign_language.training.candidates import build_candidate_pipelines, build_param_grid
from sign_language.training.evaluation import (
    compare_split_leakage,
    compute_metrics,
    compute_per_class_report,
    save_confusion_matrix_figure,
)
from sign_language.training.tuning import SearchResult, search_best_model
from sign_language.utils.exceptions import TrainingError
from sign_language.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Consolidated report produced after model selection and training."""
    winner_name: str
    winner_params: dict[str, Any]
    cv_macro_f1: float
    test_metrics: dict[str, float]
    per_class_report: dict[str, Any]
    candidate_comparison: dict[str, dict[str, Any]]
    leakage_gap: dict[str, float]
    trained_at: str
    feature_set_version: str
    model_version: str


class Trainer:
    """Manages cross-validation model selection, final refit, and artifact persistence.

    Args:
        config: Central application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._models_dir = resolve(config.paths.models) / config.model.version
        self._reports_metrics_dir = resolve(config.paths.metrics)
        self._reports_figures_dir = resolve(config.paths.figures)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        quick_mode: bool = False,
        candidate_subset: list[str] | None = None,
    ) -> TrainingReport:
        """Execute model selection, refit winner, evaluate on held-out test, and save.

        Args:
            X: Feature matrix of shape (N, feature_dim).
            y: String class labels of shape (N,).
            groups: Session ID groups of shape (N,).
            quick_mode: If True, uses default parameters without extensive grid search.
            candidate_subset: Optional list to train only specific candidates.

        Returns:
            TrainingReport: Complete summary of training and evaluation.

        Raises:
            TrainingError: If training fails or requirements are not met.
        """
        if len(X) == 0:
            raise TrainingError("Cannot train on empty dataset.")

        # Fit LabelEncoder deterministically on sorted unique classes
        unique_classes = sorted(list(np.unique(y)))
        if len(unique_classes) < 2:
            raise TrainingError(
                f"Need at least 2 distinct classes to train a classifier, found {len(unique_classes)}: {unique_classes}"
            )

        label_encoder = LabelEncoder()
        label_encoder.fit(unique_classes)
        y_encoded = label_encoder.transform(y)

        # Session-grouped train/test split
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 3:
            train_idx, val_idx, test_idx = grouped_split(
                X,
                y_encoded,
                groups,
                test_size=self._cfg.split.test_size,
                val_size=0.0,  # Cross-validation already handles validation folds
                seed=self._cfg.reproducibility.random_seed,
            )
            X_train, y_train, groups_train = X[train_idx], y_encoded[train_idx], groups[train_idx]
            X_test, y_test, groups_test = X[test_idx], y_encoded[test_idx], groups[test_idx]
        else:
            logger.warning(
                "Only %d session groups exist (%s). Grouped holdout test split requires >= 3. "
                "Evaluating on training set.",
                len(unique_groups),
                list(unique_groups),
            )
            X_train, y_train, groups_train = X, y_encoded, groups
            X_test, y_test, groups_test = X, y_encoded, groups

        all_pipelines = build_candidate_pipelines(self._cfg)
        candidates_to_run = candidate_subset or self._cfg.training.candidates

        search_results: dict[str, SearchResult] = {}
        candidate_summary: dict[str, dict[str, Any]] = {}

        for name in candidates_to_run:
            if name not in all_pipelines:
                logger.warning("Skipping unconfigured candidate '%s'", name)
                continue

            pipeline = all_pipelines[name]
            param_grid = build_param_grid(name, self._cfg) if not quick_mode else {}

            res = search_best_model(
                candidate_name=name,
                pipeline=pipeline,
                param_grid=param_grid,
                X=X_train,
                y=y_train,
                groups=groups_train,
                config=self._cfg,
                quick_mode=quick_mode,
            )
            search_results[name] = res
            candidate_summary[name] = {
                "cv_macro_f1": round(res.mean_cv_score, 4),
                "std_cv_score": round(res.std_cv_score, 4),
                "fit_time_seconds": round(res.fit_time_seconds, 2),
                "best_params": res.best_params,
            }

        # Select winner on highest CV macro F1 score
        if quick_mode:
            # In quick mode, evaluate default models on train fold to pick winner
            winner_name = candidates_to_run[0]
            winner_res = search_results[winner_name]
        else:
            winner_name = max(search_results.keys(), key=lambda k: search_results[k].mean_cv_score)
            winner_res = search_results[winner_name]

        logger.info(
            "Selected WINNER: '%s' with grouped CV macro-F1 = %.4f",
            winner_name,
            winner_res.mean_cv_score,
        )

        # Refit winning pipeline on all training data
        winning_pipeline = winner_res.best_estimator
        winning_pipeline.fit(X_train, y_train)

        # Evaluate once on held-out test data
        test_pred = winning_pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, test_pred)
        per_class_report = compute_per_class_report(
            y_test, test_pred, target_names=list(label_encoder.classes_)
        )

        logger.info(
            "Final Held-Out Test Metrics for '%s': Accuracy=%.4f, Macro-F1=%.4f",
            winner_name,
            test_metrics["accuracy"],
            test_metrics["macro_f1"],
        )

        # Run leakage experiment if requested
        leakage_results: dict[str, float] = {}
        if self._cfg.split.report_leakage_comparison:
            leakage_results = compare_split_leakage(
                winning_pipeline,
                X,
                y_encoded,
                groups,
                cv_folds=self._cfg.split.cv_folds,
                seed=self._cfg.reproducibility.random_seed,
            )

        # Save artifacts
        self._save_artifacts(
            winning_pipeline,
            label_encoder,
            winner_name,
            winner_res.best_params,
            winner_res.mean_cv_score,
            test_metrics,
            per_class_report,
            candidate_summary,
            leakage_results,
        )

        # Save confusion matrix figure
        cm_path = self._reports_figures_dir / "confusion_matrix.png"
        save_confusion_matrix_figure(
            y_test,
            test_pred,
            target_names=list(label_encoder.classes_),
            output_path=cm_path,
        )

        return TrainingReport(
            winner_name=winner_name,
            winner_params=winner_res.best_params,
            cv_macro_f1=round(winner_res.mean_cv_score, 4),
            test_metrics=test_metrics,
            per_class_report=per_class_report,
            candidate_comparison=candidate_summary,
            leakage_gap=leakage_results,
            trained_at=datetime.now().isoformat(),
            feature_set_version=self._cfg.features.feature_set_version,
            model_version=self._cfg.model.version,
        )

    def _save_artifacts(
        self,
        pipeline: Pipeline,
        label_encoder: LabelEncoder,
        winner_name: str,
        best_params: dict[str, Any],
        cv_macro_f1: float,
        test_metrics: dict[str, float],
        per_class_report: dict[str, Any],
        candidate_summary: dict[str, dict[str, Any]],
        leakage_results: dict[str, float],
    ) -> None:
        """Persist model, encoder, and model_metadata.json to models/v1/."""
        ensure_dir(self._models_dir)
        ensure_dir(self._reports_metrics_dir)

        clf_path = self._models_dir / self._cfg.model.classifier_file
        le_path = self._models_dir / self._cfg.model.label_encoder_file
        meta_path = self._models_dir / self._cfg.model.metadata_file

        joblib.dump(pipeline, str(clf_path))
        joblib.dump(label_encoder, str(le_path))

        metadata = {
            "model_version": self._cfg.model.version,
            "feature_set_version": self._cfg.features.feature_set_version,
            "feature_dim": self._cfg.features.base_dim,
            "winner": winner_name,
            "best_params": best_params,
            "classes": list(label_encoder.classes_),
            "cv_macro_f1": round(cv_macro_f1, 4),
            "test_metrics": test_metrics,
            "per_class_report": per_class_report,
            "candidates_compared": candidate_summary,
            "leakage_experiment": leakage_results,
            "trained_at": datetime.now().isoformat(),
        }

        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        # Save metrics JSON in reports/metrics
        with (self._reports_metrics_dir / "model_comparison.json").open("w", encoding="utf-8") as fh:
            json.dump(candidate_summary, fh, indent=2)

        logger.info("Model artifacts saved to %s", self._models_dir)
