"""Hyperparameter optimization with session-grouped cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from sign_language.config.schema import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Outcome of hyperparameter search for a candidate model."""
    candidate_name: str
    best_estimator: Pipeline
    best_score: float
    best_params: dict[str, Any]
    mean_cv_score: float
    std_cv_score: float
    fit_time_seconds: float


def search_best_model(
    candidate_name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: AppConfig,
    quick_mode: bool = False,
) -> SearchResult:
    """Run grouped cross-validated search for a single candidate pipeline.

    Args:
        candidate_name: Identifier of the candidate model.
        pipeline: Un-fitted Pipeline (scaler + classifier).
        param_grid: Prefixed parameter grid dictionary.
        X: Feature matrix of shape (N, feature_dim).
        y: Target label indices of shape (N,).
        groups: Session ID groups of shape (N,).
        config: Central application configuration.
        quick_mode: If True, skips search and fits with default parameters.

    Returns:
        SearchResult: Best estimator and evaluation statistics.
    """
    start_time = time.perf_counter()

    # Determine safe number of folds for StratifiedGroupKFold
    unique_groups = len(np.unique(groups))
    target_folds = config.split.cv_folds
    n_splits = min(target_folds, unique_groups)
    if n_splits < 2:
        logger.warning(
            "Only %d unique groups found; falling back to 2-fold CV for search.",
            unique_groups,
        )
        n_splits = 2

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=config.split.shuffle,
        random_state=config.reproducibility.random_seed,
    )

    scoring = config.training.search.scoring
    n_jobs = config.training.search.n_jobs

    if quick_mode or not param_grid:
        logger.info(
            "Quick mode or empty grid for candidate '%s' — fitting default configuration...",
            candidate_name,
        )
        pipeline.fit(X, y)
        elapsed = time.perf_counter() - start_time
        return SearchResult(
            candidate_name=candidate_name,
            best_estimator=pipeline,
            best_score=0.0,
            best_params={},
            mean_cv_score=0.0,
            std_cv_score=0.0,
            fit_time_seconds=elapsed,
        )

    search_method = config.training.search.method.lower().strip()
    if search_method == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=config.training.search.n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=config.reproducibility.random_seed,
            refit=True,
            verbose=0,
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            verbose=0,
        )

    logger.info(
        "Tuning '%s' using %d-fold StratifiedGroupKFold (%s, metric=%s)...",
        candidate_name,
        n_splits,
        search_method,
        scoring,
    )

    search.fit(X, y, groups=groups)
    elapsed = time.perf_counter() - start_time

    best_idx = search.best_index_
    mean_score = float(search.cv_results_["mean_test_score"][best_idx])
    std_score = float(search.cv_results_["std_test_score"][best_idx])

    logger.info(
        "Candidate '%s' best %s: %.4f (+/- %.4f) in %.2fs",
        candidate_name,
        scoring,
        mean_score,
        std_score,
        elapsed,
    )

    return SearchResult(
        candidate_name=candidate_name,
        best_estimator=search.best_estimator_,
        best_score=float(search.best_score_),
        best_params=search.best_params_,
        mean_cv_score=mean_score,
        std_cv_score=std_score,
        fit_time_seconds=elapsed,
    )
