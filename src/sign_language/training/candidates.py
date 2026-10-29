"""Scikit-learn pipeline definitions for candidate sign language classifiers.

Every candidate is wrapped in a sklearn.pipeline.Pipeline containing:
    1. StandardScaler (or configured scaler)
    2. Classifier

Keeping the scaler inside the pipeline guarantees:
    - Preprocessing statistics are fitted strictly on training folds (no data leakage).
    - Serialised model artifacts are self-contained (inference passes raw features directly).
"""

from __future__ import annotations

import logging
from typing import Any
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from sign_language.config.schema import AppConfig
from sign_language.utils.exceptions import ConfigError

logger = logging.getLogger(__name__)


def _create_scaler(scaler_name: str) -> Any | None:
    """Instantiate preprocessing scaler step."""
    name = scaler_name.lower().strip()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name in ("none", "passthrough"):
        return None
    raise ConfigError(f"Unsupported scaler '{scaler_name}'. Choose 'standard', 'minmax', or 'none'.")


def build_candidate_pipelines(config: AppConfig) -> dict[str, Pipeline]:
    """Construct sklearn Pipeline instances for all configured candidates.

    Args:
        config: Central application configuration.

    Returns:
        dict[str, Pipeline]: Map from candidate name to un-fitted Pipeline.
    """
    seed = config.reproducibility.random_seed
    scaler_type = config.training.scaler
    class_weight = config.training.class_weight

    classifiers: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight=class_weight,
            random_state=seed,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
        ),
        "svm_rbf": SVC(
            C=10.0,
            kernel="rbf",
            probability=True,
            class_weight=class_weight,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=600,
            random_state=seed,
        ),
    }

    pipelines: dict[str, Pipeline] = {}
    for name in config.training.candidates:
        if name not in classifiers:
            raise ConfigError(
                f"Candidate '{name}' in training.candidates is not recognized. "
                f"Available: {list(classifiers.keys())}"
            )

        steps: list[tuple[str, Any]] = []
        scaler = _create_scaler(scaler_type)
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("classifier", classifiers[name]))

        pipelines[name] = Pipeline(steps)

    return pipelines


def build_param_grid(candidate_name: str, config: AppConfig) -> dict[str, list[Any]]:
    """Format hyperparameter search grid with 'classifier__' prefix for Pipeline.

    Args:
        candidate_name: Name of model candidate (e.g. 'svm_rbf').
        config: Central application configuration.

    Returns:
        dict[str, list]: Prefixed parameter grid compatible with GridSearchCV.
    """
    raw_grid = config.training.grids.get(candidate_name, {})
    prefixed: dict[str, list[Any]] = {}

    for param, values in raw_grid.items():
        # YAML parses tuples as lists. MLPClassifier expects tuple for hidden_layer_sizes.
        if candidate_name == "mlp" and param == "hidden_layer_sizes":
            cleaned_values = [tuple(v) if isinstance(v, list) else v for v in values]
            prefixed[f"classifier__{param}"] = cleaned_values
        else:
            prefixed[f"classifier__{param}"] = values

    return prefixed
