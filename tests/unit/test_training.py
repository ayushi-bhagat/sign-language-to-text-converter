"""Unit tests for model candidates, hyperparameter grid formatting, and training pipeline."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from sign_language.config.loader import get_config
from sign_language.training.candidates import build_candidate_pipelines, build_param_grid
from sign_language.training.trainer import Trainer


@pytest.mark.unit
def test_candidate_pipelines_construction() -> None:
    """Verify that all 6 candidate pipelines are constructed with scaler and classifier."""
    config = get_config()
    pipelines = build_candidate_pipelines(config)

    expected_candidates = {
        "logistic_regression",
        "knn",
        "svm_rbf",
        "random_forest",
        "extra_trees",
        "mlp",
    }
    assert set(pipelines.keys()) == expected_candidates

    for name, pipe in pipelines.items():
        assert hasattr(pipe, "predict")
        step_names = [s[0] for s in pipe.steps]
        assert "scaler" in step_names
        assert "classifier" in step_names


@pytest.mark.unit
def test_param_grid_formatting() -> None:
    """Verify that hyperparameter grids are prefixed with 'classifier__'."""
    config = get_config()
    grid = build_param_grid("svm_rbf", config)
    for key in grid.keys():
        assert key.startswith("classifier__")

    # Check MLP hidden_layer_sizes tuple conversion
    mlp_grid = build_param_grid("mlp", config)
    if "classifier__hidden_layer_sizes" in mlp_grid:
        for val in mlp_grid["classifier__hidden_layer_sizes"]:
            assert isinstance(val, tuple)


@pytest.mark.unit
def test_trainer_end_to_end_on_synthetic_data() -> None:
    """Verify that Trainer runs, selects a model, and saves artifacts on synthetic data."""
    config = get_config()
    trainer = Trainer(config)

    # 3 sessions, 2 classes (HELLO, NO), 30 samples total
    session_ids = ["s1", "s2", "s3"]
    X_list, y_list, groups_list = [], [], []
    for sid in session_ids:
        for _ in range(10):
            label = "HELLO" if sid in ("s1", "s2") else "NO"
            vec = np.random.randn(63).astype(np.float32)
            if label == "HELLO":
                vec[0] += 5.0  # Make classes separable
            X_list.append(vec)
            y_list.append(label)
            groups_list.append(sid)

    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)

    report = trainer.train(
        X,
        y,
        groups,
        quick_mode=True,
        candidate_subset=["logistic_regression", "knn"],
    )

    assert report.winner_name in ("logistic_regression", "knn")
    assert report.test_metrics["accuracy"] >= 0.0
    assert report.test_metrics["macro_f1"] >= 0.0

    models_dir = Path(config.paths.models) / config.model.version
    assert (models_dir / config.model.classifier_file).exists()
    assert (models_dir / config.model.label_encoder_file).exists()
    assert (models_dir / config.model.metadata_file).exists()
