"""Unit tests for session-disjoint dataset splitting."""

import numpy as np
import pytest

from sign_language.preprocessing.splitter import (
    get_stratified_group_kfold,
    grouped_split,
    random_split_for_leakage_demo,
)


@pytest.fixture
def mock_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic samples with known session groupings."""
    # 5 sessions: s1, s2, s3 (class A), s4, s5 (class B)
    # 20 samples per session = 100 samples total
    session_ids = ["s1", "s2", "s3", "s4", "s5"]
    groups_list = []
    y_list = []
    for sid in session_ids:
        label = "HELLO" if sid in ("s1", "s2", "s3") else "NO"
        groups_list.extend([sid] * 20)
        y_list.extend([label] * 20)

    X = np.random.randn(len(groups_list), 63).astype(np.float32)
    y = np.array(y_list)
    groups = np.array(groups_list)
    return X, y, groups


@pytest.mark.unit
def test_grouped_split_zero_leakage(mock_dataset: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Assert train, val, and test session groups are strictly disjoint."""
    X, y, groups = mock_dataset
    train_idx, val_idx, test_idx = grouped_split(
        X, y, groups, test_size=0.20, val_size=0.20, seed=42
    )

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

    # Critical assertions: zero session overlap
    assert len(train_groups & test_groups) == 0, f"Overlap between train and test: {train_groups & test_groups}"
    assert len(train_groups & val_groups) == 0, f"Overlap between train and val: {train_groups & val_groups}"
    assert len(val_groups & test_groups) == 0, f"Overlap between val and test: {val_groups & test_groups}"

    # All indices cover the complete dataset with no duplicates
    combined_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(combined_idx) == len(X)
    assert len(np.unique(combined_idx)) == len(X)


@pytest.mark.unit
def test_stratified_group_kfold_disjoint(mock_dataset: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Assert cross-validation folds have zero session overlap between train and val."""
    X, y, groups = mock_dataset
    folds = list(get_stratified_group_kfold(X, y, groups, n_splits=3, seed=42))
    assert len(folds) == 3

    for train_idx, val_idx in folds:
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert len(train_groups & val_groups) == 0, f"CV fold leakage: {train_groups & val_groups}"


@pytest.mark.unit
def test_random_split_exhibits_leakage(mock_dataset: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Demonstrate that naive random split leaks sessions across train and test."""
    X, y, groups = mock_dataset
    train_idx, test_idx = random_split_for_leakage_demo(X, y, test_size=0.20, seed=42)

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    # In a naive random split with 5 sessions and 100 samples, all sessions leak into both splits!
    shared_sessions = train_groups & test_groups
    assert len(shared_sessions) > 0, "Expected naive random split to demonstrate session leakage."
