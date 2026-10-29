"""Session-disjoint splitting strategies to prevent data leakage.

Consecutive video frames within a single recording session are near-duplicates
(same lighting, signer, hand orientation, and background). Randomly splitting
frames across train and test sets leads to massive data leakage and severely
inflated accuracy numbers that collapse in real-world deployment.

The functions in this module ensure that all frames belonging to any single
recording session (identified by session_id) reside exclusively in either the
train, validation, or test set.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

logger = logging.getLogger(__name__)


def grouped_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.20,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train, validation, and test sets disjoint by group.

    Args:
        X: Feature matrix of shape (N, feature_dim).
        y: Class labels of shape (N,).
        groups: Session IDs of shape (N,).
        test_size: Proportion of sessions reserved for testing (0.0 to 1.0).
        val_size: Proportion of remaining non-test sessions for validation.
        seed: Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (train_indices, val_indices, test_indices)

    Raises:
        ValueError: If input lengths do not match or too few unique groups exist.
    """
    if not (len(X) == len(y) == len(groups)):
        raise ValueError(
            f"Mismatched array lengths: X={len(X)}, y={len(y)}, groups={len(groups)}"
        )

    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError(
            f"Need at least 3 unique session groups for train/val/test split, got {len(unique_groups)}: {unique_groups}"
        )

    # 1. First split: separate test sessions from (train + val) sessions
    test_gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(test_gss.split(X, y, groups))

    # 2. Second split: separate val sessions from train sessions if val_size > 0
    if val_size > 0:
        val_gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        sub_train_idx, sub_val_idx = next(
            val_gss.split(X[train_val_idx], y[train_val_idx], groups[train_val_idx])
        )
        train_idx = train_val_idx[sub_train_idx]
        val_idx = train_val_idx[sub_val_idx]
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=int)

    # Verify session disjointness
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx]) if len(val_idx) > 0 else set()
    test_groups = set(groups[test_idx])

    assert not (train_groups & test_groups), f"Leakage: train and test share sessions {train_groups & test_groups}"
    assert not (train_groups & val_groups), f"Leakage: train and val share sessions {train_groups & val_groups}"
    assert not (val_groups & test_groups), f"Leakage: val and test share sessions {val_groups & test_groups}"

    logger.info(
        "Grouped split complete: %d train (%d sessions), %d val (%d sessions), %d test (%d sessions)",
        len(train_idx),
        len(train_groups),
        len(val_idx),
        len(val_groups),
        len(test_idx),
        len(test_groups),
    )

    return train_idx, val_idx, test_idx


def get_stratified_group_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Generate StratifiedGroupKFold cross-validation splits.

    Ensures that folds preserve class distribution as closely as possible
    while guaranteeing that each session_id appears in exactly one fold.

    Args:
        X: Feature matrix of shape (N, feature_dim).
        y: Class labels of shape (N,).
        groups: Group identifiers (session_ids) of shape (N,).
        n_splits: Number of folds (cv_folds).
        shuffle: Whether to shuffle before splitting.
        seed: Random state seed.

    Yields:
        Iterator of (train_idx, val_idx) fold splits.
    """
    unique_groups = len(np.unique(groups))
    effective_splits = min(n_splits, unique_groups)
    if effective_splits < n_splits:
        logger.warning(
            "Unique session groups (%d) < requested cv_folds (%d). Clamping folds to %d.",
            unique_groups,
            n_splits,
            effective_splits,
        )

    sgkf = StratifiedGroupKFold(
        n_splits=effective_splits,
        shuffle=shuffle,
        random_state=seed if shuffle else None,
    )
    for train_idx, val_idx in sgkf.split(X, y, groups):
        # Assert fold disjointness
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert not (train_groups & val_groups), "Leakage detected between CV train and val folds!"
        yield train_idx, val_idx


def random_split_for_leakage_demo(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a naive, un-grouped random train/test split.

    WARNING: This protocol produces data leakage because adjacent frames from
    the same recording session are distributed across both sets.
    This function exists solely to quantify and report the leakage gap in Phase 7.

    Args:
        X: Feature matrix.
        y: Class labels.
        test_size: Proportion of samples for test.
        seed: Random seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: (train_idx, test_idx)
    """
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    return train_idx, test_idx
