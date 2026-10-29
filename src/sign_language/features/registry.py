"""Feature set registry and dimension assertions.

This module is the single authority on feature set versions, their expected
dimensions, and the mapping from configuration flags to dimension sizes.
"""

from __future__ import annotations

import logging
from sign_language.config.schema import FeaturesConfig
from sign_language.utils.exceptions import FeatureVersionMismatchError

logger = logging.getLogger(__name__)

# Registry of supported feature set versions mapped to their expected dimensionality
FEATURE_SETS: dict[str, int] = {
    "v1-base-63": 63,
    "v1-ext-83": 83,
}


def expected_dim(features_cfg: FeaturesConfig) -> int:
    """Compute the expected feature vector dimension from FeaturesConfig.

    Args:
        features_cfg: FeaturesConfig section from central config.

    Returns:
        int: Expected feature dimension (63 for base, 83 for extended).

    Raises:
        ValueError: If configuration values do not match registered feature set.
    """
    if features_cfg.use_extended:
        dim = features_cfg.extended.extended_dim
    else:
        dim = features_cfg.base_dim

    expected = FEATURE_SETS.get(features_cfg.feature_set_version)
    if expected is None:
        raise ValueError(
            f"Unrecognised feature_set_version '{features_cfg.feature_set_version}'. "
            f"Registered versions: {list(FEATURE_SETS.keys())}"
        )

    if dim != expected:
        raise ValueError(
            f"Feature dimension mismatch: configured {dim} != registered {expected} "
            f"for version '{features_cfg.feature_set_version}'."
        )

    return dim


def verify_feature_version(loaded_version: str, current_version: str) -> None:
    """Verify that a model's feature set version matches the runtime feature set version.

    Args:
        loaded_version: Feature set version recorded in model_metadata.json.
        current_version: Current runtime feature set version from config.

    Raises:
        FeatureVersionMismatchError: If the versions do not match.
    """
    if loaded_version != current_version:
        raise FeatureVersionMismatchError(
            f"Model feature version '{loaded_version}' does not match "
            f"current runtime feature version '{current_version}'. "
            "Retrain the model or update config.features.feature_set_version."
        )
