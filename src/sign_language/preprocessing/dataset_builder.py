"""Dataset aggregation, validation, and serialization.

Aggregates individual session feature arrays into consolidated X, y, and groups
matrices, validates dataset targets and class balance, and produces an auditable
build manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import numpy as np

from sign_language.config.schema import AppConfig
from sign_language.data_collection.persistence import DatasetReader
from sign_language.features.registry import verify_feature_version
from sign_language.utils.exceptions import DatasetError
from sign_language.utils.paths import ensure_dir, resolve

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetReport:
    """Summary metrics of a built dataset."""
    total_samples: int
    num_classes: int
    classes: list[str]
    sample_counts: dict[str, int]
    session_counts: dict[str, int]
    imbalance_ratio: float
    feature_dim: int
    feature_set_version: str
    built_at: str


class DatasetBuilder:
    """Aggregates and validates raw features into unified training matrices.

    Args:
        config: Central application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        self._features_dir = resolve(config.paths.data_features)
        self._processed_dir = resolve(config.paths.data_processed)
        self._reader = DatasetReader(self._features_dir)

    def build(self, allow_partial: bool = False) -> DatasetReport:
        """Merge session samples, validate criteria, and write consolidated arrays.

        Args:
            allow_partial: If True, relaxes the strict vocabulary completeness
                           and minimum sample checks. Useful for developing and
                           testing models incrementally before the full 13-class
                           dataset is collected.

        Returns:
            DatasetReport: Summary statistics of the aggregated dataset.

        Raises:
            DatasetError: If data files are missing, corrupt, or fail validation checks.
        """
        if not self._reader.manifest_exists():
            raise DatasetError(
                f"Dataset manifest not found in {self._features_dir}. "
                "Run scripts/collect_data.py to collect samples first."
            )

        try:
            X, y, session_ids = self._reader.load()
        except Exception as exc:
            raise DatasetError(f"Failed to load dataset: {exc}") from exc

        if len(y) == 0:
            raise DatasetError("Dataset contains zero samples.")

        sample_counts = self._reader.class_counts()
        session_dict = self._reader.session_counts()
        session_counts = {cls: len(s_dict) for cls, s_dict in session_dict.items()}

        classes_found = sorted(list(sample_counts.keys()))
        min_count = min(sample_counts.values()) if sample_counts else 0
        max_count = max(sample_counts.values()) if sample_counts else 0
        imbalance_ratio = float(max_count / min_count) if min_count > 0 else 0.0

        # Feature version check
        current_version = self._cfg.features.feature_set_version
        manifest_rows = self._reader._read_manifest()
        versions_present = {row["feature_set_version"] for row in manifest_rows}
        for ver in versions_present:
            verify_feature_version(ver, current_version)

        # Validation rules
        if not allow_partial:
            self._validate_full(classes_found, sample_counts, session_counts, imbalance_ratio)
        else:
            logger.warning(
                "Running in partial-dataset mode: strict minimums and full vocabulary checks are relaxed."
            )

        y_arr = np.array(y, dtype=object)
        groups_arr = np.array(session_ids, dtype=object)

        # Ensure output directories exist
        ensure_dir(self._processed_dir)

        # Write consolidated numpy arrays to data/processed
        np.save(str(self._processed_dir / "X.npy"), X)
        np.save(str(self._processed_dir / "y.npy"), y_arr)
        np.save(str(self._processed_dir / "groups.npy"), groups_arr)

        report = DatasetReport(
            total_samples=len(y),
            num_classes=len(classes_found),
            classes=classes_found,
            sample_counts=sample_counts,
            session_counts=session_counts,
            imbalance_ratio=round(imbalance_ratio, 2),
            feature_dim=X.shape[1],
            feature_set_version=current_version,
            built_at=datetime.now().isoformat(),
        )

        # Save build manifest
        manifest_path = self._processed_dir / "build_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2)

        logger.info(
            "Dataset successfully built in %s (%d samples, %d classes, dim=%d)",
            self._processed_dir,
            len(y),
            len(classes_found),
            X.shape[1],
        )
        return report

    def _validate_full(
        self,
        classes_found: list[str],
        sample_counts: dict[str, int],
        session_counts: dict[str, int],
        imbalance_ratio: float,
    ) -> None:
        """Enforce strict Phase 4 pre-training validation gates."""
        configured_classes = set(self._cfg.dataset.classes)
        present_classes = set(classes_found)
        missing_classes = configured_classes - present_classes
        if missing_classes:
            raise DatasetError(
                f"Missing {len(missing_classes)} classes from vocabulary: {sorted(missing_classes)}. "
                "Collect samples for all configured classes, or pass allow_partial=True."
            )

        min_samples = self._cfg.dataset.samples_per_class_minimum
        under_sampled = [cls for cls, cnt in sample_counts.items() if cnt < min_samples]
        if under_sampled:
            details = [f"{cls} ({sample_counts[cls]} < {min_samples})" for cls in under_sampled]
            raise DatasetError(
                f"Classes below minimum sample threshold ({min_samples}): {', '.join(details)}"
            )

        min_sessions = self._cfg.dataset.sessions_per_class_minimum
        under_sessioned = [cls for cls, s_cnt in session_counts.items() if s_cnt < min_sessions]
        if under_sessioned:
            details = [f"{cls} ({session_counts[cls]} < {min_sessions})" for cls in under_sessioned]
            raise DatasetError(
                f"Classes below minimum session threshold ({min_sessions}): {', '.join(details)}"
            )

        max_imbalance = self._cfg.training.fail_on_class_imbalance_ratio
        if imbalance_ratio > max_imbalance:
            raise DatasetError(
                f"Class imbalance ratio {imbalance_ratio:.2f} exceeds threshold {max_imbalance:.2f}."
            )
