"""Integration tests for RealtimePipeline lifecycle and error handling."""

import pytest

from sign_language.config.loader import get_config
from sign_language.inference.pipeline import RealtimePipeline
from sign_language.utils.exceptions import InferenceError


@pytest.mark.integration
def test_pipeline_unstarted_step_raises_inference_error() -> None:
    """Calling step() before start() must raise InferenceError."""
    config = get_config()
    pipeline = RealtimePipeline(config)
    assert not pipeline._is_running

    with pytest.raises(InferenceError, match="Pipeline is not running"):
        pipeline.step()
