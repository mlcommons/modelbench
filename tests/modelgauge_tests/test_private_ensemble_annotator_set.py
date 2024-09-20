import os
from unittest.mock import Mock, patch

import pytest

from modelgauge.suts.together_client import TogetherApiKey


def test_can_load():
    try:
        # EnsembleAnnotator is required by the private annotators
        # If we can import it, then the EnsembleAnnotatorSet can be instantiated
        from modelgauge.private_ensemble_annotator_set import EnsembleAnnotatorSet

        assert True
    except:
        # The EnsembleAnnotator can't be implemented, so the EnsembleAnnotatorSet can't either
        with pytest.raises(NotImplementedError):
            from modelgauge.private_ensemble_annotator_set import EnsembleAnnotatorSet


def test_annotators():
    try:
        from modelgauge.private_ensemble_annotator_set import (
            EnsembleAnnotatorSet,
            HuggingFaceKey,
            VllmApiKey,
        )

        os.environ["VLLM_ENDPOINT_URL"] = "fake"
        annotators = EnsembleAnnotatorSet(
            secrets={
                "together_api_key": TogetherApiKey("fake"),
                "huggingface_key": HuggingFaceKey("fake"),
                "vllm_api_key": VllmApiKey("fake"),
            }
        )
        assert len(annotators.annotators) == 4
    except:
        # The EnsembleAnnotator can't be implemented, so the EnsembleAnnotatorSet can't either
        with pytest.raises(NotImplementedError):
            from modelgauge.private_ensemble_annotator_set import EnsembleAnnotatorSet
