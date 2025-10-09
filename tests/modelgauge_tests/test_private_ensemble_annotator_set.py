import os
from unittest import mock


@mock.patch.dict(os.environ, {"VLLM_ENDPOINT_URL": "https://example.org/"}, clear=True)
def test_can_load():
    """This just makes sure things are properly connected. Fuller testing is in the private code."""
    try:
        # from modelgauge.annotator_set import BasicAnnotatorSet
        from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
        from modelgauge.auth.vllm_key import VllmApiKey
        from modelgauge.suts.together_client import TogetherApiKey
        from modelgauge.annotators.wildguard_annotator import WildguardAnnotator

        in_private = True
    except:
        in_private = False

    assert False

    if in_private:
        # annotator_set = EnsembleAnnotatorSet(
        #     secrets={
        #         "together_api_key": TogetherApiKey("fake"),
        #         "huggingface_inference_token": HuggingFaceInferenceToken("fake"),
        #         "vllm_api_key": VllmApiKey("fake"),
        #     }
        # )
        # assert len(annotator_set.annotators) == 4
        # TODO: Update to new ensemble code
        assert False
