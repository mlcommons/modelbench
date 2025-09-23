import logging
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub.inference._providers._common import TaskProviderHelper


def test_ensure_annoying_hf_warnings_suppressed(caplog):
    from modelgauge.suts import huggingface_sut_factory as sut_factory  # noqa: F401

    hf_logger_name = "huggingface_hub.inference._providers._common"

    helper = TaskProviderHelper(provider="together", base_url="https://api", task="conversational")

    mocked_mapping = SimpleNamespace(
        provider="together",
        task="conversational",
        status="error",
        provider_id="model-x",
    )

    with patch(
        "huggingface_hub.inference._providers._common._fetch_inference_provider_mapping",
        return_value=[mocked_mapping],
    ):
        with caplog.at_level(logging.WARNING):
            _ = helper._prepare_mapping_info("some-model")

    # No WARNING should be recorded from the HF logger due to suppression at ERROR level
    assert not any(
        rec.name == hf_logger_name and rec.levelno == logging.WARNING for rec in caplog.records
    ), "Expected no WARNING records from huggingface_hub.inference._providers._common"
