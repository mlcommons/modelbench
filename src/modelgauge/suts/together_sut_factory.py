import logging
from tkinter import N

from together import Together  # type: ignore

from airrlogger.log_config import get_logger

from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.dynamic_sut_factory import DynamicDriverSUTFactory, DynamicSUTFactory, ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import TogetherChatSUT, TogetherDedicatedChatSUT

logger = get_logger(__name__)
logging.getLogger("together_sut_factory").setLevel(logging.ERROR)


class TogetherSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "together"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)

    def _find_serverless(self, model: str) -> str | None:
        try:
            self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Anybody home?"},
                ],
            )
            return model
        except Exception as e:
            logger.error(f"Error looking up serverless model {model} on together: {e}")
            return None

    def _find_dedicated(self, model: str) -> str | None:
        try:
            endpoints = self.client.endpoints.list(type="dedicated", mine=True)
            for endpoint in endpoints.data:
                if endpoint.model.lower() == model:
                    return endpoint.name
        except Exception as e:
            logger.error(f"Error looking up dedicated endpoints for {model} on together: {e}")
            return None

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(TogetherApiKey)
        return [api_key]

    def make_sut(self, sut_definition: SUTDefinition) -> PromptResponseSUT:
        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        model = sut_metadata.external_model_name().lower()
        # first try serverless
        model_name = self._find_serverless(model)
        if model_name is not None:
            return TogetherChatSUT(
                sut_definition.dynamic_uid,
                sut_metadata.external_model_name(),
                *self.injected_secrets(),
            )
        # serverless failed; try dedicated.
        endpoint_name = self._find_dedicated(model)
        if endpoint_name is not None:
            return TogetherDedicatedChatSUT(
                sut_definition.dynamic_uid,
                endpoint_name,
                *self.injected_secrets(),
            )

        raise ModelNotSupportedError(
            f"Model {sut_metadata.external_model_name()} not found or not available on together serverless nor dedicated endpoints."
        )
