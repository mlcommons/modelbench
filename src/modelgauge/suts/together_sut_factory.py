import logging
import os

from airrlogger.log_config import get_logger
from together import Together  # type: ignore

from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.dynamic_sut_factory import (
    DynamicDriverSUTFactory,
    ModelNotSupportedError,
)
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import TogetherChatSUT, TogetherDedicatedChatSUT

logger = get_logger(__name__)
logging.getLogger("together_sut_factory").setLevel(logging.ERROR)


class TogetherSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "together"

    def __init__(self, raw_secrets: RawSecrets, prefer_dedicated: bool | None = None):
        super().__init__(raw_secrets)
        self._client = None
        self.prefer_dedicated = prefer_dedicated

    @property
    def prefer_dedicated(self) -> bool:
        return self._prefer_dedicated

    @prefer_dedicated.setter
    def prefer_dedicated(self, value: bool | None) -> None:
        if value is None:
            env_value = os.environ.get("TOGETHER_PREFER_DEDICATED", None)
            value = True if env_value is None else env_value.lower() in ("1", "true", "yes")
        self._prefer_dedicated = value

    @property
    def client(self) -> Together:
        if self._client is None:
            api_key = self.injected_secrets()[0]
            self._client = Together(api_key=api_key.value)
        return self._client

    @client.setter
    def client(self, value: Together) -> None:
        self._client = value

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
            logger.info(f"Error looking up serverless model {model} on together: {e}")
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

    def make_sut(self, sut_definition: SUTDefinition, prefer_dedicated: bool | None = None) -> PromptResponseSUT:
        if prefer_dedicated is not None:
            self.prefer_dedicated = prefer_dedicated

        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        model = sut_metadata.external_model_name().lower()

        attempts = [
            ("dedicated", self._find_dedicated),
            ("serverless", self._find_serverless),
        ]
        if not self.prefer_dedicated:
            attempts.reverse()

        for kind, finder in attempts:
            result = finder(model)
            if result is not None:
                if kind == "serverless":
                    return TogetherChatSUT(
                        sut_definition.dynamic_uid,
                        sut_metadata.external_model_name(),
                        *self.injected_secrets(),
                    )
                return TogetherDedicatedChatSUT(
                    sut_definition.dynamic_uid,
                    result,
                    *self.injected_secrets(),
                )

        raise ModelNotSupportedError(
            f"Model {sut_metadata.external_model_name()} not found or not available on together serverless nor dedicated endpoints."
        )
