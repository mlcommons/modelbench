import logging

from together import Together  # type: ignore

from airrlogger.log_config import get_logger

from modelgauge.auth.together_key import TogetherApiKey
from modelgauge.dynamic_sut_factory import DynamicDriverSUTFactory, ModelNotSupportedError
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import TogetherChatSUT, TogetherDedicatedChatSUT

logger = get_logger(__name__)
logging.getLogger("together_sut_factory").setLevel(logging.ERROR)


class TogetherServerlessSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "together-serverless"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None

    @property
    def client(self) -> Together:
        if self._client is None:
            api_key = self.injected_secrets()[0]
            self._client = Together(api_key=api_key.value)
        return self._client

    @client.setter
    def client(self, value: Together) -> None:
        self._client = value

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(TogetherApiKey)
        return [api_key]

    def _find(self, model: str) -> str | None:
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

    def make_sut(self, sut_definition: SUTDefinition) -> TogetherChatSUT:
        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        model = sut_metadata.external_model_name().lower()
        model_name = self._find(model)
        if model_name is None:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together serverless."
            )
        return TogetherChatSUT(
            sut_definition.dynamic_uid,
            sut_metadata.external_model_name(),
            *self.injected_secrets(),
        )


class TogetherDedicatedSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "together-dedicated"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None

    @property
    def client(self) -> Together:
        if self._client is None:
            api_key = self.injected_secrets()[0]
            self._client = Together(api_key=api_key.value)
        return self._client

    @client.setter
    def client(self, value: Together) -> None:
        self._client = value

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(TogetherApiKey)
        return [api_key]

    def _find(self, model: str) -> str | None:
        try:
            endpoints = self.client.endpoints.list(type="dedicated", mine=True)
            for endpoint in endpoints.data:
                if endpoint.model.lower() == model:
                    return endpoint.name
        except Exception as e:
            logger.error(f"Error looking up dedicated endpoints for {model} on together: {e}")
        return None

    def make_sut(self, sut_definition: SUTDefinition) -> TogetherDedicatedChatSUT:
        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        model = sut_metadata.external_model_name().lower()
        endpoint_name = self._find(model)
        if endpoint_name is None:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found on together dedicated endpoints."
            )
        return TogetherDedicatedChatSUT(
            sut_definition.dynamic_uid,
            endpoint_name,
            *self.injected_secrets(),
        )
