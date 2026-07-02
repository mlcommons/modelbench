import logging

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
        self.serverless_factory = TogetherServerlessSUTFactory(raw_secrets)
        self.dedicated_factory = TogetherDedicatedSUTFactory(raw_secrets)

    def get_secrets(self) -> list[InjectSecret]:
        return []

    def make_sut(self, sut_definition: SUTDefinition) -> PromptResponseSUT:
        try:
            return self.serverless_factory.make_sut(sut_definition)
        except ModelNotSupportedError:
            # is there a dedicated option? probably not, but we check anyway
            try:
                return self.dedicated_factory.make_sut(sut_definition)
            except ModelNotSupportedError:
                raise ModelNotSupportedError(
                    f"Together doesn't know model {sut_definition.external_model_name()}, or you need credentials for its repo."
                )


class TogetherServerlessSUTFactory(DynamicSUTFactory):

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None  # Lazy load.

    @property
    def client(self) -> Together:
        if self._client is None:
            api_key = self.injected_secrets()[0]
            self._client = Together(api_key=api_key.value)
        return self._client

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(TogetherApiKey)
        return [api_key]

    def _find(self, sut_metadata: DynamicSUTMetadata):
        try:
            model = sut_metadata.external_model_name().lower()
            self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Anybody home?"},
                ],
            )
        except Exception as e:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together: {e}"
            )

        return model

    def make_sut(self, sut_definition: SUTDefinition) -> TogetherChatSUT:
        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        model_name = self._find(sut_metadata)
        if not model_name:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together."
            )

        return TogetherChatSUT(
            sut_definition.dynamic_uid,
            sut_metadata.external_model_name(),
            *self.injected_secrets(),
        )


class TogetherDedicatedSUTFactory(DynamicSUTFactory):

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None  # Lazy load.

    @property
    def client(self) -> Together:
        if self._client is None:
            api_key = self.injected_secrets()[0]
            self._client = Together(api_key=api_key.value)
        return self._client

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(TogetherApiKey)
        return [api_key]

    def _find(self, sut_metadata: DynamicSUTMetadata):
        model = sut_metadata.external_model_name().lower()
        try:
            endpoints = self.client.endpoints.list(type="dedicated", mine=True)
            for endpoint in endpoints.data:
                if endpoint.model.lower() == model:
                    return endpoint.name
        except Exception as e:
            logger.error(f"Error looking up dedicated endpoints for {model} on together: {e}")
        return None

    def make_sut(self, sut_definition: SUTDefinition) -> TogetherChatSUT:
        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        endpoint_name = self._find(sut_metadata)
        if not endpoint_name:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together dedicated endpoints."
            )

        return TogetherDedicatedChatSUT(
            sut_definition.dynamic_uid,
            endpoint_name,
            *self.injected_secrets(),
        )
