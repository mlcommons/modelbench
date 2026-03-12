from modelgauge.dynamic_sut_factory import DynamicSUTFactoryDriver, ModelNotSupportedError
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.mistral_client import MistralAIAPIKey, MistralAIClient
from modelgauge.suts.mistral_sut import MistralAISut


class MistralSUTFactory(DynamicSUTFactoryDriver):
    DRIVER_NAME = "mistral"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None  # Lazy load.

    @property
    def client(self) -> MistralAIClient:
        if self._client is None:
            api_key = self.injected_secrets()[0]
            self._client = MistralAIClient(api_key)
        return self._client

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(MistralAIAPIKey)
        return [api_key]

    def make_sut(self, sut_definition: SUTDefinition) -> SUT:
        model_name = sut_definition.to_dynamic_sut_metadata().external_model_name()

        try:
            self.client.model_info(model_name)
        except Exception as e:
            raise ModelNotSupportedError(f"Model {model_name} not found or not available on mistral: {e}")

        return MistralAISut(sut_definition.dynamic_uid, model_name, *self.injected_secrets())
