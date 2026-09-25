from openai import OpenAI

from modelgauge.auth.openai_compatible_secrets import OpenAICompatibleApiKey
from modelgauge.dynamic_sut_factory import (
    DynamicSUTFactory,
    ModelNotSupportedError,
)
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.openai_client import OpenAIResponsesSUT

NUM_RETRIES = 7


class BaseOpenAISUTFactory(DynamicSUTFactory):
    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self.provider = None  # must be set in child classes and  match name of section (scope) in secrets.toml
        self._client = None

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(OpenAICompatibleApiKey.for_provider(provider=self.provider))
        return [api_key]

    @property
    def client(self) -> OpenAI:
        if not self._client:
            self._client = self._make_client()
        return self._client

    def _make_client(self) -> OpenAI:
        [api_key] = self.injected_secrets()
        _client = OpenAI(api_key=api_key.value, max_retries=NUM_RETRIES)
        return _client


class OpenAISUTFactory(BaseOpenAISUTFactory):
    """OpenAI SUT hosted by OpenAI"""

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self.provider = "openai"

    def _model_exists(self, sut_definition: SUTDefinition):
        try:
            self.client.models.retrieve(sut_definition.get("model"))  # type: ignore
        except:
            return False
        return True

    def make_sut(self, sut_definition: SUTDefinition) -> OpenAIResponsesSUT:
        if not self._model_exists(sut_definition):
            raise ModelNotSupportedError(
                f"Model {sut_definition.external_model_name()} not found or not available on openai."
            )
        return OpenAIResponsesSUT(sut_definition.uid, sut_definition.get("model"), client=self.client)  # type: ignore
