from openai import OpenAI, NotFoundError

from modelgauge.auth.openai_compatible_secrets import OpenAICompatibleApiKey
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError, ProviderNotFoundError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.suts.openai_client import OpenAIChat

DRIVER_NAME = "openai"
NUM_RETRIES = 7


class OpenAICompatibleSUTFactory(DynamicSUTFactory):

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self.provider: str = "openai"  # must match name of section (scope) in secrets.toml
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
        arguments = {
            "api_key": api_key.value,
            "max_retries": NUM_RETRIES,
        }
        _client = OpenAI(**arguments)
        return _client

    def make_sut(self, sut_metadata: DynamicSUTMetadata, **kwargs) -> OpenAIChat:
        self.provider = sut_metadata.provider  # type: ignore
        if not self.provider or self.provider == "openai":
            factory = OpenAISUTFactory(self.raw_secrets)
        else:
            factory_class = OPENAI_SUT_FACTORIES.get(self.provider, None)
            if not factory_class:
                raise ProviderNotFoundError(f"I don't know how to make a {self.provider} SUT with the OpenAI client")
            factory = factory_class(self.raw_secrets, **kwargs)
        return factory.make_sut(sut_metadata)


class OpenAISUTFactory(OpenAICompatibleSUTFactory):
    """OpenAI SUT hosted by OpenAI"""

    def _model_exists(self, sut_metadata: DynamicSUTMetadata):
        try:
            self.client.models.retrieve(sut_metadata.model)
        except NotFoundError as nfe:
            raise ModelNotSupportedError from nfe
        return True

    def make_sut(self, sut_metadata: DynamicSUTMetadata) -> OpenAIChat:
        if not self._model_exists(sut_metadata):
            raise ModelNotSupportedError(f"Model {sut_metadata.model} not found or not available on openai.")
        return OpenAIChat(DynamicSUTMetadata.make_sut_uid(sut_metadata), sut_metadata.model, client=self.client)


class OpenAIGenericSUTFactory(OpenAICompatibleSUTFactory):
    """A SUT that uses the OpenAI client, not hosted by OpenAI"""

    def __init__(self, raw_secrets: RawSecrets, base_url: str | None = None):
        super().__init__(raw_secrets)
        self.base_url = base_url

    def _make_client(self):
        assert self.base_url
        [api_key] = self.injected_secrets()
        arguments = {"api_key": api_key.value, "max_retries": NUM_RETRIES, "base_url": self.base_url}
        _client = OpenAI(**arguments)
        return _client

    def make_sut(self, sut_metadata: DynamicSUTMetadata, **kwargs) -> OpenAIChat:
        base_url = kwargs.get("base_url", None)
        self.provider = sut_metadata.provider  # type: ignore
        if base_url:
            self.base_url = base_url
        return OpenAIChat(DynamicSUTMetadata.make_sut_uid(sut_metadata), sut_metadata.model, client=self.client)


# this is how you add a new OpenAI-compatible SUT
class DemoOpenAICompatibleSUTFactory(OpenAIGenericSUTFactory):
    def __init__(self, raw_secrets, **kwargs):
        super().__init__(raw_secrets)
        self.provider = "demo"
        self.base_url = "https://example.net/v1/"

        # the SUT UID is maker/model:demo:openai
        # the credentials in secrets.toml must be:
        # [demo]
        # api_key = "abcd"


OPENAI_SUT_FACTORIES: dict = {"demo": DemoOpenAICompatibleSUTFactory}
