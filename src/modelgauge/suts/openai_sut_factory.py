from openai import OpenAI, NotFoundError

from modelgauge.auth.openai_compatible_secrets import OpenAICompatibleApiKey
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError, ProviderNotFoundError
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.openai_client import OpenAIChat

DRIVER_NAME = "openai"
NUM_RETRIES = 7


class OpenAICompatibleSUTFactory(DynamicSUTFactory):

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

    def make_sut(self, sut_definition: SUTDefinition) -> OpenAIChat:
        factory = factory_class = None
        self.provider = sut_definition.get("provider")  # type: ignore

        if not self.provider or self.provider == "openai":
            factory = OpenAISUTFactory(self.raw_secrets)
        else:
            factory_class = OPENAI_SUT_FACTORIES.get(self.provider, None)
            # we don't have a prebuilt factory...
            if not factory_class:
                # ... but maybe we have credentials and a base url, and we can try to make a SUT
                base_url = sut_definition.get("base_url", None)
                has_secret = self.provider in self.raw_secrets
                if base_url and has_secret:
                    factory_class = OpenAIGenericSUTFactory
            if factory_class:
                factory = factory_class(self.raw_secrets)
            else:
                raise ProviderNotFoundError(f"I don't know how to make a {self.provider} SUT with the OpenAI client")
        return factory.make_sut(sut_definition)


class OpenAISUTFactory(OpenAICompatibleSUTFactory):
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

    def make_sut(self, sut_definition: SUTDefinition) -> OpenAIChat:
        if not self._model_exists(sut_definition):
            raise ModelNotSupportedError(
                f"Model {sut_definition.external_model_name()} not found or not available on openai."
            )
        return OpenAIChat(sut_definition.uid, sut_definition.get("model"), client=self.client)  # type: ignore


class OpenAIGenericSUTFactory(OpenAICompatibleSUTFactory):
    """A SUT that uses the OpenAI client, not hosted by OpenAI"""

    def __init__(self, raw_secrets: RawSecrets, base_url: str | None = None):
        super().__init__(raw_secrets)
        self.base_url = base_url

    def _make_client(self):
        assert self.base_url
        [api_key] = self.injected_secrets()
        _client = OpenAI(api_key=api_key.value, base_url=self.base_url, max_retries=NUM_RETRIES)
        return _client

    def make_sut(self, sut_definition: SUTDefinition, base_url: str | None = None) -> OpenAIChat:
        the_base_url = sut_definition.get("base_url", None)
        if base_url:
            the_base_url = base_url
        self.provider = sut_definition.get("provider")  # type: ignore
        if the_base_url:
            self.base_url = the_base_url
        return OpenAIChat(sut_definition.uid, sut_definition.get("model"), client=self.client)  # type: ignore


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


class ModelShipOpenAICompatibleSUTFactory(OpenAIGenericSUTFactory):
    def __init__(self, raw_secrets, **kwargs):
        super().__init__(raw_secrets)
        self.provider = "modelship"
        self.base_url = "http://mlc2:8123/v1/"


OPENAI_SUT_FACTORIES: dict = {"demo": DemoOpenAICompatibleSUTFactory, "modelship": ModelShipOpenAICompatibleSUTFactory}
