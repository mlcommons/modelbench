from openai import OpenAI, NotFoundError

from modelgauge.auth.openai_compatible_secrets import (
    OpenAIApiKey,
    OpenAIOrganization,
    OpenAICompatibleApiKey,
    OpenAICompatibleBaseURL,
)
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.suts.openai_client import OpenAIChat

DRIVER_NAME = "openai"
NUM_RETRIES = 7


class OpenAICompatibleSUTFactory(DynamicSUTFactory):

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)

    def get_secrets(self) -> list[InjectSecret]:
        pass

    def make_sut(self, sut_metadata: DynamicSUTMetadata) -> OpenAIChat:
        if not sut_metadata.provider:
            factory = OpenAISUTFactory(self.raw_secrets)
        else:
            factory = OpenAIGenericSUTFactory(self.raw_secrets)
            factory.provider = sut_metadata.provider
        return factory.make_sut(sut_metadata)


class OpenAISUTFactory(DynamicSUTFactory):
    """A SUT that uses the OpenAI client, hosted anywhere"""

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None

    @property
    def client(self) -> OpenAI:
        if not self._client:
            api_key, organization = self.injected_secrets()
            self._client = OpenAI(api_key=api_key.value, organization=organization.value, max_retries=NUM_RETRIES)
        return self._client

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(OpenAIApiKey)
        organization = InjectSecret(OpenAIOrganization)
        return [api_key, organization]

    def _model_exists(self, sut_metadata: DynamicSUTMetadata):
        try:
            self.client.models.retrieve(sut_metadata.model)
        except NotFoundError:
            return False
        return True

    def make_sut(self, sut_metadata: DynamicSUTMetadata) -> OpenAIChat:
        if not self._model_exists(sut_metadata):
            raise ModelNotSupportedError(f"Model {sut_metadata.model} not found or not available on openai.")
        return OpenAIChat(DynamicSUTMetadata.make_sut_uid(sut_metadata), sut_metadata.model, client=self.client)


class OpenAIGenericSUTFactory(DynamicSUTFactory):
    """A SUT that uses the OpenAI client, hosted by openai. The required secrets are different."""

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self.provider: str | None = ""  # require to load the correct secrets
        self._client = None

    @property
    def client(self) -> OpenAI:
        if not self._client:
            api_key, base_url = self.injected_secrets()
            self._client = OpenAI(api_key=api_key.value, base_url=base_url.value, max_retries=NUM_RETRIES)
        return self._client

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(OpenAICompatibleApiKey.for_provider(provider=self.provider))
        base_url = InjectSecret(OpenAICompatibleBaseURL.for_provider(provider=self.provider))
        return [api_key, base_url]

    def make_sut(self, sut_metadata: DynamicSUTMetadata) -> OpenAIChat:
        self.provider = sut_metadata.provider
        return OpenAIChat(DynamicSUTMetadata.make_sut_uid(sut_metadata), sut_metadata.model, client=self.client)
