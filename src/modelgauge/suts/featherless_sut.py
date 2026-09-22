from openai import OpenAI

from modelgauge.auth.openai_compatible_secrets import OpenAICompatibleApiKey
from modelgauge.dynamic_sut_factory import (
    DynamicDriverSUTFactory,
    ModelNotSupportedError,
)
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
    ProducesPerTokenLogProbabilities,
)
from modelgauge.sut_definition import SUTDefinition
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.suts.openai_client import OpenAIChatSUT
from modelgauge.suts.openai_sut_factory import NUM_RETRIES, BaseOpenAISUTFactory

FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class FeatherlessSUT(OpenAIChatSUT):
    """
    Documented at https://featherless.ai/docs/api-overview-and-common-options
    """


class FeatherlessSUTFactory(BaseOpenAISUTFactory, DynamicDriverSUTFactory):
    DRIVER_NAME = "featherless"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self.provider = "featherless"
        self.base_url = FEATHERLESS_BASE_URL

    def get_secrets(self) -> list[InjectSecret]:
        return [InjectSecret(OpenAICompatibleApiKey.for_provider("featherless"))]

    def _make_client(self) -> OpenAI:
        [api_key] = self.injected_secrets()
        return OpenAI(api_key=api_key.value, base_url=self.base_url, max_retries=NUM_RETRIES)

    def _canonical_model_name(self, model_name: str) -> str | None:
        try:
            data = self.client.models.list().data
        except Exception:
            return None
        for entry in data:
            if entry.id.lower() == model_name.lower():
                return entry.id
        return None

    def make_sut(self, sut_definition: SUTDefinition) -> SUT:
        model_name = sut_definition.external_model_name()
        canonical_name = self._canonical_model_name(model_name)
        if not canonical_name:
            raise ModelNotSupportedError(f"Model {model_name} not found or not available on featherless.")
        return FeatherlessSUT(sut_definition.uid, canonical_name, client=self.client)

    def list_suts(self) -> list[SUTDefinition] | None:
        data = self.client.models.list().data
        result = []
        for entry in data:
            maker, model = entry.id.split("/", 1)
            result.append(
                SUTDefinition(
                    driver=self.DRIVER_NAME,
                    maker=maker,
                    model=model,
                )
            )
        return result
