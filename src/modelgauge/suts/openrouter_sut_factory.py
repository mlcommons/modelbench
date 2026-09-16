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
from modelgauge.suts.openai_client import OpenAIResponsesSUT
from modelgauge.suts.openai_sut_factory import NUM_RETRIES, BaseOpenAISUTFactory

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class OpenRouterSUT(OpenAIResponsesSUT):
    """
    Documented at https://openrouter.ai/docs
    """


class OpenRouterSUTFactory(BaseOpenAISUTFactory, DynamicDriverSUTFactory):
    DRIVER_NAME = "openrouter"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self.provider = "openrouter"
        self.base_url = OPENROUTER_BASE_URL

    def get_secrets(self) -> list[InjectSecret]:
        return [InjectSecret(OpenAICompatibleApiKey.for_provider("openrouter"))]

    def _make_client(self) -> OpenAI:
        [api_key] = self.injected_secrets()
        return OpenAI(api_key=api_key.value, base_url=self.base_url, max_retries=NUM_RETRIES)

    def _model_exists(self, model_name: str) -> bool:
        try:
            data = self.client.models.list().data
        except Exception:
            return False
        return any(entry.id == model_name for entry in data)

    def make_sut(self, sut_definition: SUTDefinition) -> SUT:
        model_name = sut_definition.external_model_name()
        if not self._model_exists(model_name):
            raise ModelNotSupportedError(f"Model {model_name} not found or not available on openrouter.")
        return OpenRouterSUT(sut_definition.uid, model_name, client=self.client)

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
