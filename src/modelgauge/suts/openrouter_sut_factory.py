from modelgauge.secret_values import RawSecrets
from modelgauge.suts.openai_sut_factory import OPENAI_SUT_FACTORIES, OpenAIGenericSUTFactory

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterSUTFactory(OpenAIGenericSUTFactory):

    def __init__(self, raw_secrets: RawSecrets, base_url: str | None = OPENROUTER_BASE_URL):
        super().__init__(raw_secrets)
        self.provider = "openrouter"
        self.base_url = base_url


OPENAI_SUT_FACTORIES["openrouter"] = OpenRouterSUTFactory
