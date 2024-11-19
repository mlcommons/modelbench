from mistralai.models import ChatCompletionResponse
from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.suts.mistral_client import MistralAIAPIKey, MistralAIClient

from pydantic import BaseModel

_USER_ROLE = "user"


class MistralAIRequest(BaseModel):
    model: str
    messages: list[dict]


class MistralAIResponse(ChatCompletionResponse):
    """The ChatCompletionResponse class from Mistral matches our Response
    objects now, but we subclass it for consistency and so we can adjust it
    in case the upstream object changes."""

    pass


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class MistralAISut(PromptResponseSUT):
    """A MistralAI SUT hosted on MistralAI."""

    def __init__(
        self,
        uid: str,
        model_name: str,
        model_version: str,
        api_key: MistralAIAPIKey,
    ):
        super().__init__(uid)
        self.model_name = model_name
        self.model_version = model_version
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if not self._client:
            self._client = MistralAIClient(self.model_name, self._api_key)
        return self._client

    def translate_text_prompt(self, prompt: TextPrompt) -> MistralAIRequest:
        return MistralAIRequest(model=self.model_name, messages=[{"role": _USER_ROLE, "content": prompt.text}])

    def evaluate(self, request: MistralAIRequest) -> ChatCompletionResponse:
        response = self.client.request(request.model_dump(exclude_none=True))  # type: ignore
        return response

    def translate_response(self, request: MistralAIRequest, response: MistralAIResponse) -> SUTResponse:
        completions = []
        for choice in response.choices:
            text = choice.message.content
            assert text is not None
            completions.append(SUTCompletion(text=str(text)))
        return SUTResponse(completions=completions)


MISTRAL_API_KEY = InjectSecret(MistralAIAPIKey)

model_name = "ministral-8b"
model_version = "latest"
model_uid = f"{model_name}-{model_version}"
SUTS.register(MistralAISut, model_uid, model_uid, model_version, MISTRAL_API_KEY)
