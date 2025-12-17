from typing import Optional

from mistralai.models import ChatCompletionResponse, ClassificationResponse, SDKError
from pydantic import BaseModel

from modelgauge.prompt import TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTOptions, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.suts.mistral_client import MistralAIAPIKey, MistralAIClient

_USER_ROLE = "user"


class MistralAIRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: Optional[float] = None
    max_tokens: Optional[int]
    n: int = 1  # Number of completions to generate.


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
        api_key: MistralAIAPIKey,
    ):
        super().__init__(uid)
        self.model_name = model_name
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if not self._client:
            self._client = MistralAIClient(self.model_name, self._api_key)
        return self._client

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> MistralAIRequest:
        args = {"model": self.model_name, "messages": [{"role": _USER_ROLE, "content": prompt.text}]}
        if options.temperature is not None:
            args["temperature"] = options.temperature
        if options.max_tokens is not None:
            args["max_tokens"] = options.max_tokens
        return MistralAIRequest(**args)

    @retry(transient_exceptions=[SDKError])
    def evaluate(self, request: MistralAIRequest) -> ChatCompletionResponse:
        response = self.client.request(request.model_dump(exclude_none=True))  # type: ignore
        return response

    def translate_response(self, request: MistralAIRequest, response: MistralAIResponse) -> SUTResponse:
        assert len(response.choices) == 1, f"Expected 1 completion, got {len(response.choices)}."
        text = response.choices[0].message.content
        assert text is not None
        return SUTResponse(text=str(text))


class MistralAIResponseWithModerations(BaseModel):
    """Mistral's ChatCompletionResponse object + moderation scores."""

    response: ChatCompletionResponse  # Contains multiple completions.
    moderations: dict[int, ClassificationResponse]  # Keys correspond to a choice's index field.


def register_suts_for_model(model_name):
    MISTRAL_API_KEY = InjectSecret(MistralAIAPIKey)
    # Register standard SUT.
    model_uid = f"mistralai-{model_name}"
    SUTS.register(MistralAISut, model_uid, model_name, MISTRAL_API_KEY)


register_suts_for_model("ministral-8b-2410")
register_suts_for_model("mistral-large-2411")
register_suts_for_model("mistral-large-2402")
