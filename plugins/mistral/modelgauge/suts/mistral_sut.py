import warnings
from typing import Optional

from mistralai.models import ChatCompletionResponse, ClassificationResponse, SDKError
from modelgauge.prompt import TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.suts.mistral_client import MistralAIAPIKey, MistralAIClient

from pydantic import BaseModel

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

    def translate_text_prompt(self, prompt: TextPrompt) -> MistralAIRequest:
        args = {"model": self.model_name, "messages": [{"role": _USER_ROLE, "content": prompt.text}]}
        if prompt.options.temperature is not None:
            args["temperature"] = prompt.options.temperature
        if prompt.options.max_tokens is not None:
            args["max_tokens"] = prompt.options.max_tokens
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


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class MistralAIModeratedSut(PromptResponseSUT):
    """A MistralAI SUT hosted on MistralAI with safety moderation based on the following recipe:
    https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/moderation/system-level-guardrails.ipynb#scrollTo=OlnXFkLo5sKX
    """

    def __init__(
        self,
        uid: str,
        model_name: str,
        moderation_model_name: str,
        num_generations: int,
        temperature: float,
        threshold: float,
        api_key: MistralAIAPIKey,
    ):
        assert (
            num_generations > 1
        ), "The moderation strategy uses a sampling-based mechanism. num_generations should be greater than 1."
        super().__init__(uid)
        self.model_name = model_name
        self.moderation_model_name = moderation_model_name
        self.num_generations = num_generations
        self.temperature = temperature
        self.threshold = threshold
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if not self._client:
            self._client = MistralAIClient(self.model_name, self._api_key)
        return self._client

    def translate_text_prompt(self, prompt: TextPrompt) -> MistralAIRequest:
        # Warn if prompt options are overridden.
        if prompt.options.temperature is not None and prompt.options.temperature != self.temperature:
            warnings.warn(
                f"This SUT overrides the prompt's temperature value of {prompt.options.temperature} to {self.temperature}."
            )

        args = {"model": self.model_name, "messages": [{"role": _USER_ROLE, "content": prompt.text}]}
        if prompt.options.max_tokens is not None:
            args["max_tokens"] = prompt.options.max_tokens
        return MistralAIRequest(temperature=self.temperature, n=self.num_generations, **args)

    @retry(transient_exceptions=[SDKError])
    def evaluate(self, request: MistralAIRequest) -> MistralAIResponseWithModerations:
        response = self.client.request(request.model_dump(exclude_none=True))  # type: ignore
        assert (
            len(response.choices) == self.num_generations
        ), f"Expected {self.num_generations} completions, got {len(response.choices)}."
        moderations = {}
        for choice in response.choices:
            scores = self.client.score_conversation(
                self.moderation_model_name, request.messages[0]["content"], choice.message.content
            )
            moderations[choice.index] = scores
        return MistralAIResponseWithModerations(response=response, moderations=moderations)

    def translate_response(self, request: MistralAIRequest, response: MistralAIResponseWithModerations) -> SUTResponse:
        # Sort list of completions by moderation score.
        scores_list = []
        for choice in response.response.choices:
            text = choice.message.content
            assert text is not None
            scores = response.moderations[choice.index]
            assert len(scores.results) == 1, f"Expected 1 moderation object, got {len(scores.results)}."
            category_scores = scores.results[0].category_scores
            max_score = max(category_scores.values())
            scores_list.append((text, max_score))

        sorted_responses = sorted(scores_list, key=lambda x: x[1])
        lowest_score = sorted_responses[0][1]
        if lowest_score >= self.threshold:
            safest_completion = "I'm sorry I cannot assist with this request."
        else:
            safest_completion = str(sorted_responses[0][0])
        return SUTResponse(text=str(safest_completion))


def register_suts_for_model(model_name):
    MISTRAL_API_KEY = InjectSecret(MistralAIAPIKey)
    # Register standard SUT.
    model_uid = f"mistralai-{model_name}"
    SUTS.register(MistralAISut, model_uid, model_name, MISTRAL_API_KEY)
    # Register moderated SUT.
    moderation_model_name = "mistral-moderation-latest"
    num_generations = 5
    temperature = 0.3
    threshold = 0.2
    moderated_model_uid = f"mistralai-{model_name}-moderated"
    SUTS.register(
        MistralAIModeratedSut,
        moderated_model_uid,
        model_name,
        moderation_model_name,
        num_generations,
        temperature,
        threshold,
        MISTRAL_API_KEY,
    )


register_suts_for_model("ministral-8b-2410")
register_suts_for_model("mistral-large-2411")
