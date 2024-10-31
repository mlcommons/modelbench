from typing import List, Optional

from huggingface_hub import (  # type: ignore
    ChatCompletionOutput,
    get_inference_endpoint,
    InferenceClient,
    InferenceEndpointStatus,
)
from huggingface_hub.utils import HfHubHTTPError  # type: ignore
from pydantic import BaseModel

from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse, TokenProbability, TopTokens
from modelgauge.sut_capabilities import AcceptsTextPrompt, ProducesPerTokenLogProbabilities
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

HUGGING_FACE_TIMEOUT = 60 * 20


class ChatMessage(BaseModel):
    content: str
    role: str


class HuggingFaceChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    logprobs: bool
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


@modelgauge_sut(capabilities=[AcceptsTextPrompt, ProducesPerTokenLogProbabilities])
class HuggingFaceChatCompletionSUT(PromptResponseSUT[HuggingFaceChatCompletionRequest, ChatCompletionOutput]):
    """A Hugging Face SUT that is hosted on a dedicated inference endpoint and uses the chat_completion API."""

    def __init__(self, uid: str, inference_endpoint: str, token: HuggingFaceInferenceToken):
        super().__init__(uid)
        self.token = token
        self.inference_endpoint = inference_endpoint
        self.client = None

    def _create_client(self, inference_endpoint: str, token: str):
        endpoint = get_inference_endpoint(inference_endpoint, token=token)

        if endpoint.status in [
            InferenceEndpointStatus.PENDING,
            InferenceEndpointStatus.INITIALIZING,
            InferenceEndpointStatus.UPDATING,
        ]:
            print(f"Endpoint starting. Status: {endpoint.status}. Waiting up to {HUGGING_FACE_TIMEOUT}s to start.")
            endpoint.wait(HUGGING_FACE_TIMEOUT)
        elif endpoint.status == InferenceEndpointStatus.SCALED_TO_ZERO:
            print("Endpoint scaled to zero... requesting to resume.")
            try:
                endpoint.resume(running_ok=True)
            except HfHubHTTPError:
                raise ConnectionError("Failed to resume endpoint. Please resume manually.")
            print(f"Requested resume. Waiting up to {HUGGING_FACE_TIMEOUT}s to start.")
            endpoint.wait(HUGGING_FACE_TIMEOUT)
        elif endpoint.status != InferenceEndpointStatus.RUNNING:
            raise ConnectionError(
                f"Endpoint is not running: Please contact admin to ensure endpoint is starting or running (status: {endpoint.status})"
            )
        return InferenceClient(base_url=endpoint.url, token=token)

    def translate_text_prompt(self, prompt: TextPrompt) -> HuggingFaceChatCompletionRequest:
        logprobs = False
        if prompt.options.top_logprobs is not None:
            logprobs = True
        return HuggingFaceChatCompletionRequest(
            messages=[ChatMessage(role="user", content=prompt.text)],
            logprobs=logprobs,
            **prompt.options.model_dump(),
        )

    def evaluate(self, request: HuggingFaceChatCompletionRequest) -> ChatCompletionOutput:
        if self.client is None:
            self.client = self._create_client(inference_endpoint=self.inference_endpoint, token=self.token.value)

        request_dict = request.model_dump(exclude_none=True)
        return self.client.chat_completion(**request_dict)  # type: ignore

    def translate_response(
        self, request: HuggingFaceChatCompletionRequest, response: ChatCompletionOutput
    ) -> SUTResponse:
        completions = []
        for choice in response.choices:
            text = choice.message.content
            assert text is not None
            logprobs: Optional[List[TopTokens]] = None
            if request.logprobs:
                logprobs = []
                assert choice.logprobs is not None, "Expected logprobs, but not returned."
                lobprobs_sequence = choice.logprobs.content
                for token in lobprobs_sequence:
                    top_tokens = []
                    for top_logprob in token.top_logprobs:
                        top_tokens.append(TokenProbability(token=top_logprob.token, logprob=top_logprob.logprob))
                    logprobs.append(TopTokens(top_tokens=top_tokens))

            completions.append(SUTCompletion(text=text, top_logprobs=logprobs))
        return SUTResponse(completions=completions)


HF_SECRET = InjectSecret(HuggingFaceInferenceToken)

SUTS.register(
    HuggingFaceChatCompletionSUT,
    "gemma-2-9b-it-hf",
    "gemma-2-9b-it-qfa",
    HF_SECRET,
)

SUTS.register(
    HuggingFaceChatCompletionSUT,
    "mistral-nemo-instruct-2407-hf",
    "mistral-nemo-instruct-2407-mgt",
    HF_SECRET,
)

SUTS.register(
    HuggingFaceChatCompletionSUT,
    "qwen2-5-7b-instruct-hf",
    "qwen2-5-7b-instruct-hgy",
    HF_SECRET,
)

SUTS.register(
    HuggingFaceChatCompletionSUT,
    "llama-3.1-8b-instruct-hf",
    "llama-3-1-8b-instruct-iti",
    HF_SECRET,
)
