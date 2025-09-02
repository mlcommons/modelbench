from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Dict, List, Optional

from huggingface_hub import get_inference_endpoint, InferenceClient, InferenceEndpointStatus  # type: ignore
from huggingface_hub.utils import HfHubHTTPError  # type: ignore
from pydantic import BaseModel
from tenacity import retry, TryAgain, stop_after_attempt, wait_random_exponential
from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.prompt import TextPrompt, ChatPrompt
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTOptions, SUTResponse, TokenProbability, TopTokens
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt, ProducesPerTokenLogProbabilities
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

HUGGING_FACE_TIMEOUT = 60 * 20


class ChatMessage(BaseModel):
    content: str
    role: str


class HuggingFaceChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class HuggingFaceChatCompletionOutput(BaseModel):
    choices: List[Dict]
    created: Optional[int] = None
    id: Optional[str] = None
    model: Optional[str] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[Dict] = None


class BaseHuggingFaceChatCompletionSUT(
    PromptResponseSUT[HuggingFaceChatCompletionRequest, HuggingFaceChatCompletionOutput], ABC
):
    """A Huggingface SUT that uses the chat_completion API."""

    def __init__(self, uid: str, token: HuggingFaceInferenceToken):
        super().__init__(uid)
        self.token = token
        self.client: InferenceClient | None = None

    @abstractmethod
    def _create_client(self) -> InferenceClient:
        """Create the InferenceClient for the SUT. Must be implemented by subclasses."""
        pass

    @retry(stop=stop_after_attempt(7), wait=wait_random_exponential())
    def evaluate(self, request: HuggingFaceChatCompletionRequest) -> HuggingFaceChatCompletionOutput:
        if self.client is None:
            self.client = self._create_client()

        try:
            request_dict = request.model_dump(exclude_none=True)
            response = self.client.chat_completion(**request_dict)  # type: ignore
            # Convert to cacheable pydantic object.
            return HuggingFaceChatCompletionOutput(
                choices=[asdict(choice) for choice in response.choices],
                created=response.created,
                id=response.id,
                model=response.model,
                system_fingerprint=response.system_fingerprint,
                usage=asdict(response.usage),
            )
        except HfHubHTTPError as hf_error:
            if hf_error.response.status_code >= 500:
                raise TryAgain
            else:
                raise
        except Exception as other_error:
            raise

    def translate_response(
        self, request: HuggingFaceChatCompletionRequest, response: HuggingFaceChatCompletionOutput
    ) -> SUTResponse:
        assert len(response.choices) == 1, f"Expected a single response message, got {len(response.choices)}."
        choice = response.choices[0]
        text = choice["message"]["content"]
        assert text is not None
        logprobs: Optional[List[TopTokens]] = None
        if request.logprobs:
            logprobs = []
            assert choice["logprobs"] is not None, "Expected logprobs, but not returned."
            lobprobs_sequence = choice["logprobs"]["content"]
            for token in lobprobs_sequence:
                top_tokens = []
                for top_logprob in token["top_logprobs"]:
                    top_tokens.append(TokenProbability(token=top_logprob["token"], logprob=top_logprob["logprob"]))
                logprobs.append(TopTokens(top_tokens=top_tokens))
        return SUTResponse(text=text, top_logprobs=logprobs)


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt, ProducesPerTokenLogProbabilities])
class HuggingFaceChatCompletionDedicatedSUT(BaseHuggingFaceChatCompletionSUT):
    """A Hugging Face SUT that is hosted on a dedicated inference endpoint and uses the chat_completion API."""

    def __init__(self, uid: str, inference_endpoint: str, token: HuggingFaceInferenceToken):
        super().__init__(uid, token)
        self.inference_endpoint = inference_endpoint

    def _create_client(self):
        endpoint = get_inference_endpoint(self.inference_endpoint, token=self.token.value)

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

        return InferenceClient(base_url=endpoint.url, token=self.token.value)

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> HuggingFaceChatCompletionRequest:
        logprobs = None
        if options.top_logprobs is not None:
            logprobs = True
        return HuggingFaceChatCompletionRequest(
            messages=[ChatMessage(role="user", content=prompt.text)],
            logprobs=logprobs,
            **options.model_dump(),
        )

    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> HuggingFaceChatCompletionRequest:
        logprobs = None
        if options.top_logprobs is not None:
            logprobs = True
        return HuggingFaceChatCompletionRequest(
            messages=[ChatMessage(role=p.role.lower(), content=p.text) for p in prompt.messages],
            logprobs=logprobs,
            **options.model_dump(),
        )


@modelgauge_sut(capabilities=[AcceptsChatPrompt, AcceptsTextPrompt, ProducesPerTokenLogProbabilities])
class HuggingFaceChatCompletionServerlessSUT(BaseHuggingFaceChatCompletionSUT):
    """A SUT hosted by an inference provider on huggingface."""

    def __init__(self, uid: str, model: str, provider: str, token: HuggingFaceInferenceToken):
        super().__init__(uid, token)
        self.model = model
        self.provider = provider

    def _create_client(self):
        return InferenceClient(
            provider=self.provider,
            api_key=self.token.value,
        )

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> HuggingFaceChatCompletionRequest:
        logprobs = None
        if options.top_logprobs is not None:
            logprobs = True
        return HuggingFaceChatCompletionRequest(
            model=self.model,
            messages=[ChatMessage(role="user", content=prompt.text)],
            logprobs=logprobs,
            **options.model_dump(),
        )

    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> HuggingFaceChatCompletionRequest:
        messages = []
        for message in prompt.messages:
            messages.append(ChatMessage(content=message.text, role=message.role.lower()))
        return HuggingFaceChatCompletionRequest(
            model=self.model,
            messages=messages,
            **options.model_dump(),
        )


HF_SECRET = InjectSecret(HuggingFaceInferenceToken)

SUTS.register(
    HuggingFaceChatCompletionDedicatedSUT,
    "nvidia-llama-3-1-nemotron-nano-8b-v1",
    "llama-3-1-nemotron-nano-8b-v-uhu",
    HF_SECRET,
)

DEDICATED_SUTS_AND_SERVERS = {
    "athene-v2-chat": "ijf",
    "aya-expanse-8b": "rfn",
    "gemma-2-9b-it": "plf",
    "gemma-2-9b-it-simpo": "zop",
    "gemma-3-12b-it": "mxk",
    "granite-3-1-8b-instruct": "twe",
    "llama-3-1-tulu-3-8b": "bzk",  # check
    "llama-3-1-tulu-3-70b": "ome",
    "mistral-nemo-instruct-2407": "mgt",
    "mixtral-8x22b-instruct-v0-1": "kog",
    "olmo-2-1124-13b-instruct": "ibo",
    "olmo-2-0325-32b-instruct": "yft",
    "qwen1-5-110b-chat": "gek",
    "qwen2-5-7b-instruct": "hgy",
    "qwq-32b": "usw",
    "yi-1-5-34b-chat": "nlm",  # check
}

for sut, endpoint in DEDICATED_SUTS_AND_SERVERS.items():
    SUTS.register(
        HuggingFaceChatCompletionDedicatedSUT,
        sut + "-hf",
        sut + "-" + endpoint,
        HF_SECRET,
    )

SUTS.register(
    HuggingFaceChatCompletionServerlessSUT,
    "cohere-c4ai-command-a-03-2025-hf",
    "CohereLabs/c4ai-command-a-03-2025",
    "cohere",
    HF_SECRET,
)

SUTS.register(
    HuggingFaceChatCompletionServerlessSUT,
    "meta-llama-3_1-8b-instruct-hf-nebius",
    "meta-llama/Llama-3.1-8B-Instruct",
    "nebius",
    HF_SECRET,
)

SUTS.register(
    HuggingFaceChatCompletionServerlessSUT,
    "google-gemma-3-27b-it-hf-nebius",
    "google/gemma-3-27b-it",
    "nebius",
    HF_SECRET,
)
