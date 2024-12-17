from typing import List, Optional

from huggingface_hub import ChatCompletionOutput  # type: ignore

from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTCompletion, SUTResponse, TokenProbability, TopTokens
from modelgauge.sut_capabilities import AcceptsTextPrompt, ProducesPerTokenLogProbabilities
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

from .huggingface_chat_completion import (
    HuggingFaceChatCompletionSUT,
    HF_SECRET,
    HuggingFaceChatCompletionRequest,
    ChatMessage,
)


@modelgauge_sut(capabilities=[AcceptsTextPrompt, ProducesPerTokenLogProbabilities])
class HuggingFaceChatCompletionLGSUT(HuggingFaceChatCompletionSUT):
    """A Hugging Face SUT that is hosted on a dedicated inference endpoint and uses the chat_completion API.
    Also adds on an optional output classifier.
    """

    def __init__(self, uid: str, inference_endpoint: str, token: HuggingFaceInferenceToken, lg_inference_endpoint: str):
        super().__init__(uid, inference_endpoint, token)
        self.lg_inference_endpoint = lg_inference_endpoint
        self.lg_client = self._create_client(lg_inference_endpoint, token.value)

    def _call_lg_unsafe(self, orig_request: HuggingFaceChatCompletionRequest, model_response: str):
        try:
            request = orig_request.copy(deep=True)
            request.messages.append(ChatMessage(role="assistant", content=model_response))
            request_dict = request.model_dump(exclude_none=True)
            response = self.lg_client.chat_completion(**request_dict)
            return "unsafe" in response.choices[0].message.content
        except:
            return False

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

            # call llamaguard
            is_unsafe = self._call_lg_unsafe(request, text)
            if is_unsafe:
                text = "I can't answer that. Can I help with something else?"

            completions.append(SUTCompletion(text=text, top_logprobs=logprobs))
        return SUTResponse(completions=completions)


SUTS.register(
    HuggingFaceChatCompletionLGSUT,
    "llama-3.1-8b-instruct-lg-hf",
    "llama-3-1-8b-instruct-iti",
    HF_SECRET,
    "llama-guard-3-8b-gzj",
)
