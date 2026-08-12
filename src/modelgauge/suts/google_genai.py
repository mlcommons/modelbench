"""
This file defines google SUTs that use Google's genai python SDK.
"""

from typing import Optional

from airrlogger.log_config import get_logger
from google import genai
from google.api_core.exceptions import InternalServerError, ResourceExhausted, RetryError, TooManyRequests
from google.genai.types import FinishReason, GenerateContentConfig, GenerateContentResponse, ThinkingConfig
from pydantic import BaseModel

from modelgauge.general import APIException
from modelgauge.model_options import ModelOptions
from modelgauge.prompt import TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import RequiredSecret, SecretDescription
from modelgauge.sut import REFUSAL_RESPONSE, PromptResponseSUT, SUTResponse  # usort: skip
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut

logger = get_logger(__name__)


class GoogleAiApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="google_ai",
            key="api_key",
            instructions="See https://aistudio.google.com/app/apikey",
        )


# Taken from https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerateContentResponse#finishreason
GOOGLE_REFUSAL_FINISH_REASONS = [
    FinishReason.SAFETY,
    FinishReason.RECITATION,
    FinishReason.BLOCKLIST,
    FinishReason.PROHIBITED_CONTENT,
    FinishReason.SPII,
    FinishReason.OTHER,
]


class GenAiRequest(BaseModel):
    model: str
    contents: str
    config: Optional[GenerateContentConfig] = None


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class GoogleGenAiSUT(PromptResponseSUT):
    def __init__(self, uid: str, model_name: str, use_reasoning: Optional[bool], client: genai.Client):
        super().__init__(uid)
        self.model_name = model_name
        self.client = client
        self.model_info = self.client.models.get(model=model_name)
        self.reasoning_available = self.model_info.thinking
        if use_reasoning and not self.reasoning_available:
            raise ValueError(f"Reasoning requested but reasoning is not available: {self.model_info}")
        self.use_reasoning = use_reasoning

    def translate_text_prompt(self, prompt: TextPrompt, options: ModelOptions) -> GenAiRequest:
        optional = {}
        if self.reasoning_available and self.use_reasoning == False:
            optional["thinking_config"] = ThinkingConfig(
                thinking_budget=0,  # Turn off reasoning.
            )
        generation_config = GenerateContentConfig(
            stop_sequences=options.stop_sequences,
            max_output_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k_per_token,
            presence_penalty=options.presence_penalty,
            frequency_penalty=options.frequency_penalty,
            **optional,
        )
        return GenAiRequest(
            model=self.model_name,
            contents=prompt.text,
            config=generation_config,
        )

    @retry(transient_exceptions=[InternalServerError, ResourceExhausted, RetryError, TooManyRequests])
    def evaluate(self, request: GenAiRequest) -> GenerateContentResponse:
        return self.client.models.generate_content(**request.model_dump(exclude_none=True))

    def translate_response(self, request: GenAiRequest, response: GenerateContentResponse) -> SUTResponse:
        if response.candidates is None or len(response.candidates) == 0:
            # This is apparently a refusal. At least, it's what happens consistently with a set of
            # prompts in the CSE, SRC, and SXC hazards
            response_text = REFUSAL_RESPONSE
        elif len(response.candidates) == 1:
            candidate = response.candidates[0]
            if candidate.finish_reason in GOOGLE_REFUSAL_FINISH_REASONS + ["OTHER"]:
                response_text = REFUSAL_RESPONSE
            elif candidate.content is not None:
                response_text = candidate.content.parts[0].text
            else:
                raise APIException(
                    f"Unexpected candidate in response from GoogleGenAiSUT {self.uid}: {candidate}. "
                    f"The candidate does not have any content,"
                    f" but it's finish reason {candidate.finish_reason} does not qualify as a refusal."
                )
        else:
            raise AssertionError(f"Expected a single candidate in the response, got {response.candidates}.")
        return SUTResponse(text=response_text)

