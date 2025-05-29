"""
This file defines google SUTs that use Google's genai python SDK.
"""

from typing import Optional

from google import genai
from google.api_core.exceptions import (
    InternalServerError,
    ResourceExhausted,
    RetryError,
    TooManyRequests,
)
from google.genai.types import GenerateContentConfig, GenerateContentResponse, ThinkingConfig
from pydantic import BaseModel

from modelgauge.general import APIException
from modelgauge.prompt import TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import REFUSAL_RESPONSE, PromptResponseSUT, SUTOptions, SUTResponse  # usort: skip
from modelgauge.suts.google_generativeai import (
    GOOGLE_REFUSAL_FINISH_REASONS,
    GoogleAiApiKey,
)  # Both SDKs use the same API key.
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS


class GenAiRequest(BaseModel):
    model: str
    contents: str
    config: Optional[GenerateContentConfig] = None


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class GoogleGenAiSUT(PromptResponseSUT[GenAiRequest, GenerateContentResponse]):
    def __init__(self, uid: str, model_name: str, reasoning: bool, api_key: GoogleAiApiKey):
        super().__init__(uid)
        self.model_name = model_name
        self.client: Optional[genai.Client] = None
        self.reasoning = reasoning
        self.api_key = api_key.value

    def _load_client(self) -> genai.Client:
        return genai.Client(api_key=self.api_key)

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> GenAiRequest:
        optional = {}
        if not self.reasoning:
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
        if self.client is None:
            # Handle lazy init.
            self.client = self._load_client()
        return self.client.models.generate_content(**request.model_dump(exclude_none=True))

    def translate_response(self, request: GenAiRequest, response: GenerateContentResponse) -> SUTResponse:
        assert (
            len(response.candidates) <= 1
        ), f"Expected a single candidate in the response, got {len(response.candidates)}."
        if len(response.candidates) == 0:
            # This is apparently a refusal. At least, it's what happens consistently with a set of
            # prompts in the CSE, SRC, and SXC hazards
            response_text = REFUSAL_RESPONSE
        else:
            candidate = response.candidates[0]
            if candidate.finish_reason in GOOGLE_REFUSAL_FINISH_REASONS:
                response_text = REFUSAL_RESPONSE
            elif candidate.content is not None:
                response_text = candidate.content.parts[0].text
            else:
                raise APIException(
                    f"Unexpected candidate in response from GoogleGenAiSUT {self.uid}: {candidate}. "
                    f"The candidate does not have any content,"
                    f" but it's finish reason {candidate.finish_reason} does not qualify as a refusal."
                )

        return SUTResponse(text=response_text)


models = ["gemini-2.5-flash-preview-05-20"]
for model in models:
    SUTS.register(
        GoogleGenAiSUT,
        f"google-genai-{model}-no-reasoning",
        model,
        False,
        InjectSecret(GoogleAiApiKey),
    )
