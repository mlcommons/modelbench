import google.generativeai as genai  # type: ignore
from pydantic import BaseModel
from typing import Dict, List, Optional

from modelgauge.general import APIException
from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import REFUSAL_RESPONSE, PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS

FinishReason = genai.protos.Candidate.FinishReason

# Taken from https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerateContentResponse#finishreason
GOOGLE_REFUSAL_FINISH_REASONS = [
    FinishReason.SAFETY,
    FinishReason.RECITATION,
    FinishReason.BLOCKLIST,
    FinishReason.PROHIBITED_CONTENT,
    FinishReason.SPII,
]


class GoogleAiApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="google_ai",
            key="api_key",
            instructions="See https://aistudio.google.com/app/apikey",
        )


class GoogleGenAiConfig(BaseModel):
    """Generation config for Google Gen AI requests.

    Based on https://ai.google.dev/api/generate-content#v1beta.GenerationConfig
    """

    stop_sequences: Optional[List[str]] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


class GoogleGenAiRequest(BaseModel):
    contents: str
    generation_config: GoogleGenAiConfig


class GoogleGenAiResponse(BaseModel):
    class Candidate(BaseModel):
        content: Optional[Dict] = None
        finish_reason: int

    candidates: List[Candidate]
    usage_metadata: Dict


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class GoogleGenAiSUT(PromptResponseSUT[GoogleGenAiRequest, GoogleGenAiResponse]):
    def __init__(self, uid: str, model_name: str, api_key: GoogleAiApiKey):
        super().__init__(uid)
        self.model_name = model_name
        self.model: Optional[genai.GenerativeModel] = None
        genai.configure(api_key=api_key.value)

    def _load_client(self) -> genai.GenerativeModel:
        return genai.GenerativeModel(self.model_name)

    def translate_text_prompt(self, prompt: TextPrompt) -> GoogleGenAiRequest:
        generation_config = GoogleGenAiConfig(
            stop_sequences=prompt.options.stop_sequences,
            max_output_tokens=prompt.options.max_tokens,
            temperature=prompt.options.temperature,
            top_p=prompt.options.top_p,
            top_k=prompt.options.top_k_per_token,
            presence_penalty=prompt.options.presence_penalty,
            frequency_penalty=prompt.options.frequency_penalty,
        )
        return GoogleGenAiRequest(contents=prompt.text, generation_config=generation_config)

    def evaluate(self, request: GoogleGenAiRequest) -> GoogleGenAiResponse:
        if self.model is None:
            # Handle lazy init.
            self.model = self._load_client()
        response = self.model.generate_content(**request.model_dump(exclude_none=True))
        # Convert to pydantic model
        return GoogleGenAiResponse(**response.to_dict())

    def translate_response(self, request: GoogleGenAiRequest, response: GoogleGenAiResponse) -> SUTResponse:
        completions = []
        for candidate in response.candidates:
            if candidate.finish_reason in GOOGLE_REFUSAL_FINISH_REASONS:
                completions.append(SUTCompletion(text=REFUSAL_RESPONSE))
            elif candidate.content is not None:
                completions.append(SUTCompletion(text=candidate.content["parts"][0]["text"]))
            else:
                raise APIException(
                    f"Unexpected candidate in response from GoogleGenAiSUT {self.uid}: {candidate}.\nThe candidate does not have any content but did not finish early for any known safety reasons."
                )
        return SUTResponse(completions=completions)


SUTS.register(GoogleGenAiSUT, "gemini-1.5-flash", "gemini-1.5-flash", InjectSecret(GoogleAiApiKey))
SUTS.register(GoogleGenAiSUT, "gemini-1.0-pro", "gemini-1.0-pro", InjectSecret(GoogleAiApiKey))
SUTS.register(GoogleGenAiSUT, "gemini-1.5-pro", "gemini-1.5-pro", InjectSecret(GoogleAiApiKey))
