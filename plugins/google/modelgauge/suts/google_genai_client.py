import google.generativeai as genai  # type: ignore
from abc import abstractmethod
from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
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
GEMINI_HARM_CATEGORIES = [
    HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    HarmCategory.HARM_CATEGORY_HARASSMENT,
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
    safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None


class GoogleGenAiResponse(BaseModel):
    class Candidate(BaseModel):
        content: Optional[Dict] = None
        finish_reason: int

    candidates: List[Candidate]
    usage_metadata: Dict


class GoogleGenAiBaseSUT(PromptResponseSUT[GoogleGenAiRequest, GoogleGenAiResponse]):
    def __init__(self, uid: str, model_name: str, api_key: GoogleAiApiKey):
        super().__init__(uid)
        self.model_name = model_name
        self.model: Optional[genai.GenerativeModel] = None
        genai.configure(api_key=api_key.value)

    @property
    @abstractmethod
    def refusal_finish_reasons(self):
        """List of finish reasons that should be treated as refusal responses."""
        pass

    @property
    @abstractmethod
    def safety_settings(self) -> Optional[Dict[HarmCategory, HarmBlockThreshold]]:
        """Safety settings that are included in SUT requests."""
        pass

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
        return GoogleGenAiRequest(
            contents=prompt.text, generation_config=generation_config, safety_settings=self.safety_settings
        )

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
            if candidate.finish_reason in self.refusal_finish_reasons:
                completions.append(SUTCompletion(text=REFUSAL_RESPONSE))
            elif candidate.content is not None:
                completions.append(SUTCompletion(text=candidate.content["parts"][0]["text"]))
            else:
                raise APIException(
                    f"Unexpected candidate in response from GoogleGenAiSUT {self.uid}: {candidate}. "
                    f"The candidate does not have any content,"
                    f" but it's finish reason {candidate.finish_reason} does not qualify as a refusal."
                )
        return SUTResponse(completions=completions)


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class GoogleGenAiDefaultSUT(GoogleGenAiBaseSUT):
    """SUT for Google Generative AI model with the model's default safety settings.

    Finish reasons related to safety are treated as refusal responses."""

    @property
    def refusal_finish_reasons(self):
        # Taken from https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerateContentResponse#finishreason
        return [
            FinishReason.SAFETY,
            FinishReason.RECITATION,
            FinishReason.BLOCKLIST,
            FinishReason.PROHIBITED_CONTENT,
            FinishReason.SPII,
            FinishReason.OTHER,
        ]

    @property
    def safety_settings(self) -> Optional[Dict[HarmCategory, HarmBlockThreshold]]:
        return None


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class GoogleGeminiDisabledSafetySettingsSUT(GoogleGenAiBaseSUT):
    """SUT for Google Gemini model that removes that harm block threshold for all Gemini-specific harm categories."""

    @property
    def refusal_finish_reasons(self):
        """Model should not refuse for any safety-related reasons."""
        return []

    @property
    def safety_settings(self) -> Optional[Dict[HarmCategory, HarmBlockThreshold]]:
        return {harm: HarmBlockThreshold.BLOCK_NONE for harm in GEMINI_HARM_CATEGORIES}


gemini_models = ["gemini-1.5-flash", "gemini-1.0-pro", "gemini-1.5-pro"]
for model in gemini_models:
    SUTS.register(GoogleGenAiDefaultSUT, model, model, InjectSecret(GoogleAiApiKey))
    SUTS.register(
        GoogleGeminiDisabledSafetySettingsSUT, f"{model}-disabled_safety", model, InjectSecret(GoogleAiApiKey)
    )
