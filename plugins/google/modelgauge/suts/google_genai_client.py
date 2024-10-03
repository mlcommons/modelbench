import google.generativeai as genai
from pydantic import BaseModel
from typing import Optional

from modelgauge.prompt import ChatPrompt, ChatRole, SUTOptions, TextPrompt
from modelgauge.secret_values import InjectSecret, OptionalSecret, RequiredSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS


class GoogleAiApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="google_ai",
            key="api_key",
            instructions="See https://aistudio.google.com/app/apikey",
        )


class GoogleGenAiRequest(BaseModel):
    text: str


class GoogleGenAiResponse(BaseModel):
    text: str


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
        # TODO: Add options
        return GoogleGenAiRequest(text=prompt.text)

    def evaluate(self, request: GoogleGenAiRequest) -> GoogleGenAiResponse:
        if self.model is None:
            # Handle lazy init.
            self.model = self._load_client()
        # TODO
        return self.model.generate_content(request.text)

    def translate_response(self, request: GoogleGenAiRequest, response: GoogleGenAiResponse) -> SUTResponse:
        return SUTResponse(completions=[SUTCompletion(text=response.text)])


SUTS.register(GoogleGenAiSUT, "gemini-1.5-flash", "gemini-1.5-flash", InjectSecret(GoogleAiApiKey))
