import difflib

from google import genai

from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.secret_values import RawSecrets, InjectSecret
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.google_genai import GoogleGenAiSUT, GoogleAiApiKey

DRIVER_NAME = "google"


class GoogleSUTFactory(DynamicSUTFactory):
    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(GoogleAiApiKey)
        return [api_key]

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._gemini_client = None  # Lazy load.

    def gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self._gemini_secret().value)
        return self._gemini_client

    def _gemini_secret(self) -> GoogleAiApiKey:
        return self.injected_secrets()[0]

    def make_sut(self, sut_definition: SUTDefinition) -> SUT:
        model_names = [m.name.replace("models/", "") for m in self.gemini_client().models.list()]
        requested_model = sut_definition.to_dynamic_sut_metadata().model
        if requested_model not in model_names:
            raise ModelNotSupportedError(
                f"{requested_model} not found in Gemini models. Closest options are {difflib.get_close_matches(requested_model, model_names, cutoff=0.1)}"
            )

        return GoogleGenAiSUT(
            sut_definition.dynamic_uid, requested_model, sut_definition.get("reasoning", False), self._gemini_secret()
        )
