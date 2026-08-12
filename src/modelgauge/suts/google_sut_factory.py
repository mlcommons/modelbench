import difflib
from typing import Optional

from google import genai

from modelgauge.dynamic_sut_factory import DynamicDriverSUTFactory, ModelNotSupportedError
from modelgauge.secret_values import RawSecrets, InjectSecret
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.google_genai import GoogleGenAiSUT, GoogleAiApiKey


class GoogleSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "google"

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
        modelinfo_by_name = {m.name.replace("models/", ""): m for m in self.gemini_client().models.list()}
        requested_model = sut_definition.to_dynamic_sut_metadata().model
        if requested_model not in modelinfo_by_name:
            raise ModelNotSupportedError(
                f"{requested_model} not found in Gemini models. Closest options are {difflib.get_close_matches(requested_model, modelinfo_by_name.keys(), cutoff=0.1)}"
            )
        selected_modelinfo = modelinfo_by_name[requested_model]
        if "generateContent" not in selected_modelinfo.supported_actions:
            raise ModelNotSupportedError(
                f"{requested_model} does not support generateContent; only works with {selected_modelinfo.supported_actions}"
            )

        if selected_modelinfo.thinking:
            reasoning: Optional[bool] = sut_definition.get("reasoning")
        else:
            reasoning = False

        return GoogleGenAiSUT(
            sut_definition.dynamic_uid, requested_model, reasoning, self.gemini_client()
        )

    def list_suts(self) -> list[SUTDefinition]:
        all_options = self.gemini_client().models.list()
        compatible_options = [m for m in all_options if "generateContent" in m.supported_actions]
        result = []
        for m in compatible_options:
            result.append(SUTDefinition(
                driver=self.DRIVER_NAME,
                maker="google",
                model=m.name.replace("models/", "")
            ))
        return result
