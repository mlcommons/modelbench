import difflib

from google import genai

from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.secret_values import RawSecrets, InjectSecret
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.google_genai import GoogleGenAiSUT
from modelgauge.suts.google_generativeai import GoogleAiApiKey

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
                f"{requested_model} not found in Gemini models. Closest options are {difflib.get_close_matches(requested_model, model_names)}")

        return GoogleGenAiSUT(
            sut_definition.dynamic_uid,
            requested_model,
            sut_definition.get("reasoning", False),
            self._gemini_secret()
        )

# class TogetherSUTFactory(DynamicSUTFactory):
#     def __init__(self, raw_secrets: RawSecrets):
#         super().__init__(raw_secrets)
#         self._client = None  # Lazy load.
#
#     @property
#     def client(self) -> Together:
#         if self._client is None:
#             api_key = self.injected_secrets()[0]
#             self._client = Together(api_key=api_key.value)
#         return self._client
#
#     def get_secrets(self) -> list[InjectSecret]:
#         api_key = InjectSecret(TogetherApiKey)
#         return [api_key]
#
#     def _find(self, sut_metadata: DynamicSUTMetadata):
#         model = None
#         try:
#             model = sut_metadata.external_model_name().lower()
#             self.client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {"role": "user", "content": "Anybody home?"},
#                 ],
#             )
#         except Exception as e:
#             raise ModelNotSupportedError(
#                 f"Model {sut_metadata.external_model_name()} not found or not available on together: {e}"
#             )
#
#         return model
#
#     def make_sut(self, sut_definition: SUTDefinition) -> TogetherChatSUT:
#         sut_metadata = sut_definition.to_dynamic_sut_metadata()
#         model_name = self._find(sut_metadata)
#         if not model_name:
#             raise ModelNotSupportedError(
#                 f"Model {sut_metadata.external_model_name()} not found or not available on together."
#             )
#
#         assert sut_metadata.driver == DRIVER_NAME
#         return TogetherChatSUT(
#             sut_definition.dynamic_uid,
#             sut_metadata.external_model_name(),
#             *self.injected_secrets(),
#         )
