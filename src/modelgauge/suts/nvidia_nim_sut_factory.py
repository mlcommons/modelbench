from openai import OpenAI

from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.nvidia_nim_api_client import BASE_URL, NvidiaNIMApiClient, NvidiaNIMApiKey
from modelgauge.suts.openai_sut_factory import OpenAIGenericSUTFactory


class NvidiaNIMSUTFactory(OpenAIGenericSUTFactory):
    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets, BASE_URL)

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(NvidiaNIMApiKey)
        return [api_key]

# class NvidiaNIMSUTFactory(DynamicSUTFactory):
#     def __init__(self, raw_secrets: RawSecrets):
#         super().__init__(raw_secrets)
#         self._client = None  # Lazy load.
#
#     @property
#     def client(self) -> OpenAI:
#         if self._client is None:
#             api_key = self.injected_secrets()[0].value
#             self._client = OpenAI(api_key=api_key, base_url=BASE_URL)
#         return self._client
#
#     def get_secrets(self) -> list[InjectSecret]:
#         api_key = InjectSecret(NvidiaNIMApiKey)
#         return [api_key]
#
#     def model_exists(self, model: str) -> bool:
#         models = self.client.models.list()
#         for m in models:
#             if m["id"].lower() == model.lower():
#                 return True
#         return False
#
#     def make_sut(self, sut_definition: SUTDefinition) -> SUT:
#         model_name = sut_definition.to_dynamic_sut_metadata().external_model_name()
#
#         try:
#             self.client.model_info(model_name)
#         except Exception as e:
#             raise ModelNotSupportedError(f"Model {model_name} not found or not available on mistral: {e}")
#
#         return NvidiaNIMApiClient(sut_definition.dynamic_uid, model_name, *self.injected_secrets())
