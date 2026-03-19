from openai import OpenAI

from modelgauge.dynamic_sut_factory import DynamicDriverSUTFactory, ModelNotSupportedError
from modelgauge.secret_values import InjectSecret
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.nvidia_nim_api_client import BASE_URL, NvidiaNIMApiKey, NvidiaNIMApiClient


class NvidiaNIMSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "nvidia-nim"

    def __init__(self, raw_secrets):
        super().__init__(raw_secrets)
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.injected_secrets()[0].value, base_url=BASE_URL)
        return self._client

    def get_secrets(self) -> list[InjectSecret]:
        return [InjectSecret(NvidiaNIMApiKey)]

    def _model_exists(self, model_name: str) -> bool:
        try:
            self.client.models.retrieve(model_name)  # type: ignore
            return True
        except:
            return False

    def make_sut(self, sut_definition: SUTDefinition) -> NvidiaNIMApiClient:
        model_name = sut_definition.external_model_name()
        if not self._model_exists(model_name):
            raise ModelNotSupportedError(f"Model {model_name} not found or not available on nvidia NIM.")
        return NvidiaNIMApiClient(sut_definition.uid, model_name, *self.injected_secrets())
