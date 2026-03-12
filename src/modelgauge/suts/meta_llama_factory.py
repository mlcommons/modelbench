from llama_api_client import LlamaAPIClient

from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.meta_llama_client import MetaLlamaApiKey, MetaLlamaModeratedSUT, MetaLlamaSUT


class LlamaSUTFactory(DynamicSUTFactory):
    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = self.injected_secrets()[0].value
            self._client = LlamaAPIClient(api_key=api_key)
        return self._client

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(MetaLlamaApiKey)
        return [api_key]

    def _get_model_name(self, model) -> str | None:
        """Llama API model names are case sensitive."""
        models = self.client.models.list()
        for m in models:
            if m.id.lower() == model.lower():
                return m.id
        return None

    def make_sut(self, sut_definition: SUTDefinition, moderated: bool = False) -> SUT:
        model_name = sut_definition.to_dynamic_sut_metadata().external_model_name()
        model_name = self._get_model_name(model_name)

        if model_name is None:
            raise ModelNotSupportedError(f"Model {model_name} not found or not available via Llama API.")

        if moderated:
            return MetaLlamaModeratedSUT(sut_definition.uid, model_name, *self.injected_secrets())
        return MetaLlamaSUT(sut_definition.uid, model_name, *self.injected_secrets())
