from openai import OpenAI, NotFoundError

from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import InjectSecret
from modelgauge.suts.openai_client import OpenAIApiKey, OpenAIChat, OpenAIOrgId


DRIVER_NAME = "openai"


class OpenAISUTFactory(DynamicSUTFactory):
    @staticmethod
    def get_secrets() -> list[InjectSecret]:
        api_key = InjectSecret(OpenAIApiKey)
        org_id = InjectSecret(OpenAIOrgId)
        return [api_key, org_id]

    def _model_exists(self, sut_metadata: DynamicSUTMetadata):
        api_key, org_id = self.injected_secrets()

        client = OpenAI(api_key=api_key.value, organization=org_id.value, max_retries=7)

        try:
            client.models.retrieve(sut_metadata.model)
        except NotFoundError:
            return False
        return True

    def make_sut(self, sut_metadata: DynamicSUTMetadata) -> OpenAIChat:
        if not self._model_exists(sut_metadata):
            raise ModelNotSupportedError(f"Model {sut_metadata.model} not found or not available on openai.")

        assert sut_metadata.driver == DRIVER_NAME
        return OpenAIChat(DynamicSUTMetadata.make_sut_uid(sut_metadata), sut_metadata.model, *self.injected_secrets())
