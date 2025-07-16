from openai import OpenAI, NotFoundError

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, ModelNotSupportedError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import InjectSecret
from modelgauge.suts.openai_client import OpenAIApiKey, OpenAIChat, OpenAIOrgId


DRIVER_NAME = "openai"


class OpenAISUTFactory(DynamicSUTFactory):
    @staticmethod
    def _model_exists(sut_metadata: DynamicSUTMetadata):
        secrets = load_secrets_from_config()
        api_key = OpenAIApiKey.make(secrets).value
        org_id = OpenAIOrgId.make(secrets).value

        client = OpenAI(api_key=api_key, organization=org_id, max_retries=7)

        try:
            client.models.retrieve(sut_metadata.model)
        except NotFoundError:
            return False
        return True

    @staticmethod
    def make_sut(sut_metadata: DynamicSUTMetadata):
        if not OpenAISUTFactory._model_exists(sut_metadata):
            raise ModelNotSupportedError(f"Model {sut_metadata.model} not found or not available on openai.")

        assert sut_metadata.driver == DRIVER_NAME
        return (
            OpenAIChat,
            DynamicSUTMetadata.make_sut_uid(sut_metadata),
            sut_metadata.model,
            InjectSecret(OpenAIApiKey),
            InjectSecret(OpenAIOrgId),
        )
