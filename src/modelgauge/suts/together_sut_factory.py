import logging

from together import Together  # type: ignore

from airrlogger.log_config import get_logger

from modelgauge.auth.together_secrets import TogetherApiKey, TogetherProjectId
from modelgauge.dynamic_sut_factory import DynamicDriverSUTFactory, ModelNotSupportedError
from modelgauge.general import APIException
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut import PromptResponseSUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import TogetherChatSUT, TogetherDedicatedChatSUT

logger = get_logger(__name__)
logging.getLogger("together_sut_factory").setLevel(logging.ERROR)


class TogetherSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "together"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client = None

    @property
    def client(self) -> Together:
        if self._client is None:
            api_key = self.injected_secrets()[0]
            self._client = Together(api_key=api_key.value)
        return self._client

    @client.setter
    def client(self, value: Together) -> None:
        self._client = value

    def _find_serverless(self, model: str) -> str | None:
        try:
            self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Anybody home?"},
                ],
            )
            return model
        except Exception as e:
            logger.info(f"Error looking up serverless model {model} on together: {e}")
        return None

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(TogetherApiKey)
        project_id = InjectSecret(TogetherProjectId)
        return [api_key, project_id]

    def make_sut(self, sut_definition: SUTDefinition) -> PromptResponseSUT:
        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        model = sut_metadata.external_model_name().lower()
        # first try serverless
        model_name = self._find_serverless(model)
        api_key, project_id = self.injected_secrets()
        if model_name is not None:
            return TogetherChatSUT(
                sut_definition.dynamic_uid,
                sut_metadata.external_model_name(),
                api_key,
            )
        # serverless failed; try dedicated.
        try:
            return TogetherDedicatedChatSUT(
                sut_definition.dynamic_uid, sut_metadata.external_model_name(), api_key, project_id
            )
        except APIException as e:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together serverless nor dedicated endpoints. Try removing the maker name e.g. `gpt-oss-20b` instead of `openai/gpt-oss-20b`."
            )
