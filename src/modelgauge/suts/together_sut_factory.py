import logging

from airrlogger.log_config import get_logger
from together import Together  # type: ignore

from modelgauge.auth.together_secrets import TogetherApiKey, TogetherProjectId
from modelgauge.dynamic_sut_factory import (
    DynamicDriverSUTFactory,
    ModelNotSupportedError,
)
from modelgauge.general import APIException
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.together_client import (
    TogetherChatSUT,
    TogetherDedicatedChatSUT,
    _retrying_request,
)

logger = get_logger(__name__)
logging.getLogger("together_sut_factory").setLevel(logging.ERROR)


class TogetherServerlessSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "together-serverless"

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

    def _find(self, model: str) -> str | None:
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

    def make_sut(self, sut_definition: SUTDefinition) -> TogetherChatSUT:
        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        model = sut_metadata.external_model_name().lower()
        model_name = self._find(model)
        api_key = self.injected_secrets()[0]
        if model_name is None:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together serverless."
            )
        return TogetherChatSUT(
            sut_definition.dynamic_uid,
            sut_metadata.external_model_name(),
            api_key,
        )


class TogetherDedicatedSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "together-dedicated"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)

    def list_suts(self) -> list[SUTDefinition]:
        api_key, project_id = self.injected_secrets()
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {api_key.value}",
        }
        response = _retrying_request(
            f"https://api.together.ai/v2/projects/{project_id.value}/endpoints",
            headers,
            None,
            "GET",
        )

        definitions: list[SUTDefinition] = []
        seen_uids: set[str] = set()
        for endpoint in response.json().get("data", []):
            for deployment in endpoint.get("deployments", []):
                model_name = deployment.get("name")
                if not isinstance(model_name, str) or not model_name.strip():
                    continue

                model_name = model_name.strip()
                if "/" in model_name:
                    maker, model = model_name.split("/", 1)
                    definition = SUTDefinition(
                        driver=self.DRIVER_NAME,
                        maker=maker,
                        model=model,
                    )
                else:
                    definition = SUTDefinition(
                        driver=self.DRIVER_NAME,
                        model=model_name,
                    )

                if definition.uid not in seen_uids:
                    definitions.append(definition)
                    seen_uids.add(definition.uid)

        return definitions

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(TogetherApiKey)
        project_id = InjectSecret(TogetherProjectId)
        return [api_key, project_id]

    def make_sut(self, sut_definition: SUTDefinition) -> TogetherDedicatedChatSUT:
        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        api_key, project_id = self.injected_secrets()
        try:
            return TogetherDedicatedChatSUT(
                sut_definition.dynamic_uid, sut_metadata.external_model_name(), api_key, project_id
            )
        except APIException as e:
            raise ModelNotSupportedError(
                f"Model {sut_metadata.external_model_name()} not found or not available on together dedicated endpoints. Try removing the maker name e.g. `gpt-oss-20b` instead of `openai/gpt-oss-20b`."
            )
