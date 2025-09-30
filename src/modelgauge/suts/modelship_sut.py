from typing import Optional, Mapping, Any

from modelgauge.auth.openai_compatible_secrets import OpenAICompatibleApiKey
from modelgauge.dynamic_sut_factory import DynamicSUTFactory
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.openai_client import OpenAIChat, OpenAIChatRequest


class ModelShipSecret(RequiredSecret):
    provider: str = "modelship"

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(scope=cls.provider, key="api_key", instructions="Ask around")


class ModelShipSUT(OpenAIChat):

    def __init__(
        self,
        uid: str,
        model: str,
        vllm_options: Mapping[str, str],
        api_key: Optional[OpenAICompatibleApiKey] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(uid, model, api_key=api_key, base_url=base_url)
        self.vllm_options = vllm_options

    def request_as_dict_for_client(self, request: OpenAIChatRequest) -> dict[str, Any]:
        request_as_dict = super().request_as_dict_for_client(request)
        request_as_dict["metadata"] = {"vllm_options": self.vllm_options}
        return request_as_dict


class ModelShipSUTFactory(DynamicSUTFactory):
    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(ModelShipSecret)
        return [api_key]

    def make_sut(self, sut_definition: SUTDefinition) -> ModelShipSUT:
        base_url = "http://mlc2:8123/v1/"
        [api_key] = self.injected_secrets()
        model = sut_definition.get("maker") + "/" + sut_definition.get("model")
        return ModelShipSUT(
            sut_definition.uid, model, self.vllm_options_for(sut_definition), base_url=base_url, api_key=api_key
        )

    def vllm_options_for(self, sut_definition: SUTDefinition) -> Mapping[str, str | int | float | bool | None] | None:
        return {k[5:]: v for k, v in sut_definition.get_matching("vllm-").items()}
