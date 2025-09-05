from modelgauge.dynamic_sut_factory import DynamicSUTFactory
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.openai_client import OpenAIChat


class ModelShipSecret(RequiredSecret):
    provider: str = "modelship"

    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(scope=cls.provider, key="api_key", instructions="Ask around")


class ModelShipSUTFactory(DynamicSUTFactory):
    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(ModelShipSecret)
        return [api_key]

    def make_sut(self, sut_definition: SUTDefinition) -> SUT:
        base_url = "http://mlc2:8123/v1/"
        [api_key] = self.injected_secrets()
        model = sut_definition.get("maker") + "/" + sut_definition.get("model")
        return OpenAIChat(sut_definition.uid, model, base_url=base_url, api_key=api_key)
