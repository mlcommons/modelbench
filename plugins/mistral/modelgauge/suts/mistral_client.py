from mistralai import Mistral

from modelgauge.secret_values import RequiredSecret, SecretDescription


class MistralAIAPIKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="mistralai",
            key="api_key",
            instructions="MistralAI API key. See https://docs.mistral.ai/getting-started/quickstart/",
        )


class MistralAIClient:
    def __init__(
        self,
        model_name: str,
        api_key: MistralAIAPIKey,
    ):
        self.model_name = model_name
        self.api_key = api_key.value
        self._client = None

    @property
    def client(self) -> Mistral:
        self._client = Mistral(api_key=self.api_key)
        return self._client

    def request(self, req: dict):
        try:
            response = self.client.chat.complete(**req)
            return response
        except Exception as exc:
            raise
