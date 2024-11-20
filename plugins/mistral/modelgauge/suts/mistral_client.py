from mistralai import Mistral

from mistralai.models import HTTPValidationError, SDKError
from mistralai.utils import BackoffStrategy, RetryConfig

from modelgauge.secret_values import RequiredSecret, SecretDescription


BACKOFF_INITIAL_MILLIS = 500
BACKOFF_MAX_INTERVAL_MILLIS = 10_000
BACKOFF_EXPONENT = 1.1
BACKOFF_MAX_ELAPSED_MILLIS = 60_000


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
        if not self._client:
            self._client = Mistral(
                api_key=self.api_key,
                retry_config=RetryConfig(
                    "backoff",
                    BackoffStrategy(
                        BACKOFF_INITIAL_MILLIS,
                        BACKOFF_MAX_INTERVAL_MILLIS,
                        BACKOFF_EXPONENT,
                        BACKOFF_MAX_INTERVAL_MILLIS,
                    ),
                    True,
                ),
            )
        return self._client

    def request(self, req: dict):
        response = None
        try:
            # print(req)
            response = self.client.chat.complete(**req)
            # print(response)
            return response
        # TODO check if this actually happens
        except HTTPValidationError as exc:
            raise (exc)
        # TODO check if the retry strategy takes care of this
        except SDKError as exc:
            raise (exc)
        # TODO what else can happen?
        except Exception as exc:
            raise (exc)
