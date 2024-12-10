from mistralai import Mistral
from mistralai.models import HTTPValidationError, SDKError
from mistralai.utils import BackoffStrategy, RetryConfig

from modelgauge.secret_values import RequiredSecret, SecretDescription

BACKOFF_INITIAL_MILLIS = 500
BACKOFF_MAX_INTERVAL_MILLIS = 10_000
BACKOFF_EXPONENT = 1.1
BACKOFF_MAX_ELAPSED_MILLIS = 120_000


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
                timeout_ms=BACKOFF_MAX_ELAPSED_MILLIS * 3,
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

    @staticmethod
    def _make_request(endpoint, kwargs: dict):
        try:
            response = endpoint(**kwargs)
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

    def request(self, req: dict):
        if self.client.chat.sdk_configuration._hooks.before_request_hooks:
            # work around bug in client
            self.client.chat.sdk_configuration._hooks.before_request_hooks = []
        return self._retry_request(self.client.chat.complete, req)

    def score_conversation(self, model, prompt, response):
        """Returns moderation object for a conversation."""
        req = {
            "model": model,
            "inputs": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
        }
        return self._retry_request(self.client.classifiers.moderate_chat, req)
