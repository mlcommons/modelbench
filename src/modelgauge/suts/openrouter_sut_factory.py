import difflib

from openai import OpenAI

from modelgauge.dynamic_sut_factory import (
    DynamicDriverSUTFactory,
    ModelNotSupportedError,
)
from modelgauge.secret_values import (
    InjectSecret,
    RawSecrets,
    RequiredSecret,
    SecretDescription,
)
from modelgauge.sut import SUT
from modelgauge.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
    ProducesPerTokenLogProbabilities,
)
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.openai_client import OpenAIResponsesSUT

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_NUM_RETRIES = 7


class OpenRouterApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="openrouter",
            key="api_key",
            instructions="Create an OpenRouter API key at https://openrouter.ai/settings/keys.",
        )


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class OpenRouterResponsesSUT(OpenAIResponsesSUT):
    def __init__(
        self,
        uid: str,
        model: str,
        client: OpenAI,
        provider: str | None = None,
    ):
        super().__init__(uid=uid, model=model, client=client)
        self.provider = provider

    def _call_client(self, request):
        request_args = self.request_as_dict_for_client(request)
        if self.provider:
            return self.client.responses.create(
                **request_args,
                extra_body={
                    "provider": {
                        "only": [self.provider],
                        "allow_fallbacks": False,
                    }
                },
            )
        return self.client.responses.create(**request_args)


class OpenRouterSUTFactory(DynamicDriverSUTFactory):
    DRIVER_NAME = "openrouter"

    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self._client: OpenAI | None = None

    def get_secrets(self) -> list[InjectSecret]:
        return [InjectSecret(OpenRouterApiKey)]

    def _secret(self) -> OpenRouterApiKey:
        return self.injected_secrets()[0]

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._secret().value,
                base_url=OPENROUTER_BASE_URL,
                max_retries=OPENROUTER_NUM_RETRIES,
            )
        return self._client

    @staticmethod
    def _definition_for_model_id(model_id: str) -> SUTDefinition | None:
        # OpenRouter variants use ':' in model IDs (for example :free and
        # :nitro), while ModelBench uses ':' as its dynamic SUT UID separator.
        # Keep discovery unambiguous until the UID format has an explicit
        # representation for those variants.
        if ":" in model_id:
            return None

        maker, separator, model = model_id.partition("/")
        if not separator:
            return SUTDefinition(
                driver=OpenRouterSUTFactory.DRIVER_NAME,
                model=model_id,
            )

        return SUTDefinition(
            driver=OpenRouterSUTFactory.DRIVER_NAME,
            maker=maker,
            model=model,
        )

    def list_suts(self) -> list[SUTDefinition]:
        definitions: list[SUTDefinition] = []
        for model in self.client.models.list():
            definition = self._definition_for_model_id(model.id)
            if definition is not None:
                definitions.append(definition)
        return definitions

    def make_sut(self, sut_definition: SUTDefinition) -> SUT:
        requested_model = sut_definition.external_model_name()
        available_models = {definition.external_model_name() for definition in self.list_suts()}

        if requested_model not in available_models:
            closest = difflib.get_close_matches(
                requested_model,
                available_models,
                cutoff=0.1,
            )
            raise ModelNotSupportedError(
                f"Model {requested_model} not found or not available on OpenRouter. " f"Closest options are {closest}"
            )

        provider = sut_definition.get("provider")
        provider_name = provider if isinstance(provider, str) and provider else None

        return OpenRouterResponsesSUT(
            uid=sut_definition.uid,
            model=requested_model,
            client=self.client,
            provider=provider_name,
        )
