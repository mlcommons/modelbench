import logging
import huggingface_hub as hfh
from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.dynamic_sut_factory import (
    DynamicSUTFactory,
    ModelNotSupportedError,
    ProviderNotFoundError,
)

from modelgauge.log_config import get_logger
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.huggingface_chat_completion import (
    BaseHuggingFaceChatCompletionSUT,
    HuggingFaceChatCompletionDedicatedSUT,
    HuggingFaceChatCompletionServerlessSUT,
)

DRIVER_NAME = "hfrelay"

logger = get_logger(__name__)
# Set HF logging to ERROR because its default logger level is DEBUG.
# There are also many warnings which are not really actionable and very repetitive.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class HuggingFaceSUTFactory(DynamicSUTFactory):
    def __init__(self, raw_secrets: RawSecrets):
        super().__init__(raw_secrets)
        self.serverless_factory = HuggingFaceChatCompletionServerlessSUTFactory(raw_secrets)
        self.dedicated_factory = HuggingFaceChatCompletionDedicatedSUTFactory(raw_secrets)

    def get_secrets(self) -> list[InjectSecret]:
        hf_token = InjectSecret(HuggingFaceInferenceToken)
        return [hf_token]

    def make_sut(self, sut_definition: SUTDefinition) -> BaseHuggingFaceChatCompletionSUT:
        try:
            return self.serverless_factory.make_sut(sut_definition)
        except ProviderNotFoundError:
            # is there a dedicated option? probably not, but we check anyway
            try:
                return self.dedicated_factory.make_sut(sut_definition)
            except ProviderNotFoundError:
                raise ModelNotSupportedError(
                    f"Huggingface doesn't know model {sut_definition.external_model_name()}, or you need credentials for its repo."
                )


class HuggingFaceChatCompletionServerlessSUTFactory(DynamicSUTFactory):

    def get_secrets(self) -> list[InjectSecret]:
        hf_token = InjectSecret(HuggingFaceInferenceToken)
        return [hf_token]

    @staticmethod
    def find_inference_provider_for(model_name) -> list | None:
        try:
            inference_providers = hfh.model_info(model_name, expand=["inferenceProviderMapping"])
            providers = inference_providers.inference_provider_mapping
            if not providers:
                raise ProviderNotFoundError(f"No provider found for {model_name}")
            return providers
        except hfh.errors.RepositoryNotFoundError as mexc:
            logger.error(f"Huggingface doesn't know model {model_name}, or you need credentials for its repo: {mexc}")
            raise ModelNotSupportedError from mexc

    @staticmethod
    def _find(sut_definition: SUTDefinition) -> str:
        model_name = sut_definition.external_model_name()
        provider: str = sut_definition.get("provider")  # type: ignore
        inference_providers = HuggingFaceChatCompletionServerlessSUTFactory.find_inference_provider_for(model_name)
        for ip in inference_providers:
            if ip.provider == provider:
                return provider
        msg = f"{model_name} is not available on {provider} via Huggingface"
        raise ProviderNotFoundError(msg)

    def make_sut(self, sut_definition: SUTDefinition) -> HuggingFaceChatCompletionServerlessSUT:
        logger.info(
            f"Looking up serverless inference endpoints for {sut_definition.external_model_name()} on {sut_definition.get('provider')}..."
        )
        model_name = sut_definition.external_model_name()
        found_provider = HuggingFaceChatCompletionServerlessSUTFactory._find(sut_definition)
        sut_uid = sut_definition.dynamic_uid
        return HuggingFaceChatCompletionServerlessSUT(
            sut_uid,
            model_name,
            found_provider,
            *self.injected_secrets(),
        )


class HuggingFaceChatCompletionDedicatedSUTFactory(DynamicSUTFactory):

    def get_secrets(self) -> list[InjectSecret]:
        hf_token = InjectSecret(HuggingFaceInferenceToken)
        return [hf_token]

    @staticmethod
    def _find(sut_definition: SUTDefinition) -> str | None:
        """Find endpoint, if it exists."""
        model_name = sut_definition.external_model_name()
        try:
            endpoints = hfh.list_inference_endpoints()
            for e in endpoints:
                if e.repository.lower() == model_name:
                    if e.status != "running":
                        try:
                            e.resume()
                        except Exception as ie:
                            logger.error(
                                f"Found endpoint for {model_name} but unable to start it. Check your token's permissions. {ie}"
                            )
                    return e.name
        except Exception as oe:
            logger.error(f"Error looking up dedicated endpoints for {model_name}: {oe}")
        return None

    def make_sut(self, sut_definition: SUTDefinition) -> HuggingFaceChatCompletionDedicatedSUT:
        endpoint_name = HuggingFaceChatCompletionDedicatedSUTFactory._find(sut_definition)
        if not endpoint_name:
            raise ProviderNotFoundError(
                f"No dedicated inference endpoint found for {sut_definition.external_model_name()}."
            )
        sut_uid = sut_definition.dynamic_uid
        return HuggingFaceChatCompletionDedicatedSUT(sut_uid, endpoint_name, self.injected_secrets())
