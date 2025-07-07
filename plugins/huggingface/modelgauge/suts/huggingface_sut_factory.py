import logging

import huggingface_hub as hfh
from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.dynamic_sut_factory import (
    DynamicSUTFactory,
    ModelNotSupportedError,
    ProviderNotFoundError,
)

from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata, UnknownSUTDriverError

from modelgauge.secret_values import InjectSecret
from modelgauge.suts.huggingface_chat_completion import (
    HuggingFaceChatCompletionDedicatedSUT,
    HuggingFaceChatCompletionServerlessSUT,
)

DRIVER_NAME = "hfrelay"


def make_sut(sut_metadata: DynamicSUTMetadata, *args, **kwargs) -> tuple | None:
    if sut_metadata.is_proxied():
        if sut_metadata.driver != DRIVER_NAME:
            raise UnknownSUTDriverError(f"Unknown driver '{sut_metadata.driver}'")
        return HuggingFaceChatCompletionServerlessSUTFactory.make_sut(sut_metadata)
    else:
        # is there a serverless option?
        sut = HuggingFaceChatCompletionServerlessSUTFactory.make_sut(sut_metadata)
        if not sut:
            # is there a dedicated option? probably not, but we check anyway
            sut = HuggingFaceChatCompletionDedicatedSUTFactory.make_sut(sut_metadata)
        if not sut:
            raise ModelNotSupportedError(
                f"Huggingface doesn't know model {sut_metadata.external_model_name()}, or you need credentials for its repo."
            )


def find_inference_provider_for(model_name) -> dict | None:
    try:
        inference_providers = hfh.model_info(model_name, expand="inferenceProviderMapping")
        providers = inference_providers.inference_provider_mapping
        if not providers:
            raise ProviderNotFoundError(f"No provider found for {model_name}")
        return providers
    except hfh.errors.RepositoryNotFoundError as mexc:
        logging.error(f"Huggingface doesn't know model {model_name}, or you need credentials for its repo: {mexc}")
        raise ModelNotSupportedError from mexc


class HuggingFaceSUTFactory(DynamicSUTFactory):

    @staticmethod
    def get_secrets() -> InjectSecret:
        hf_token = InjectSecret(HuggingFaceInferenceToken)
        return hf_token


class HuggingFaceChatCompletionServerlessSUTFactory(HuggingFaceSUTFactory):

    @staticmethod
    def find(sut_metadata: DynamicSUTMetadata) -> str | None:
        model_name = sut_metadata.external_model_name()
        provider: str = sut_metadata.provider  # type: ignore
        inference_providers = find_inference_provider_for(model_name)
        found = inference_providers.get(provider, None)  # type: ignore
        if not found:
            msg = f"{model_name} is not available on {provider} via Huggingface"
            raise ProviderNotFoundError(msg)
        return provider

    @staticmethod
    def make_sut(sut_metadata: DynamicSUTMetadata) -> tuple | None:
        logging.info(
            f"Looking up serverless inference endpoints for {sut_metadata.model} on {sut_metadata.provider}..."
        )
        model_name = sut_metadata.external_model_name()
        found_provider = HuggingFaceChatCompletionServerlessSUTFactory.find(sut_metadata)
        if not found_provider:
            logging.error(f"{sut_metadata.model} on {sut_metadata.provider} not found")
            return None
        sut_uid = DynamicSUTMetadata.make_sut_uid(sut_metadata)
        return (
            HuggingFaceChatCompletionServerlessSUT,
            sut_uid,
            model_name,
            found_provider,
            HuggingFaceSUTFactory.get_secrets(),
        )


class HuggingFaceChatCompletionDedicatedSUTFactory(HuggingFaceSUTFactory):

    @staticmethod
    def find(sut_metadata: DynamicSUTMetadata) -> str | None:
        model_name = sut_metadata.external_model_name()
        try:
            endpoints = hfh.list_inference_endpoints()
            for e in endpoints:
                if e.repository == model_name and e.status != "running":
                    try:
                        e.resume()
                    except Exception as ie:
                        logging.error(
                            f"Found endpoint for {model_name} but unable to start it. Check your token's permissions. {ie}"
                        )
                    return e.name
        except Exception as oe:
            logging.error(f"Error looking up dedicated endpoints for {model_name}: {oe}")
        return None

    @staticmethod
    def make_sut(sut_metadata: DynamicSUTMetadata) -> tuple | None:
        endpoint_name = HuggingFaceChatCompletionDedicatedSUTFactory.find(sut_metadata)
        if not endpoint_name:
            return None
        sut_uid = DynamicSUTMetadata.make_sut_uid(sut_metadata)
        return (HuggingFaceChatCompletionDedicatedSUT, sut_uid, endpoint_name, HuggingFaceSUTFactory.get_secrets())
