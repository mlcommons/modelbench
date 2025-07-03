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


def make_sut(sut_uid: str, *args, **kwargs) -> tuple | None:
    sut_metadata: DynamicSUTMetadata = DynamicSUTMetadata.parse_sut_uid(sut_uid)
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
                f"Huggingface doesn't know model {sut_uid}, or you need credentials for its repo."
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
    def find(
        sut_metadata: DynamicSUTMetadata, requested_provider: str = "", find_alternative: bool = False
    ) -> str | None:
        model_name = sut_metadata.external_model_name()
        if not requested_provider:
            requested_provider = sut_metadata.provider

        found_provider = None
        try:
            inference_providers = find_inference_provider_for(model_name)
            found = inference_providers.get(requested_provider, None)  # type: ignore
            if found:
                found_provider = requested_provider
            else:
                if find_alternative:
                    for alt_provider, _ in inference_providers.inference_provider_mapping.items():  # type: ignore
                        found_provider = str(alt_provider)
                        break  # we just grab the first one; is that the right choice?
            if not found_provider:
                if requested_provider:
                    msg = f"{model_name} is not available on {requested_provider} via Huggingface"
                else:
                    msg = f"No provider found for {model_name}"
                raise ProviderNotFoundError(msg)
        except Exception as e:
            logging.error(
                f"Error looking up inference providers for {model_name} and provider {requested_provider}: {e}"
            )
            raise
        return found_provider

    @staticmethod
    def make_sut(sut_metadata: DynamicSUTMetadata) -> tuple | None:
        logging.info(
            f"Looking up serverless inference endpoints for {sut_metadata.model} on {sut_metadata.provider}..."
        )
        model_name = sut_metadata.external_model_name()
        found_provider = HuggingFaceChatCompletionServerlessSUTFactory.find(sut_metadata)
        if found_provider:
            sut_uid = DynamicSUTMetadata.make_sut_uid(sut_metadata)
            return (
                HuggingFaceChatCompletionServerlessSUT,
                sut_uid,
                model_name,
                found_provider,
                HuggingFaceSUTFactory.get_secrets(),
            )
        else:
            logging.error(f"{sut_metadata.model} on {sut_metadata.provider} not found")
            return None


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
        if endpoint_name:
            sut_uid = DynamicSUTMetadata.make_sut_uid(sut_metadata)
            return (HuggingFaceChatCompletionDedicatedSUT, sut_uid, endpoint_name, HuggingFaceSUTFactory.get_secrets())
        return None
