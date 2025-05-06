import logging

import huggingface_hub as hfh
from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.dynamic_sut import (
    DynamicSUTMaker,
    ModelNotSupportedError,
    ProviderNotFoundError,
    UnknownProxyError,
)
from modelgauge.secret_values import InjectSecret
from modelgauge.suts.huggingface_chat_completion import (
    HuggingFaceChatCompletionDedicatedSUT,
    HuggingFaceChatCompletionServerlessSUT,
)


def make_sut(model_name: str, provider: str = "") -> tuple | None:
    proxy, implied_provider, vendor, model = DynamicSUTMaker.parse_sut_name(model_name)
    if proxy not in {"hf", "huggingface", "hf-inference", "hf-relay"}:
        raise UnknownProxyError(f"SUT name {model_name} references an unknown proxy {proxy}")

    if implied_provider and not provider:
        provider = implied_provider

    # SUT proxied (relayed) by HF to a provider like Nebius
    if provider:
        return HuggingFaceChatCompletionServerlessSUTMaker.make_sut(model_name, provider)
    else:
        return HuggingFaceChatCompletionDedicatedSUTMaker.make_sut(model_name)


def find_inference_provider_for(name) -> dict | None:
    try:
        inference_providers = hfh.model_info(name, expand="inferenceProviderMapping")
        providers = inference_providers.inference_provider_mapping
        if not providers:
            raise ProviderNotFoundError(f"No provider found for {name}")
        return providers
    except hfh.errors.RepositoryNotFoundError as mexc:
        logging.error(f"Huggingface doesn't know model {name}, or you need credentials for its repo: {mexc}")
        raise ModelNotSupportedError from mexc


class HuggingFaceSUTMaker(DynamicSUTMaker):

    @staticmethod
    def make_sut_id(model_name: str) -> str:
        chunks = []
        _, provider, vendor, model = DynamicSUTMaker.parse_sut_name(model_name)
        if vendor:
            chunks.append(vendor)
        chunks.append(model)

        # add hf at end if it's not already in the chunks
        hf_identifiers = {"hf", "huggingface", "hf-inference", "hf-relay"}
        if len(hf_identifiers & set(chunks)) == 0:
            chunks.append("hf")

        # add hf as the provider (host) if it's not hf (which is already accounted for)
        if provider and provider not in hf_identifiers:
            chunks.append(provider)

        return "-".join(chunks).lower()

    @staticmethod
    def get_secrets() -> InjectSecret:
        hf_token = InjectSecret(HuggingFaceInferenceToken)
        return hf_token


class HuggingFaceChatCompletionServerlessSUTMaker(HuggingFaceSUTMaker):

    @staticmethod
    def find(model_name, provider, find_alternative: bool = False) -> str | None:
        name = DynamicSUTMaker.extract_model_name(model_name)
        found_provider = None
        try:
            inference_providers = find_inference_provider_for(name)
            found = inference_providers.get(provider, None)
            if found:
                found_provider = provider
            else:
                if find_alternative:
                    for alt_provider, _ in inference_providers.inference_provider_mapping.items():
                        found_provider = str(alt_provider)
                        break  # we just grab the first one; is that the right choice?
            if not found_provider:
                raise ProviderNotFoundError(f"No provider found for {model_name}")
        except Exception as e:
            logging.error(
                f"Error looking up inference providers for {model_name} aka {name} and provider {provider}: {e}"
            )
            raise
        return found_provider

    @staticmethod
    def make_sut(model_name: str, provider: str = "") -> tuple | None:
        logging.info(f"Looking up serverless inference endpoints for {model_name} on {provider}...")
        found_provider = HuggingFaceChatCompletionServerlessSUTMaker.find(model_name, provider)
        if found_provider:
            if not model_name.lower().startswith(f"hf/"):
                model_full_name = f"hf/{found_provider}/{model_name}"
            else:
                model_full_name = model_name.replace(provider, found_provider)
            sut_id = HuggingFaceSUTMaker.make_sut_id(model_full_name)
            return (
                HuggingFaceChatCompletionServerlessSUT,
                sut_id,
                DynamicSUTMaker.extract_model_name(model_name),
                found_provider,
                HuggingFaceSUTMaker.get_secrets(),
            )
        else:
            logging.error(f"{model_name} on {provider} not found")
            return None


class HuggingFaceChatCompletionDedicatedSUTMaker(HuggingFaceSUTMaker):

    @staticmethod
    def find(model_name) -> str | None:
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
    def make_sut(model_name) -> tuple | None:
        name = HuggingFaceChatCompletionDedicatedSUTMaker.find(model_name)
        if name:
            sut_id = HuggingFaceSUTMaker.make_sut_id(model_name)
            return (HuggingFaceChatCompletionDedicatedSUT, sut_id, name, HuggingFaceSUTMaker.get_secrets())
        return None
