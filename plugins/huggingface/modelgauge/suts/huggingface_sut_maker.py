import logging

import huggingface_hub as hfh
from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import SUTMaker
from modelgauge.suts.huggingface_chat_completion import (
    HuggingFaceChatCompletionDedicatedSUT,
    HuggingFaceChatCompletionServerlessSUT,
)


class HuggingFaceSUTMaker(SUTMaker):

    @staticmethod
    def make_sut_id(model_name: str) -> str:
        chunks = []
        proxy, provider, vendor, model = SUTMaker.parse_sut_name(model_name)
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
    def make_sut(model_name: str, provider: str = "") -> tuple | None:
        proxy, implied_provider, vendor, model = SUTMaker.parse_sut_name(model_name)
        if implied_provider and not provider:
            provider = implied_provider

        # SUT proxied (relayed) by HF to a provider like Nebius
        if provider:
            return HuggingFaceChatCompletionServerlessSUTMaker.make_sut(model_name, provider)
        else:
            return HuggingFaceChatCompletionDedicatedSUTMaker.make_sut(model_name)

    @staticmethod
    def get_secrets():
        hf_token = InjectSecret(HuggingFaceInferenceToken)
        return hf_token

    @staticmethod
    def exists(name: str) -> bool:
        found = False
        found_models = hfh.list_models(search=name, limit=1)
        print(found_models)
        for _ in found_models:
            found = True
            break
        return found


class HuggingFaceChatCompletionServerlessSUTMaker(HuggingFaceSUTMaker):

    @staticmethod
    def find(model_name, provider, find_alternative: bool = False) -> str | None:
        name = SUTMaker.extract_model_name(model_name)
        found_provider = None
        try:
            inference_providers = hfh.model_info(name, expand="inferenceProviderMapping")
            found = inference_providers.inference_provider_mapping.get(provider, None)
            if found:
                found_provider = provider
            else:
                if find_alternative:
                    for alt_provider, _ in inference_providers.inference_provider_mapping.items():
                        found_provider = str(alt_provider)
                        break  # we just grab the first one; is that the right choice?
        except Exception as e:
            logging.error(
                f"Error looking up inference providers for {model_name} aka {name} and provider {provider}: {e}"
            )
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
                SUTMaker.extract_model_name(model_name),
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
