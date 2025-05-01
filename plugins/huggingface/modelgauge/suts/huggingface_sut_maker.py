from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import SUTMaker
from modelgauge.suts.huggingface_chat_completion import (
    HuggingFaceChatCompletionDedicatedSUT,
    HuggingFaceChatCompletionServerlessSUT,
)

# USAGE
# SUTS.register(*SUTMaker.make_sut("huggingface", "SomeVendor/some-model", HF_SECRET))


class HuggingFaceSUTMaker(SUTMaker):

    @staticmethod
    def make_sut_id(model_name: str):
        chunks = []

        proxy, provider, vendor, model = SUTMaker.parse_sut_name(model_name)
        if vendor:
            chunks.append(vendor)
        chunks.append(model)
        if "hf" not in chunks:
            chunks.append("hf")
        if provider and provider != "hf":
            chunks.append(provider)
        return "-".join(chunks).lower()

    @staticmethod
    def make_sut(model_name: str, provider: str = ""):
        proxy, implicit_provider, vendor, model = SUTMaker.parse_sut_name(model_name)
        if implicit_provider and not provider:
            provider = implicit_provider

        # SUT proxied (relayed) by HF to a provider like Nebius
        if provider:
            return HuggingFaceSUTMaker.make_serverless_sut(model_name, provider)
        else:
            return HuggingFaceSUTMaker.make_dedicated_sut(model_name)

    @staticmethod
    def make_serverless_sut(model_name: str, provider: str = ""):
        sut_id = HuggingFaceSUTMaker.make_sut_id(model_name)
        if not model_name.lower().startswith(f"hf/{provider.lower()}"):
            model_full_name = f"hf/{provider}/{model_name}"
        else:
            model_full_name = model_name
        sut_id = HuggingFaceSUTMaker.make_sut_id(model_full_name)
        return (
            HuggingFaceChatCompletionServerlessSUT,
            sut_id,
            model_name,
            provider,
            InjectSecret(HuggingFaceInferenceToken),
        )

    @staticmethod
    def make_dedicated_sut(model):
        # conceivably look up available dedicated inference endpoints
        # and return a SUT if one was found or could be created
        raise NotImplementedError(f"Dynamic SUTs with dedicated HF endpoints aren't implemented yet.")
