import dataclasses
from enum import Enum
from typing import Type

from newhelm.secret_values import InjectSecret
from newhelm.sut_registry import SUTS
from newhelm.suts.huggingface_client import HuggingFaceSUT, HuggingFaceToken
from newhelm.suts.together_client import TogetherApiKey, TogetherCompletionsSUT


@dataclasses.dataclass
class SutDescription:
    key: str
    display_name: str
    newhelm_class: Type = None
    newhelm_key: str = None

    def __hash__(self):
        return super().__hash__()


class NewhelmSut(SutDescription, Enum):
    # Models commented out work except for RealToxicityPrompts, so we can enable them once we switch to a new test
    # ALPACA_7B = "alpaca-7b", "Stanford Alpaca (7B)", TogetherCompletionsSUT, "togethercomputer/alpaca-7b"
    GPT2 = "gpt2", "OpenAI GPT-2"
    LLAMA_2_7B = "llama-2-7b", "Meta Llama 2, 7b parameters"
    LLAMA_2_13B = "llama-2-13b", "Meta Llama 2, 13b parameters"
    LLAMA_2_70B = "llama-2-70b", "Meta Llama 2, 70b parameters"
    MISTRAL_7B = "mistral-7b", "Mistral 7B v0.1", TogetherCompletionsSUT, "mistralai/Mistral-7B-v0.1"
    # MISTRAL_8X_7B = "mistral-8x-7b", "Mixtral-8x7B (46.7B)", TogetherCompletionsSUT, "mistralai/Mixtral-8x7B-v0.1" # too many 503 errors
    # MISTRAL_7B_INSTRUCT = (
    #     "mistral-7b-instruct",
    #     "Mistral 7B Instruct v0.2",
    #     TogetherChatSUT,
    #     "mistralai/Mistral-7B-Instruct-v0.2"
    # )
    # MISTRAL_8X_7B_INSTRUCT = (
    #     "mistral-8x-7b-instruct",
    #     "Mixtral-8x7B Instruct (46.7B)",
    #     TogetherCompletionsSUT,
    #     "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # )
    MYTHOMAX = "mythomax-l2-13b", "MythoMax L2 13B", TogetherCompletionsSUT, "Gryphe/MythoMax-L2-13b"
    OLMO_7B = "olmo-7b", "AllenAI Olmo (7B)", TogetherCompletionsSUT, "allenai/OLMo-7B-Instruct"
    PYTHIA_70M = "pythia-70m", "EleutherAI Pythia, 70m parameters", HuggingFaceSUT, "EleutherAI/pythia-70m"
    QWEN_7B = "qwen-7b", "Qwen 1.5 (7B)", TogetherCompletionsSUT, "Qwen/Qwen1.5-7B"
    # REDPAJAMA_7B = "redpajama-7b", "Together RedPajama INCITE 7B", TogetherCompletionsSUT, "togethercomputer/RedPajama-INCITE-7B-Base"
    NOUS = "nous-7b", "Together StripedHyena Nous", TogetherCompletionsSUT, "togethercomputer/StripedHyena-Nous-7B"
    YI_BASE_6B = "yi-base-6b", "01-ai Yi Base (6B)", TogetherCompletionsSUT, "zero-one-ai/Yi-6B"


for sut in NewhelmSut:
    required_secrets = {
        TogetherCompletionsSUT: (InjectSecret(TogetherApiKey),),
        HuggingFaceSUT: (InjectSecret(HuggingFaceToken),),
    }
    if sut.newhelm_class is not None and sut.newhelm_key is not None:
        SUTS.register(sut.newhelm_class, sut.newhelm_key, *required_secrets[sut.newhelm_class])
