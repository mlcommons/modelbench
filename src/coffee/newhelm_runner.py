import dataclasses
from enum import Enum
from typing import Type, Union

from newhelm.secret_values import InjectSecret
from newhelm.sut_registry import SUTS
from newhelm.suts.huggingface_client import HuggingFaceSUT, HuggingFaceToken
from newhelm.suts.together_client import TogetherApiKey, TogetherCompletionsSUT


@dataclasses.dataclass
class SutDescription:
    key: str
    display_name: str
    newhelm_class: Union[None, Type] = None
    newhelm_key: str = ""

    def __hash__(self):
        return super().__hash__()


class NewhelmSut(SutDescription, Enum):
    ALPACA_7B = "alpaca-7b", "Stanford Alpaca (7B)"
    DEEPSEEK_67B = (
        "deepseek-67b",
        "DeepSeek LLM Chat (67B)",
        TogetherCompletionsSUT,
        "deepseek-ai/deepseek-llm-67b-chat",
    )
    LLAMA_2_7B = "llama-2-7b-chat", "Meta Llama 2, 7b parameters"
    LLAMA_2_13B = "llama-2-13b-chat", "Meta Llama 2, 13b parameters"
    LLAMA_2_70B = "llama-2-70b-chat", "Meta Llama 2, 70b parameters"
    MISTRAL_7B = "mistral-7b", "Mistral 7B Instruct v0.2", TogetherCompletionsSUT, "mistralai/Mistral-7B-Instruct-v0.2"
    MIXTRAL_8X_7B = (
        "mixtral-8x-7b",
        "Mixtral-8x7B Instruct v0.1 (46.7B)",
        TogetherCompletionsSUT,
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    )
    QWEN_72B = "qwen-72b", "Qwen 1.5 (72B)", TogetherCompletionsSUT, "Qwen/Qwen1.5-72B-Chat"
    NOUS = (
        "stripedhyena-nous-7b",
        "Together StripedHyena Nous 7B",
        TogetherCompletionsSUT,
        "togethercomputer/StripedHyena-Nous-7B",
    )
    WIZARDLM_13B = "wizardlm-13b", "WizardLM v1.2 (13B)", TogetherCompletionsSUT, "WizardLM/WizardLM-13B-V1.2"
    YI_34B_CHAT = "yi-34b", "01-ai Yi Chat (34B)", TogetherCompletionsSUT, "zero-one-ai/Yi-34B-Chat"


for sut in NewhelmSut:
    required_secrets = {
        TogetherCompletionsSUT: (InjectSecret(TogetherApiKey),),
        HuggingFaceSUT: (InjectSecret(HuggingFaceToken),),
    }
    if sut.newhelm_class is not None and sut.key is not None and sut.key not in dict(SUTS.items()):
        # print(f"registering {sut} to {list(dict(SUTS.items()).keys())}")
        SUTS.register(sut.newhelm_class, sut.key, sut.newhelm_key, *required_secrets[sut.newhelm_class])
