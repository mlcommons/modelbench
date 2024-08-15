import dataclasses
from enum import Enum
from typing import Type, Union

from modelgauge.secret_values import InjectSecret
from modelgauge.sut_registry import SUTS
from modelgauge.suts.together_client import TogetherApiKey, TogetherCompletionsSUT, TogetherChatSUT


@dataclasses.dataclass
class SutDescription:
    key: str
    modelgauge_class: Union[None, Type] = None
    modelgauge_key: str = ""

    def __hash__(self):
        return super().__hash__()


class ModelGaugeSut(SutDescription, Enum):
    ALPACA_7B = "alpaca-7b"
    CHRONOS_HERMES_13B = "chronos-hermes-13b", TogetherChatSUT, "Austism/chronos-hermes-13b"
    DEEPSEEK_67B = "deepseek-67b", TogetherChatSUT, "deepseek-ai/deepseek-llm-67b-chat"
    GEMMA_7B = "gemma-7b", TogetherChatSUT, "google/gemma-7b-it"
    LLAMA_2_7B = "llama-2-7b-chat"
    LLAMA_2_13B = "llama-2-13b-chat"
    LLAMA_2_70B = "llama-2-70b-chat"
    MISTRAL_7B = "mistral-7b", TogetherChatSUT, "mistralai/Mistral-7B-Instruct-v0.2"
    MIXTRAL_8X_7B = "mixtral-8x-7b", TogetherChatSUT, "mistralai/Mixtral-8x7B-Instruct-v0.1"
    OPENCHAT_3_5 = "openchat-3_5", TogetherChatSUT, "openchat/openchat-3.5-1210"
    QWEN_72B = "qwen-72b", TogetherChatSUT, "Qwen/Qwen1.5-72B-Chat"
    NOUS = "stripedhyena-nous-7b", TogetherChatSUT, "togethercomputer/StripedHyena-Nous-7B"
    VICUNA_13B = "vicuna-13b", TogetherChatSUT, "lmsys/vicuna-13b-v1.5"
    WIZARDLM_13B = "wizardlm-13b", TogetherChatSUT, "WizardLM/WizardLM-13B-V1.2"

    def instance(self, secrets):
        if not hasattr(self, "_instance"):
            if not secrets:
                return None
            self._instance = SUTS.make_instance(self.key, secrets=secrets)
        return self._instance

    def instance_initialization(self):
        instance = self.instance(None)
        if instance:
            return instance.initialization_record


for sut in ModelGaugeSut:
    required_secrets = {
        TogetherCompletionsSUT: (InjectSecret(TogetherApiKey),),
        TogetherChatSUT: (InjectSecret(TogetherApiKey),),
    }
    if sut.modelgauge_class is not None and sut.key is not None and sut.key not in dict(SUTS.items()):
        SUTS.register(sut.modelgauge_class, sut.key, sut.modelgauge_key, *required_secrets[sut.modelgauge_class])
