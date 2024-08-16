import dataclasses
import functools

from modelgauge.secret_values import InjectSecret
from modelgauge.sut_registry import SUTS
from modelgauge.suts.together_client import TogetherApiKey, TogetherCompletionsSUT, TogetherChatSUT


@dataclasses.dataclass
class SutDescription:
    key: str


@dataclasses.dataclass
class ModelGaugeSut(SutDescription):

    def __hash__(self):
        return super().__hash__()

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


def _register_required_suts():
    suts_to_register = {
        "deepseek-67b": (TogetherChatSUT, "deepseek-ai/deepseek-llm-67b-chat"),
        "gemma-7b": (TogetherChatSUT, "google/gemma-7b-it"),
        "mistral-7b": (TogetherChatSUT, "mistralai/Mistral-7B-Instruct-v0.2"),
        "mixtral-8x-7b": (TogetherChatSUT, "mistralai/Mixtral-8x7B-Instruct-v0.1"),
        "openchat-3_5": (TogetherChatSUT, "openchat/openchat-3.5-1210"),
        "qwen-72b": (TogetherChatSUT, "Qwen/Qwen1.5-72B-Chat"),
        "stripedhyena-nous-7b": (TogetherChatSUT, "togethercomputer/StripedHyena-Nous-7B"),
        "vicuna-13b": (TogetherChatSUT, "lmsys/vicuna-13b-v1.5"),
        "wizardlm-13b": (TogetherChatSUT, "WizardLM/WizardLM-13B-V1.2"),
    }

    required_secrets = {
        TogetherCompletionsSUT: (InjectSecret(TogetherApiKey),),
        TogetherChatSUT: (InjectSecret(TogetherApiKey),),
    }

    for key, details in suts_to_register.items():
        SUTS.register(details[0], key, details[1], *required_secrets[details[0]])


_register_required_suts()


@functools.cache
def sut_for_key(name: str) -> ModelGaugeSut:
    return ModelGaugeSut(name)


SUTS_FOR_V_0_5 = [
    sut_for_key(k)
    for k in [
        "alpaca-7b",
        "chronos-hermes-13b",
        "deepseek-67b",
        "gemma-7b",
        "llama-2-7b-chat",
        "llama-2-13b-chat",
        "llama-2-70b-chat",
        "mistral-7b",
        "mixtral-8x-7b",
        "openchat-3_5",
        "qwen-72b",
        "stripedhyena-nous-7b",
        "vicuna-13b",
        "wizardlm-13b",
    ]
]

# class ModelGaugeSut(SutDescription, Enum):
#     ALPACA_7B = "alpaca-7b" "qwen-72b"
#     CHRONOS_HERMES_13B = "chronos-hermes-13b", TogetherChatSUT, "Austism/chronos-hermes-13b"
#     DEEPSEEK_67B = "deepseek-67b", TogetherChatSUT, "deepseek-ai/deepseek-llm-67b-chat"
#     GEMMA_7B = "gemma-7b", TogetherChatSUT, "google/gemma-7b-it"
#     LLAMA_2_7B = "llama-2-7b-chat"
#     LLAMA_2_13B = "llama-2-13b-chat"
#     LLAMA_2_70B = "llama-2-70b-chat"
#     MISTRAL_7B = "mistral-7b", TogetherChatSUT, "mistralai/Mistral-7B-Instruct-v0.2"
#     MIXTRAL_8X_7B = "mixtral-8x-7b", TogetherChatSUT, "mistralai/Mixtral-8x7B-Instruct-v0.1"
#     OPENCHAT_3_5 = "openchat-3_5", TogetherChatSUT, "openchat/openchat-3.5-1210"
#     QWEN_72B = "qwen-72b", TogetherChatSUT, "Qwen/Qwen1.5-72B-Chat"
#     NOUS = "stripedhyena-nous-7b", TogetherChatSUT, "togethercomputer/StripedHyena-Nous-7B"
#     VICUNA_13B = "vicuna-13b", TogetherChatSUT, "lmsys/vicuna-13b-v1.5"
#     WIZARDLM_13B = "wizardlm-13b", TogetherChatSUT, "WizardLM/WizardLM-13B-V1.2"
#
#
# for sut in ModelGaugeSut:
#     required_secrets = {
#         TogetherCompletionsSUT: (InjectSecret(TogetherApiKey),),
#         TogetherChatSUT: (InjectSecret(TogetherApiKey),),
#     }
#     if sut.modelgauge_class is not None and sut.key is not None and sut.key not in dict(SUTS.items()):
#         SUTS.register(sut.modelgauge_class, sut.key, sut.modelgauge_key, *required_secrets[sut.modelgauge_class])
