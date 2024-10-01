import dataclasses
import functools

from modelgauge.secret_values import InjectSecret
from modelgauge.sut_registry import SUTS
from modelgauge.suts.together_client import TogetherApiKey, TogetherCompletionsSUT, TogetherChatSUT


@dataclasses.dataclass
class SutDescription:
    key: str

    @property
    def uid(self):
        return self.key


@dataclasses.dataclass
class ModelGaugeSut(SutDescription):
    @classmethod
    @functools.cache
    def for_key(cls, key: str) -> "ModelGaugeSut":
        valid_keys = [item[0] for item in SUTS.items()]
        if key in valid_keys:
            return ModelGaugeSut(key)
        else:
            raise ValueError(f"Unknown SUT {key}; valid keys are {valid_keys}")

    def __hash__(self):
        return self.key.__hash__()

    def instance(self, secrets):
        if not hasattr(self, "_instance"):
            if secrets is None:
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

SUTS_FOR_V_0_5 = [
    ModelGaugeSut.for_key(k)
    for k in [
        "deepseek-67b",
        "llama-2-13b-chat",
        "mistral-7b",
        "mixtral-8x-7b",
        "qwen-72b",
    ]
]

# Our main model runner, TogetherAI, deprecated a number of models due to lack of use circa 2024-08-29.
# This is the original set of models used for the published v0.5 benchmark.
#
ORIGINAL_SUT_UIDS_FOR_V_0_5 = [
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
