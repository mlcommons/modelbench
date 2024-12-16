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
