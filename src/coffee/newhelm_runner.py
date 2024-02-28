import dataclasses
from enum import Enum

from newhelm.sut_registry import SUTS
from newhelm.suts.huggingface_client import HuggingFaceSUT
from newhelm.suts.together_client import TogetherCompletionsSUT


@dataclasses.dataclass
class SutDescription:
    key: str
    display_name: str
    newhelm_class = None

    def __hash__(self):
        return super().__hash__()


SUTS.register("pythia-70m", HuggingFaceSUT, "EleutherAI/pythia-70m")
SUTS.register("mistral-7b", TogetherCompletionsSUT, "mistralai/Mistral-7B-v0.1")
# SUTS.register("redpajama-incite-3b", TogetherCompletionsSUT, "togethercomputer/RedPajama-INCITE-7B-Base")
SUTS.register("yi-base-6b", TogetherCompletionsSUT, "zero-one-ai/Yi-6B")


class NewhelmSut(SutDescription, Enum):
    GPT2 = "gpt2", "OpenAI GPT-2"
    LLAMA_2_7B = "llama-2-7b", "Meta Llama 2, 7b parameters"
    LLAMA_2_13B = "llama-2-13b", "Meta Llama 2, 13b parameters"
    LLAMA_2_70B = "llama-2-70b", "Meta Llama 2, 70b parameters"
    MISTRAL_7B = "mistral-7b", "Mistral (7B) v0.1"
    PYTHIA_70M = "pythia-70m", "EleutherAI Pythia, 70m parameters"
    # RED_PAJAMA_7B = "redpajama-incite-3b", "TogetherAI RedPajama-INCITE (3B)"
    YI_BASE_6B = "yi-base-6b", "01-ai Yi Base (6B)"
