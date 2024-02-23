import dataclasses
from enum import Enum

from newhelm.sut_registry import SUTS
from newhelm.suts.huggingface_client import HuggingFaceSUT


@dataclasses.dataclass
class SutDescription:
    key: str
    display_name: str

    def __hash__(self):
        return super().__hash__()


SUTS.register("pythia-70m", HuggingFaceSUT, "EleutherAI/pythia-70m")


class NewhelmSut(SutDescription, Enum):
    GPT2 = "gpt2", "OpenAI GPT-2"
    LLAMA_2_7B = "llama-2-7b", "Meta Llama 2, 7b parameters"
    PYTHIA_70M = "pythia-70m", "EleutherAI Pythia, 70m parameters"
