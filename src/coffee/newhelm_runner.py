import dataclasses
from enum import Enum


@dataclasses.dataclass
class NewSutDescription:
    key: str
    display_name: str

    def __hash__(self):
        return super().__hash__()


class NewhelmSut(NewSutDescription, Enum):
    GPT2 = "gpt2", "OpenAI GPT-2"
    LLAMA_2_7B = "llama-2-7b", "Meta Llama 2, 7b parameters"
