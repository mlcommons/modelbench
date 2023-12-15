from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, List


@dataclass(frozen=True)
class Prompt:
    """What actually goes to the SUT."""

    text: str
    truncated: bool = False


@dataclass(frozen=True, kw_only=True)
class PromptTemplate:
    """All the pieces necessary for a SUT to construct a Prompt in its own form."""

    instructions_block: str = ""
    """Instructions for the task."""

    train_instance_blocks: List[str] = field(default_factory=list)
    """Train instance blocks for the prompt."""

    eval_instance_block: str
    """Evaluation instance."""


@dataclass(frozen=True)
class WindowServiceConfig:
    max_sequence_length: int


class BaseTokenizer(ABC):
    @abstractmethod
    def to_tokens(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def from_tokens(self, tokens: List[str]):
        pass


class PlaceholderTokenizer(BaseTokenizer):
    def to_tokens(self, text: str) -> List[str]:
        # Placeholder - every character is a token.
        return list(text)

    def from_tokens(self, tokens: List[str]):
        # Placeholder - join all tokens together with no whitespace.
        return "".join(tokens)


class LocalWindowService:
    """This is roughly copied over from HELM's local_window_service with simplifications."""

    def __init__(self, tokenizer: BaseTokenizer, config: WindowServiceConfig):
        self.tokenizer = tokenizer
        self.config = config

    def fits_within_context_window(self, text: str) -> bool:
        return len(self.tokenizer.to_tokens(text)) <= self.config.max_sequence_length

    def truncate_from_right(self, text: str) -> str:
        tokens = self.tokenizer.to_tokens(text)
        if len(tokens) <= self.config.max_sequence_length:
            return text
        return self.tokenizer.from_tokens(tokens[: self.config.max_sequence_length])

    def truncate_training_then_from_right(
        self,
        prompt_template: PromptTemplate,
        template_to_string: Callable[[PromptTemplate], str],
    ) -> str:
        """Copied over from HELM's InContextLearningAdapter._make_prompt_fit.

        One big difference is passing in template_to_string, which lets the individual SUT
        decide how to convert the PromptTemplate to a string.
        """
        while len(prompt_template.train_instance_blocks) > 0:
            prompt_text = template_to_string(prompt_template)
            if self.fits_within_context_window(text=prompt_text):
                return prompt_text

            # Remove the last training example
            without_last_training_block = prompt_template.train_instance_blocks[:-1]
            prompt_template = replace(
                prompt_template, train_instance_blocks=without_last_training_block
            )

        # If removing the in-context example is still not enough, we simply truncate the prompt.
        # Following the default truncation strategy used by HuggingFace, we truncate the text from the right.
        prompt_text = template_to_string(prompt_template)
        return self.truncate_from_right(prompt_text)
