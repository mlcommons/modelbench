from abc import ABC, abstractmethod


class SUTCapability(ABC):
    """Base class for defining a capability that SUTs may have and Tests may need."""

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """Describe why to mark a SUT/Test as having/needing this capability."""
        pass


class AcceptsTextPrompt(SUTCapability):
    """The capability to take a `TextPrompt` as input.

    SUTs that report this capability must implement `translate_text_prompt()`.
    """

    @classmethod
    def description(cls) -> str:
        return "These SUTs can take a `TextPrompt` as input."


class AcceptsChatPrompt(SUTCapability):
    """The capability to take a `ChatPrompt` as input.

    SUTs that report this capability must implement `translate_chat_prompt()`.
    """

    @classmethod
    def description(cls) -> str:
        return "These SUTs can take a `ChatPrompt` as input."


class ProducesPerTokenLogProbabilities(SUTCapability):
    """The capability to produce per-token log probabilities.

    SUTs that report this capability must set the `top_logprobs` field in SUTResponse, if logprobs are requested.
    """

    @classmethod
    def description(cls) -> str:
        return "These SUTs set the 'top_logprobs' field in SUTResponse."
