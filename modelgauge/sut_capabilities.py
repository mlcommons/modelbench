from abc import ABC, abstractmethod


class SUTCapability(ABC):
    """Base class for defining a capability that SUTs may have and Tests may need."""

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """Describe why to mark a SUT/Test as having/needing this capability."""
        pass


class AcceptsTextPrompt(SUTCapability):
    @classmethod
    def description(cls) -> str:
        return "These SUTs can take a `TextPrompt` as input."


class AcceptsChatPrompt(SUTCapability):
    @classmethod
    def description(cls) -> str:
        return "These SUTs can take a `ChatPrompt` as input."


class ProducesPerTokenLogProbabilities(SUTCapability):
    @classmethod
    def description(cls) -> str:
        return "These SUTs set the 'top_logprobs' field in SUTResponse."
