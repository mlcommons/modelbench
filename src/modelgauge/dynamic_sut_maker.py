from abc import ABC, abstractmethod

from modelgauge.secret_values import InjectSecret

SEPARATOR = ":"


class ModelNotSupportedError(Exception):
    """Use when requesting a dynamic SUT from a correct proxy (e.g. Huggingface)
    and a correct provider (e.g. nebius, cohere) that doesn't support that model."""

    pass


class ProviderNotFoundError(Exception):
    """Use when requesting a dynamic SUT from a correct proxy (e.g. Huggingface)
    with an unknown or inactive provider (e.g. nebius, cohere)."""

    pass


class UnknownProxyError(Exception):
    """Use when requesting a dynamic SUT that can't be created because the proxy
    isn't known, e.g. for now it's not hf"""

    pass


# non-exhaustive list of strings that identify service providers, drivers, and model makers
# used as hints to disambiguate a SUT UID if needed
KNOWN_PROVIDERS = {
    "hf",
    "cerebras",
    "falai",
    "fal-ai",
    "fireworks",
    "hfinference",
    "hf-inference",
    "hyperbolic",
    "together",
    "baseten",
    "azure",
    "cohere",
    "nebius",
    "mistralai",
    "vertexai",
    "novita",
    "sambanova",
    "replicate",
}
KNOWN_DRIVERS = {"hfrelay"}
KNOWN_VENDORS = {
    "openai",
    "google",
    "meta",
    "microsoft",
    "deepseek",
    "mistralai",
    "nvidia",
    "alibaba",
    "zhipu",
    "cohere",
    "ibm",
    "internlm",
    "ai2",
    "ai21labs",
    "01ai",
}


class DynamicSUTMaker(ABC):

    @staticmethod
    @abstractmethod
    def get_secrets() -> InjectSecret:
        pass

    @staticmethod
    @abstractmethod
    def find(name: str):
        pass

    @staticmethod
    @abstractmethod
    def make_sut(name: str):
        pass
