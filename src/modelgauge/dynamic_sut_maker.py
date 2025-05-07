from abc import ABC, abstractmethod
from typing import List

from modelgauge.secret_values import InjectSecret


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


class DynamicSUTMaker(ABC):

    @staticmethod
    @abstractmethod
    def get_secrets() -> InjectSecret:
        pass

    @staticmethod
    def parse_sut_name(name: str) -> tuple[str, str, str, str]:
        """A dynamic SUT name looks like hf/nebius/google/gemma-3-27b-it
        hf = proxy (passes requests through to...)
        provider = nebius (runs model by...)
        vendor = google (creates model named...)
        model = gemma 3 27b it
        """

        chunks = name.split("/")
        match len(chunks):
            case 4:
                proxy, provider, vendor, model = chunks
            case 3:
                provider, vendor, model = chunks
                proxy = ""
            case 2:
                vendor, model = chunks
                proxy = provider = ""
            case 1:
                model = chunks[0]
                proxy = provider = vendor = ""
            case _:
                raise ValueError(f"Invalid SUT name string {name}")

        if not model:
            raise ValueError(f"Unable to parse a model name out of {name}")
        return proxy, provider, vendor, model

    @staticmethod
    def extract_model_name(name: str) -> str:
        _, _, vendor, model = DynamicSUTMaker.parse_sut_name(name)
        if vendor:
            return f"{vendor}/{model}"
        else:
            return model

    @staticmethod
    @abstractmethod
    def find(name: str):
        pass
