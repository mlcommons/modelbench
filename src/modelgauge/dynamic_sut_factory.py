from abc import ABC, abstractmethod

from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata

from modelgauge.secret_values import InjectSecret


class ModelNotSupportedError(Exception):
    """Use when requesting a dynamic SUT from a correct proxy (e.g. Huggingface)
    and/or a correct provider (e.g. nebius, cohere) that doesn't support that model."""

    pass


class ProviderNotFoundError(Exception):
    """Use when requesting a dynamic SUT from a correct proxy (e.g. Huggingface)
    with an unknown or inactive provider (e.g. nebius, cohere)."""

    pass


class UnknownSUTMakerError(Exception):
    """Use when requesting a dynamic SUT that can't be created because the proxy
    isn't known, or the requested provider is unknown"""

    pass


class DynamicSUTFactory(ABC):
    @staticmethod
    @abstractmethod
    def make_sut(sut_metadata: DynamicSUTMetadata):
        pass
