from abc import ABC, abstractmethod

from modelgauge.dependency_injection import inject_dependencies
from modelgauge.secret_values import InjectSecret, RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_definition import SUTDefinition


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
    def __init__(self, raw_secrets: RawSecrets):
        self.raw_secrets = raw_secrets

    def injected_secrets(self):
        """Return the injected secrets as specified by `get_secrets`."""
        return inject_dependencies(self.get_secrets(), {}, secrets=self.raw_secrets)[0]

    @abstractmethod
    def get_secrets(self) -> list[InjectSecret]:
        pass

    # TODO: refactor this to use SUTDefinition instead of both the metadata and the kwargs
    @abstractmethod
    def make_sut(self, sut_definition: SUTDefinition) -> SUT:
        pass
