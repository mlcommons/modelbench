from enum import Enum

from modelgauge.config import load_secrets_from_config
from modelgauge.dynamic_sut_factory import DynamicSUTFactory, UnknownSUTMakerError
from modelgauge.dynamic_sut_metadata import DynamicSUTMetadata
from modelgauge.secret_values import RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_registry import SUTS
from modelgauge.suts.huggingface_sut_factory import HuggingFaceSUTFactory
from modelgauge.suts.openai_sut_factory import OpenAISUTFactory
from modelgauge.suts.together_sut_factory import TogetherSUTFactory


class SUTNotFoundException(Exception):
    pass


class SUTType(Enum):
    DYNAMIC = "dynamic"
    KNOWN = "known"
    UNKNOWN = "unknown"


# TODO: Auto-collect. Maybe make "driver" keys a constant in each factory module.
# Maps a string to the module and factory function in that module
# that can be used to create a dynamic sut
DYNAMIC_SUT_FACTORIES: dict = {
    "proxied": {"hfrelay": HuggingFaceSUTFactory},
    "direct": {
        "openai": OpenAISUTFactory,
        "together": TogetherSUTFactory,
        "huggingface": HuggingFaceSUTFactory,
        "hf": HuggingFaceSUTFactory,
    },
}


class SUTFactory:
    """A factory for both pre-registered and dynamic SUTs."""

    def __init__(self, sut_registry):
        self.sut_registry = sut_registry
        self.dynamic_sut_factories = self._load_dynamic_sut_factories(load_secrets_from_config())

    def _load_dynamic_sut_factories(self, secrets: RawSecrets) -> dict[str, dict[str, DynamicSUTFactory]]:
        factories: dict[str, dict[str, DynamicSUTFactory]] = {"direct": {}, "proxied": {}}
        for driver, factory_class in DYNAMIC_SUT_FACTORIES["direct"].items():
            factories["direct"][driver] = factory_class(secrets)
        for driver, factory_class in DYNAMIC_SUT_FACTORIES["proxied"].items():
            factories["proxied"][driver] = factory_class(secrets)
        return factories

    def knows(self, uid: str) -> bool:
        """Check if the registry knows about a given SUT UID. Dynamic SUTs are always considered known."""
        if self._classify_sut_uid(uid) == SUTType.DYNAMIC:
            return True
        return self.sut_registry.knows(uid)

    def make_instance(self, uid: str, *, secrets: RawSecrets) -> SUT:
        sut_type = self._classify_sut_uid(uid)
        if sut_type == SUTType.KNOWN:
            return self.sut_registry.make_instance(uid, secrets=secrets)
        elif sut_type == SUTType.DYNAMIC:
            return self._make_dynamic_sut(uid)
        else:
            raise SUTNotFoundException(f"{uid} is neither pre-registered nor a valid dynamic SUT UID.")

    def _classify_sut_uid(self, uid: str) -> SUTType:
        if uid in self.sut_registry.keys():
            return SUTType.KNOWN
        elif ":" in uid:
            return SUTType.DYNAMIC
        else:
            return SUTType.UNKNOWN

    def _make_dynamic_sut(self, uid: str) -> SUT:
        sut_metadata: DynamicSUTMetadata = DynamicSUTMetadata.parse_sut_uid(uid)

        if sut_metadata.is_proxied():
            factory = self.dynamic_sut_factories["proxied"].get(sut_metadata.driver, None)
        else:
            factory = self.dynamic_sut_factories["direct"].get(sut_metadata.driver, None)  # type: ignore
        if not factory:
            raise UnknownSUTMakerError(f'Don\'t know how to make dynamic sut "{uid}"')
        return factory.make_sut(sut_metadata)

    def keys(self) -> list[str]:
        """Mimic the registry interface."""
        return self.sut_registry.keys()

    def get_missing_dependencies(self, uid: str, *, secrets: RawSecrets):
        """Mimic the registry interface. Only obtain missing secrets for PRE-REGISTERED SUTs."""
        if self._classify_sut_uid(uid) == SUTType.DYNAMIC:
            return []
        return self.sut_registry.get_missing_dependencies(uid, secrets=secrets)


SUT_FACTORY = SUTFactory(SUTS)
