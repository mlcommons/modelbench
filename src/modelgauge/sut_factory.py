from enum import Enum

from modelgauge.instance_factory import InstanceFactory
from modelgauge.secret_values import RawSecrets
from modelgauge.sut import SUT


class SUTType(Enum):
    DYNAMIC = "dynamic"
    KNOWN = "known"
    UNKNOWN = "unknown"


class SUTFactory(InstanceFactory[SUT]):
    """A factory for both pre-registered and dynamic SUTs."""

    def make_instance(self, uid: str, *, secrets: RawSecrets) -> SUT:
        sut_type = self._classify_sut_uid(uid)
        if sut_type == SUTType.KNOWN:
            # Create SUT from the registry.
            return super().make_instance(uid, secrets=secrets)
        elif sut_type == SUTType.DYNAMIC:
            return self._make_dynamic_sut(uid)

    def _classify_sut_uid(self, uid: str) -> SUTType:
        if uid in self.keys():
            return SUTType.KNOWN
        elif ":" in uid:
            return SUTType.DYNAMIC
        else:
            return SUTType.UNKNOWN

    def _make_dynamic_sut(self, uid: str) -> SUT:
        pass


# The list of all SUT instances with assigned UIDs.
SUTS = SUTFactory()
