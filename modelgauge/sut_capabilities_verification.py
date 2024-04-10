from modelgauge.base_test import BaseTest
from modelgauge.sut import SUT
from modelgauge.sut_capabilities import SUTCapability
from typing import Sequence, Type


def assert_sut_capabilities(sut: SUT, test: BaseTest):
    """Raise a MissingSUTCapabilities if `sut` can't handle `test."""
    missing = []
    for capability in test.requires_sut_capabilities:
        if capability not in sut.capabilities:
            missing.append(capability)
    if missing:
        raise MissingSUTCapabilities(
            sut_uid=sut.uid, test_uid=test.uid, missing=missing
        )


def sut_is_capable(test: BaseTest, sut: SUT) -> bool:
    """Return True if `sut` can handle `test`."""
    try:
        assert_sut_capabilities(sut, test)
        return True
    except MissingSUTCapabilities:
        return False


def get_capable_suts(test: BaseTest, suts: Sequence[SUT]) -> Sequence[SUT]:
    """Filter `suts` to only those that can do `test`."""
    return [sut for sut in suts if sut_is_capable(test, sut)]


class MissingSUTCapabilities(AssertionError):
    def __init__(
        self, sut_uid: str, test_uid: str, missing: Sequence[Type[SUTCapability]]
    ):
        self.sut_uid = sut_uid
        self.test_uid = test_uid
        self.missing = missing

    def __str__(self):
        missing_names = [m.__name__ for m in self.missing]
        return (
            f"Test {self.test_uid} cannot run on {self.sut_uid} because "
            f"it requires the following capabilities: {missing_names}."
        )
