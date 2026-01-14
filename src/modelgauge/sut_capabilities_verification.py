from modelgauge.sut import SUT
from modelgauge.sut_capabilities import SUTCapability


def assert_sut_capabilities(sut: SUT, required_capabilities: Sequence[Type[SUTCapability]]):
    """Raise a MissingSUTCapabilities if `sut` doesn't implement all required capabilities.."""
    missing = []
    for capability in required_capabilities:
        if capability not in sut.capabilities:
            missing.append(capability)
    if missing:
        raise MissingSUTCapabilities(sut_uid=sut.uid, missing=missing)


def assert_multiple_suts_capabilities(suts: Sequence[SUT], required_capabilities: Sequence[Type[SUTCapability]]):
    """Raise a MissingSUTCapabilities if `sut` doesn't implement all required capabilities.."""
    missing = []
    for sut in suts:
        try:
            assert_sut_capabilities(sut, required_capabilities)
        except MissingSUTCapabilities as e:
            missing.append(e)
    if len(missing) == 1:
        raise missing[0]
    elif len(missing) > 1:
        raise MissingMultipleSUTsCapabilities(missing)


def sut_is_capable(sut: SUT, required_capabilities: Sequence[Type[SUTCapability]]) -> bool:
    """Return True if `sut` can handle `test`."""
    try:
        assert_sut_capabilities(sut, required_capabilities)
        return True
    except MissingSUTCapabilities:
        return False


def get_capable_suts(suts: Sequence[SUT], required_capabilities: Sequence[Type[SUTCapability]]) -> Sequence[SUT]:
    """Filter `suts` to only those that can do `test`."""
    return [sut for sut in suts if sut_is_capable(sut, required_capabilities)]


class MissingSUTCapabilities(AssertionError):
    def __init__(self, sut_uid: str, missing: Sequence[Type[SUTCapability]]):
        self.sut_uid = sut_uid
        self.missing = missing

    def __str__(self):
        missing_names = [m.__name__ for m in self.missing]
        return f"{self.sut_uid} is missing the following required capabilities: {missing_names}."


class MissingMultipleSUTsCapabilities(MissingSUTCapabilities):
    def __init__(self, missing_exceptions: Sequence[MissingSUTCapabilities]):
        self.missing_exceptions = missing_exceptions

    def __str__(self):
        messages = [str(e) for e in self.missing_exceptions]
        return "Multiple SUTs are missing required capabilities:\n" + "\n".join(messages)
