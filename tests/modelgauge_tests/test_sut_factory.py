import pytest

from modelgauge.sut import SUT
from modelgauge.sut_factory import SUTFactory, SUTNotFoundException, SUTType

KNOWN_UID = "known"
UNKNOWN_UID = "pleasedontregisterasutwiththisuid"


@pytest.fixture
def sut_factory():
    """Fixture to reset the SUT factory before each test. Simulates the SUTs global without contaminating it."""
    # SUTS._lookup.clear()
    factory = SUTFactory()
    factory.register(SUT, KNOWN_UID)
    return factory


def test_classify(sut_factory):
    assert sut_factory._classify_sut_uid(KNOWN_UID) == SUTType.KNOWN
    assert sut_factory._classify_sut_uid("google:gemma:nebius:hfrelay") == SUTType.DYNAMIC
    assert sut_factory._classify_sut_uid(UNKNOWN_UID) == SUTType.UNKNOWN


def test_make_instance_preregistered(sut_factory):
    sut = sut_factory.make_instance(KNOWN_UID, secrets={})
    assert isinstance(sut, SUT)


def test_make_instance_dynamic(sut_factory):
    # TODO
    pass


def test_make_instance_unknown_type(sut_factory):
    with pytest.raises(SUTNotFoundException):
        sut_factory.make_instance(UNKNOWN_UID, secrets={})
