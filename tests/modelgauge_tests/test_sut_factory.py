import pytest

from modelgauge.sut import SUT, SUTNotFoundException
from modelgauge.sut_factory import SUTFactory, SUTType


@pytest.fixture
def sut_factory():
    """Fixture to reset the SUT factory before each test. Simulates the SUTs global without contaminating it."""
    # SUTS._lookup.clear()
    factory = SUTFactory()
    return factory


def test_classify(sut_factory):
    sut_factory.register(SUT, "known", "something")
    assert sut_factory._classify_sut_uid("known") == SUTType.KNOWN
    assert sut_factory._classify_sut_uid("google:gemma:nebius:hfrelay") == SUTType.DYNAMIC
    assert sut_factory._classify_sut_uid("pleasedontregisterasutwiththisuid") == SUTType.UNKNOWN
