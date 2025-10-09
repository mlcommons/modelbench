import pytest

from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.sut_registry import SUTS
from modelgauge_tests.fake_sut import FakeSUT

# Need to declare global here because session start hook can't access fixtures.
_SUT_UID = "fake-sut"


def pytest_sessionstart(session):
    """Register the fake SUT during the session start."""
    SUTS.register(FakeSUT, _SUT_UID)


def pytest_sessionfinish(session, exitstatus):
    """Remove fake SUTs from registry."""
    del SUTS._lookup[_SUT_UID]


@pytest.fixture(scope="session")
def sut_uid():
    return _SUT_UID


@pytest.fixture
def sut(sut_uid):
    return FakeSUT(sut_uid)


@pytest.fixture
def isolated_annotators():
    snapshot = ANNOTATORS._lookup.copy()
    try:
        yield ANNOTATORS
    finally:
        ANNOTATORS._lookup.clear()
        ANNOTATORS._lookup.update(snapshot)
