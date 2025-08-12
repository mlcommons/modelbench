from unittest.mock import patch
import pytest

from modelgauge.dynamic_sut_factory import UnknownSUTMakerError
from modelgauge.instance_factory import InstanceFactory
from modelgauge.sut import SUT
from modelgauge.sut_factory import SUTFactory, SUTNotFoundException, SUTType
from modelgauge_tests.fake_sut import FakeSUT
from modelgauge_tests.test_dynamic_sut_factory import FakeDynamicFactory

KNOWN_UID = "known"
UNKNOWN_UID = "pleasedontregisterasutwiththisuid"
DYNAMIC_UID = "google:gemma:nebius:hfrelay"


@pytest.fixture
def sut_factory():
    """Fixture to simulates the SUTs global without contaminating it."""
    registry = InstanceFactory[SUT]()
    registry.register(SUT, KNOWN_UID)
    factory = SUTFactory(registry)
    return factory


@pytest.fixture
def sut_factory_dynamic():
    """SUT factory that patches the dynamic SUT factories."""
    registry = InstanceFactory[SUT]()
    dynamic_factories = {"driver1": FakeDynamicFactory({}), "driver2": FakeDynamicFactory({})}
    with patch(
        "modelgauge.sut_factory.SUTFactory._load_dynamic_sut_factories",
        return_value=dynamic_factories,
    ):
        sut_factory = SUTFactory(registry)
    return sut_factory


def test_classify(sut_factory):
    assert sut_factory._classify_sut_uid(KNOWN_UID) == SUTType.KNOWN
    assert sut_factory._classify_sut_uid(DYNAMIC_UID) == SUTType.DYNAMIC
    assert sut_factory._classify_sut_uid(UNKNOWN_UID) == SUTType.UNKNOWN


def test_knows(sut_factory):
    assert sut_factory.knows(KNOWN_UID) is True
    assert sut_factory.knows(DYNAMIC_UID) is True
    assert sut_factory.knows(UNKNOWN_UID) is False


def test_get_missing_dependencies_dynamic(sut_factory):
    assert sut_factory.get_missing_dependencies(DYNAMIC_UID, secrets={}) == []


def test_make_instance_preregistered(sut_factory):
    sut = sut_factory.make_instance(KNOWN_UID, secrets={})
    assert isinstance(sut, SUT)


def test_make_instance_dynamic(sut_factory_dynamic):
    sut = sut_factory_dynamic.make_instance("google/gemma:driver1", secrets={})
    assert isinstance(sut, FakeSUT)
    assert sut.uid == "google/gemma:driver1"


def test_make_instance_dynamic_unknown_driver(sut_factory_dynamic):
    with pytest.raises(UnknownSUTMakerError):
        sut_factory_dynamic.make_instance("google/gemma:unknown", secrets={})


def test_make_instance_unknown_type(sut_factory):
    with pytest.raises(SUTNotFoundException):
        sut_factory.make_instance(UNKNOWN_UID, secrets={})


# TODO: Add smoke tests?
