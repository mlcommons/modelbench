from dataclasses import dataclass
import pytest
from newhelm.instance_factory import FactoryEntry, InstanceFactory
from newhelm.secret_values import InjectSecret
from tests.fake_secrets import FakeRequiredSecret


@dataclass(frozen=True)
class MockClass:
    arg1: str = "1"
    arg2: str = "2"
    arg3: str = "3"


def test_register_and_make():
    factory = InstanceFactory[MockClass]()
    factory.register("key", MockClass)
    assert factory.make_instance("key", secrets={}) == MockClass()


def test_register_and_make_using_args():
    factory = InstanceFactory[MockClass]()
    factory.register("key", MockClass, "a", "b", "c")
    assert factory.make_instance("key", secrets={}) == MockClass("a", "b", "c")


def test_register_and_make_using_kwargs():
    factory = InstanceFactory[MockClass]()
    factory.register("key", MockClass, arg1="a", arg2="b", arg3="c")
    assert factory.make_instance("key", secrets={}) == MockClass("a", "b", "c")


def test_register_and_make_using_args_and_kwargs():
    factory = InstanceFactory[MockClass]()
    factory.register("key", MockClass, "a", "b", arg3="c")
    assert factory.make_instance("key", secrets={}) == MockClass("a", "b", "c")


def test_fails_same_key():
    factory = InstanceFactory[MockClass]()
    factory.register("some-key", MockClass)
    with pytest.raises(AssertionError) as err_info:
        factory.register("some-key", MockClass)
    assert (
        "Factory already contains some-key set to MockClass(args=(), kwargs={})."
        in str(err_info)
    )


def test_fails_missing_key():
    factory = InstanceFactory[MockClass]()
    factory.register("some-key", MockClass)

    with pytest.raises(KeyError) as err_info:
        factory.make_instance("another-key", secrets={})
    assert "No registration for another-key. Known keys: ['some-key']" in str(err_info)


def test_lists_all_items():
    factory = InstanceFactory[MockClass]()
    factory.register("k1", MockClass, "v1")
    factory.register("k2", MockClass, "v2")
    factory.register("k3", MockClass, "v3")
    assert factory.items() == [
        ("k1", FactoryEntry(MockClass, args=("v1",), kwargs={})),
        ("k2", FactoryEntry(MockClass, args=("v2",), kwargs={})),
        ("k3", FactoryEntry(MockClass, args=("v3",), kwargs={})),
    ]


def test_factory_entry_str():
    entry = FactoryEntry(MockClass, args=("v1",), kwargs={"arg2": "v2"})
    assert str(entry) == "MockClass(args=('v1',), kwargs={'arg2': 'v2'})"


class NeedsSecrets:
    def __init__(self, arg1: str, arg2: FakeRequiredSecret):
        self.arg1 = arg1
        self.secret = arg2.value


def test_injection():
    factory = InstanceFactory[NeedsSecrets]()
    factory.register("k1", NeedsSecrets, "v1", InjectSecret(FakeRequiredSecret))
    factory.register("k2", NeedsSecrets, "v2", arg2=InjectSecret(FakeRequiredSecret))
    secrets = {"some-scope": {"some-key": "some-value"}}
    k1_obj = factory.make_instance("k1", secrets=secrets)
    assert k1_obj.arg1 == "v1"
    assert k1_obj.secret == "some-value"
    k2_obj = factory.make_instance("k2", secrets=secrets)
    assert k2_obj.arg1 == "v2"
    assert k2_obj.secret == "some-value"


class KwargsSecrets:
    def __init__(self, arg1: str, **kwargs):
        self.arg1 = arg1
        self.kwargs = kwargs


def test_kwargs_injection():
    factory = InstanceFactory[KwargsSecrets]()
    factory.register("k1", KwargsSecrets, "v1")
    factory.register(
        "k2", KwargsSecrets, "v2", fake_secret=InjectSecret(FakeRequiredSecret)
    )
    secrets = {"some-scope": {"some-key": "some-value"}}
    k1_obj = factory.make_instance("k1", secrets=secrets)
    assert k1_obj.arg1 == "v1"
    assert k1_obj.kwargs == {}
    k2_obj = factory.make_instance("k2", secrets=secrets)
    assert k2_obj.arg1 == "v2"
    assert k2_obj.kwargs == {"fake_secret": FakeRequiredSecret("some-value")}
