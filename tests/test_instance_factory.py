from dataclasses import dataclass
import pytest
from newhelm.instance_factory import FactoryEntry, InstanceFactory


@dataclass(frozen=True)
class MockClass:
    arg1: str = "1"
    arg2: str = "2"
    arg3: str = "3"


def test_register_and_make():
    factory = InstanceFactory[MockClass]()
    factory.register("key", MockClass)
    assert factory.make_instance("key") == MockClass()


def test_register_and_make_using_args():
    factory = InstanceFactory[MockClass]()
    factory.register("key", MockClass, "a", "b", "c")
    assert factory.make_instance("key") == MockClass("a", "b", "c")


def test_register_and_make_using_kwargs():
    factory = InstanceFactory[MockClass]()
    factory.register("key", MockClass, arg1="a", arg2="b", arg3="c")
    assert factory.make_instance("key") == MockClass("a", "b", "c")


def test_register_and_make_using_args_and_kwargs():
    factory = InstanceFactory[MockClass]()
    factory.register("key", MockClass, "a", "b", arg3="c")
    assert factory.make_instance("key") == MockClass("a", "b", "c")


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
        factory.make_instance("another-key")
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
