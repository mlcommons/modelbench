import pytest
from newhelm.instance_registery import InstanceRegistry


def test_register_and_get():
    registry = InstanceRegistry[str]()
    registry.register("key", "value")
    assert registry.get("key") == "value"


def test_fails_same_key():
    registry = InstanceRegistry[str]()
    registry.register("some-key", "some-value")
    with pytest.raises(AssertionError) as err_info:
        registry.register("some-key", "some-value")
    assert "Registry already contains some-key set to some-value." in str(err_info)


def test_fails_missing_key():
    registry = InstanceRegistry[str]()
    registry.register("some-key", "some-value")

    with pytest.raises(KeyError) as err_info:
        registry.get("another-key")
    assert "No registration for another-key. Known keys: ['some-key']" in str(err_info)


def test_lists_all_items():
    registry = InstanceRegistry[str]()
    registry.register("k1", "v1")
    registry.register("k2", "v2")
    registry.register("k3", "v3")
    assert registry.items() == [("k1", "v1"), ("k2", "v2"), ("k3", "v3")]
