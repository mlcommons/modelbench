from newhelm.general import get_class, get_unique_id


class NestedClass:
    class Layer1:
        class Layer2:
            value: str

        layer_2: Layer2

    layer_1: Layer1


def test_unique_id():
    value = get_unique_id()
    assert isinstance(value, str)
    assert value != ""


def test_get_class():
    assert get_class("test_general", "NestedClass") == NestedClass


def test_get_class_nested():
    assert (
        get_class("test_general", "NestedClass.Layer1.Layer2")
        == NestedClass.Layer1.Layer2
    )
