import pytest
from newhelm.record_init import (
    InitializationRecord,
    get_initialization_record,
    record_init,
)


class SomeClass:
    @record_init
    def __init__(self, x, y, z):
        self.total = x + y + z


class ClassWithDefaults:
    @record_init
    def __init__(self, a=None):
        if a is None:
            self.a = "the-default"
        else:
            self.a = a


class NoDecorator:
    def __init__(self, a):
        self.a = a


class ParentWithInit:
    @record_init
    def __init__(self, one):
        self.one = one


class ChildWithInit(ParentWithInit):
    @record_init
    def __init__(self, one, two):
        super().__init__(one)
        self.two = two


class ChildNoInit(ParentWithInit):
    pass


def test_record_init_all_positional():
    obj = SomeClass(1, 2, 3)
    assert obj.total == 6
    assert obj._initialization_record == InitializationRecord(
        module="test_record_init", qual_name="SomeClass", args=[1, 2, 3], kwargs={}
    )

    returned = obj._initialization_record.recreate_object()
    assert returned.total == 6


def test_record_init_all_kwarg():
    obj = SomeClass(x=1, y=2, z=3)
    assert obj.total == 6
    assert obj._initialization_record == InitializationRecord(
        module="test_record_init",
        qual_name="SomeClass",
        args=[],
        kwargs={"x": 1, "y": 2, "z": 3},
    )

    returned = obj._initialization_record.recreate_object()
    assert returned.total == 6
    assert obj._initialization_record == returned._initialization_record


def test_record_init_mix_positional_and_kwarg():
    obj = SomeClass(1, z=3, y=2)
    assert obj.total == 6
    assert obj._initialization_record == InitializationRecord(
        module="test_record_init",
        qual_name="SomeClass",
        args=[1],
        kwargs={"y": 2, "z": 3},
    )
    returned = obj._initialization_record.recreate_object()
    assert returned.total == 6
    assert obj._initialization_record == returned._initialization_record


def test_record_init_defaults():
    obj = ClassWithDefaults()
    assert obj.a == "the-default"
    assert obj._initialization_record == InitializationRecord(
        # Note the default isn't recorded
        module="test_record_init",
        qual_name="ClassWithDefaults",
        args=[],
        kwargs={},
    )
    returned = obj._initialization_record.recreate_object()
    assert returned.a == "the-default"
    assert obj._initialization_record == returned._initialization_record


def test_record_init_defaults_overwritten():
    obj = ClassWithDefaults("foo")
    assert obj.a == "foo"
    assert obj._initialization_record == InitializationRecord(
        # Note the default isn't recorded
        module="test_record_init",
        qual_name="ClassWithDefaults",
        args=["foo"],
        kwargs={},
    )
    returned = obj._initialization_record.recreate_object()
    assert returned.a == "foo"
    assert obj._initialization_record == returned._initialization_record


def test_parent_and_child_recorded_init():
    obj = ChildWithInit(1, 2)
    assert obj._initialization_record == InitializationRecord(
        module="test_record_init", qual_name="ChildWithInit", args=[1, 2], kwargs={}
    )
    returned = obj._initialization_record.recreate_object()
    assert returned.one == obj.one
    assert returned.two == obj.two
    assert obj._initialization_record == returned._initialization_record


def test_child_no_recorded_init():
    obj = ChildNoInit(1)
    assert obj._initialization_record == InitializationRecord(
        module="test_record_init", qual_name="ChildNoInit", args=[1], kwargs={}
    )
    returned = obj._initialization_record.recreate_object()
    assert returned.one == obj.one
    assert obj._initialization_record == returned._initialization_record


def test_get_record():
    obj = SomeClass(1, 2, 3)
    assert get_initialization_record(obj) == InitializationRecord(
        module="test_record_init", qual_name="SomeClass", args=[1, 2, 3], kwargs={}
    )


def test_get_record_no_decorator():
    obj = NoDecorator(1)
    with pytest.raises(AssertionError) as err_info:
        get_initialization_record(obj)
    error_text = str(err_info.value)
    assert (
        error_text
        == "Class NoDecorator in module test_record_init needs to add `@record_init` to its `__init__` function to enable system reproducibility."
    )
