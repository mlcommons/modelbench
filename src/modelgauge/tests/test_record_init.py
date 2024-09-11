from modelgauge.record_init import InitializationRecord, add_initialization_record
from modelgauge.secret_values import SerializedSecret
from tests.fake_secrets import FakeRequiredSecret


def record(cls):
    if hasattr(cls.__init__, "wrapped"):
        return cls

    def _wrap_init(init):
        def inner(self, *args, **kwargs):
            init(self, *args, **kwargs)
            add_initialization_record(self, *args, **kwargs)

        return inner

    cls.__init__ = _wrap_init(cls.__init__)
    cls.__init__.wrapped = True
    return cls


@record
class SomeClass:
    def __init__(self, x, y, z):
        self.total = x + y + z


@record
class ClassWithDefaults:
    def __init__(self, a=None):
        if a is None:
            self.a = "the-default"
        else:
            self.a = a


class NoDecorator:
    def __init__(self, a):
        self.a = a


@record
class ParentWithInit:
    def __init__(self, one):
        self.one = one


@record
class ChildWithInit(ParentWithInit):
    def __init__(self, one, two):
        super().__init__(one)
        self.two = two


@record
class ChildNoInit(ParentWithInit):
    pass


def test_record_init_all_positional():
    obj = SomeClass(1, 2, 3)
    assert obj.total == 6
    assert obj.initialization_record == InitializationRecord(
        module="tests.test_record_init",
        class_name="SomeClass",
        args=[1, 2, 3],
        kwargs={},
    )

    returned = obj.initialization_record.recreate_object()
    assert returned.total == 6


def test_record_init_all_kwarg():
    obj = SomeClass(x=1, y=2, z=3)
    assert obj.total == 6
    assert obj.initialization_record == InitializationRecord(
        module="tests.test_record_init",
        class_name="SomeClass",
        args=[],
        kwargs={"x": 1, "y": 2, "z": 3},
    )

    returned = obj.initialization_record.recreate_object()
    assert returned.total == 6
    assert obj.initialization_record == returned.initialization_record


def test_record_init_mix_positional_and_kwarg():
    obj = SomeClass(1, z=3, y=2)
    assert obj.total == 6
    assert obj.initialization_record == InitializationRecord(
        module="tests.test_record_init",
        class_name="SomeClass",
        args=[1],
        kwargs={"y": 2, "z": 3},
    )
    returned = obj.initialization_record.recreate_object()
    assert returned.total == 6
    assert obj.initialization_record == returned.initialization_record


def test_record_init_defaults():
    obj = ClassWithDefaults()
    assert obj.a == "the-default"
    assert obj.initialization_record == InitializationRecord(
        # Note the default isn't recorded
        module="tests.test_record_init",
        class_name="ClassWithDefaults",
        args=[],
        kwargs={},
    )
    returned = obj.initialization_record.recreate_object()
    assert returned.a == "the-default"
    assert obj.initialization_record == returned.initialization_record


def test_record_init_defaults_overwritten():
    obj = ClassWithDefaults("foo")
    assert obj.a == "foo"
    assert obj.initialization_record == InitializationRecord(
        # Note the default isn't recorded
        module="tests.test_record_init",
        class_name="ClassWithDefaults",
        args=["foo"],
        kwargs={},
    )
    returned = obj.initialization_record.recreate_object()
    assert returned.a == "foo"
    assert obj.initialization_record == returned.initialization_record


def test_parent_and_child_recorded_init():
    obj = ChildWithInit(1, 2)
    assert obj.initialization_record == InitializationRecord(
        module="tests.test_record_init",
        class_name="ChildWithInit",
        args=[1, 2],
        kwargs={},
    )
    returned = obj.initialization_record.recreate_object()
    assert returned.one == obj.one
    assert returned.two == obj.two
    assert obj.initialization_record == returned.initialization_record


def test_child_no_recorded_init():
    obj = ChildNoInit(1)
    assert obj.initialization_record == InitializationRecord(
        module="tests.test_record_init", class_name="ChildNoInit", args=[1], kwargs={}
    )
    returned = obj.initialization_record.recreate_object()
    assert returned.one == obj.one
    assert obj.initialization_record == returned.initialization_record


@record
class UsesSecrets:
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.secret = arg2.value


def test_uses_secrets_arg():
    obj = UsesSecrets(1, FakeRequiredSecret("some-value"))
    assert obj.initialization_record == InitializationRecord(
        module="tests.test_record_init",
        class_name="UsesSecrets",
        args=[
            1,
            SerializedSecret(
                module="tests.fake_secrets", class_name="FakeRequiredSecret"
            ),
        ],
        kwargs={},
    )

    new_secrets = {"some-scope": {"some-key": "another-value"}}
    returned = obj.initialization_record.recreate_object(secrets=new_secrets)
    assert returned.secret == "another-value"


def test_uses_secrets_kwarg():
    obj = UsesSecrets(arg1=1, arg2=FakeRequiredSecret("some-value"))
    assert obj.initialization_record == InitializationRecord(
        module="tests.test_record_init",
        class_name="UsesSecrets",
        args=[],
        kwargs={
            "arg1": 1,
            "arg2": SerializedSecret(
                module="tests.fake_secrets", class_name="FakeRequiredSecret"
            ),
        },
    )

    new_secrets = {"some-scope": {"some-key": "another-value"}}
    returned = obj.initialization_record.recreate_object(secrets=new_secrets)
    assert returned.secret == "another-value"
