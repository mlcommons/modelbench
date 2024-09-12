import datetime
from modelgauge.general import (
    current_local_datetime,
    get_class,
    normalize_filename,
)
from pydantic import AwareDatetime, BaseModel, Field


class NestedClass:
    class Layer1:
        class Layer2:
            value: str

        layer_2: Layer2

    layer_1: Layer1


def test_get_class():
    assert get_class("tests.test_general", "NestedClass") == NestedClass


def test_get_class_nested():
    assert (
        get_class("tests.test_general", "NestedClass.Layer1.Layer2")
        == NestedClass.Layer1.Layer2
    )


class PydanticWithDateTime(BaseModel):
    timestamp: AwareDatetime = Field(default_factory=current_local_datetime)


def test_datetime_round_trip():
    original = PydanticWithDateTime()
    as_json = original.model_dump_json()
    returned = PydanticWithDateTime.model_validate_json(as_json, strict=True)
    assert original == returned


def test_datetime_serialized():
    desired = datetime.datetime(
        2017,
        8,
        21,
        11,
        47,
        0,
        123456,
        tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=61200), "MST"),
    )
    original = PydanticWithDateTime(timestamp=desired)
    assert original.model_dump_json() == (
        """{"timestamp":"2017-08-21T11:47:00.123456-07:00"}"""
    )


def test_normalize_filename():
    assert normalize_filename("a/b/c.ext") == "a_b_c.ext"
    assert normalize_filename("a-b-c.ext") == "a-b-c.ext"
