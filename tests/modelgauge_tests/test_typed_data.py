import pytest
from modelgauge.typed_data import TypedData, is_typeable
from pydantic import BaseModel
from typing import List


class LeafClass1(BaseModel):
    value: str


class LeafClass2(BaseModel):
    """Identical to the previous class to demonstrate serialization stability."""

    value: str


class DeepLeaf(BaseModel):
    """Demonstrates a complex object to store in TypedData"""

    leaf_1: LeafClass1
    leaf_2: LeafClass2


class PolymorphicList(BaseModel):
    """Layer that wants to hold any set of leaves."""

    elements: List[TypedData]


class TopLevel(BaseModel):
    """Wrapper around the polymorphic layer."""

    poly: PolymorphicList


class NestedClasses(BaseModel):
    class Layer1(BaseModel):
        class Layer2(BaseModel):
            value: str

        layer_2: Layer2

    layer_1: Layer1


def test_shallow_round_trip():
    original = LeafClass1(value="some-value")
    typed_data = TypedData.from_instance(original)
    as_json = typed_data.model_dump_json()
    assert (
        as_json
        == """{"module":"tests.test_typed_data","class_name":"LeafClass1","data":{"value":"some-value"}}"""
    )
    returned = TypedData.model_validate_json(as_json)
    assert typed_data == returned
    returned_type = returned.to_instance(LeafClass1)
    assert original == returned_type


def test_polymorphic_round_trip():
    original = TopLevel(
        poly=PolymorphicList(
            elements=[
                TypedData.from_instance(LeafClass1(value="l1")),
                TypedData.from_instance(LeafClass2(value="l2")),
                TypedData.from_instance(
                    DeepLeaf(
                        leaf_1=LeafClass1(value="deep1"),
                        leaf_2=LeafClass2(value="deep2"),
                    )
                ),
            ]
        )
    )
    typed_data = TypedData.from_instance(original)
    as_json = typed_data.model_dump_json(indent=2)
    assert (
        as_json
        == """\
{
  "module": "tests.test_typed_data",
  "class_name": "TopLevel",
  "data": {
    "poly": {
      "elements": [
        {
          "module": "tests.test_typed_data",
          "class_name": "LeafClass1",
          "data": {
            "value": "l1"
          }
        },
        {
          "module": "tests.test_typed_data",
          "class_name": "LeafClass2",
          "data": {
            "value": "l2"
          }
        },
        {
          "module": "tests.test_typed_data",
          "class_name": "DeepLeaf",
          "data": {
            "leaf_1": {
              "value": "deep1"
            },
            "leaf_2": {
              "value": "deep2"
            }
          }
        }
      ]
    }
  }
}"""
    )
    returned = TypedData.model_validate_json(as_json)
    assert typed_data == returned
    returned_type = returned.to_instance(TopLevel)
    assert original == returned_type


def test_multiple_polymorphic_layers():
    original = TopLevel(
        poly=PolymorphicList(
            elements=[
                TypedData.from_instance(
                    PolymorphicList(
                        elements=[
                            TypedData.from_instance(LeafClass1(value="l1")),
                        ]
                    )
                ),
                TypedData.from_instance(LeafClass2(value="l2")),
            ]
        )
    )
    typed_data = TypedData.from_instance(original)
    as_json = typed_data.model_dump_json(indent=2)
    assert (
        as_json
        == """\
{
  "module": "tests.test_typed_data",
  "class_name": "TopLevel",
  "data": {
    "poly": {
      "elements": [
        {
          "module": "tests.test_typed_data",
          "class_name": "PolymorphicList",
          "data": {
            "elements": [
              {
                "module": "tests.test_typed_data",
                "class_name": "LeafClass1",
                "data": {
                  "value": "l1"
                }
              }
            ]
          }
        },
        {
          "module": "tests.test_typed_data",
          "class_name": "LeafClass2",
          "data": {
            "value": "l2"
          }
        }
      ]
    }
  }
}"""
    )
    returned = TypedData.model_validate_json(as_json)
    assert typed_data == returned
    returned_type = returned.to_instance(TopLevel)
    assert original == returned_type
    # Pull out the nested parts.
    middle_layer = returned_type.poly.elements[0].to_instance(PolymorphicList)
    assert middle_layer.elements[0].to_instance(LeafClass1).value == "l1"


def test_wrong_type_deserialize():
    typed_data = TypedData.from_instance(LeafClass1(value="l1"))
    with pytest.raises(AssertionError) as err_info:
        typed_data.to_instance(LeafClass2)
    err_text = str(err_info.value)
    assert (
        err_text
        == "Cannot convert tests.test_typed_data.LeafClass1 to tests.test_typed_data.LeafClass2."
    )


def test_nested_classes():
    original = NestedClasses(
        layer_1=NestedClasses.Layer1(
            layer_2=NestedClasses.Layer1.Layer2(value="some-value")
        )
    )
    typed_data = TypedData.from_instance(original)
    as_json = typed_data.model_dump_json(indent=2)
    assert (
        as_json
        == """\
{
  "module": "tests.test_typed_data",
  "class_name": "NestedClasses",
  "data": {
    "layer_1": {
      "layer_2": {
        "value": "some-value"
      }
    }
  }
}"""
    )

    returned = TypedData.model_validate_json(as_json)
    assert typed_data == returned
    returned_type = returned.to_instance(NestedClasses)
    assert original == returned_type


def test_to_instance_no_argument():
    original = LeafClass1(value="some-value")
    typed_data = TypedData.from_instance(original)
    as_json = typed_data.model_dump_json()
    assert (
        as_json
        == """{"module":"tests.test_typed_data","class_name":"LeafClass1","data":{"value":"some-value"}}"""
    )
    returned = TypedData.model_validate_json(as_json)
    assert typed_data == returned
    returned_type = returned.to_instance()  # Defaults to None
    assert original == returned_type


def test_to_instance_no_argument_nested_type():
    original = NestedClasses.Layer1.Layer2(value="some-value")
    typed_data = TypedData.from_instance(original)
    as_json = typed_data.model_dump_json()
    assert (
        as_json
        == """{"module":"tests.test_typed_data","class_name":"NestedClasses.Layer1.Layer2","data":{"value":"some-value"}}"""
    )
    returned = TypedData.model_validate_json(as_json)
    assert typed_data == returned
    returned_type = returned.to_instance()  # Defaults to None
    assert original == returned_type


def test_dict_round_trip():
    original = {"a": 1, "b": 2}
    typed_data = TypedData.from_instance(original)
    as_json = typed_data.model_dump_json()
    assert (
        as_json == """{"module":"builtins","class_name":"dict","data":{"a":1,"b":2}}"""
    )
    returned = TypedData.model_validate_json(as_json)
    assert typed_data == returned
    returned_type = returned.to_instance()
    assert original == returned_type


class InheritedBaseModel(LeafClass1):
    pass


def test_is_typeable():
    assert is_typeable(LeafClass1(value="1"))
    assert is_typeable(InheritedBaseModel(value="1"))
    assert is_typeable({"foo": 1234})
    assert not is_typeable(1234)
    assert not is_typeable({1234: "7"})
