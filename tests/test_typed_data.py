from typing import List
from pydantic import BaseModel
import pytest

from newhelm.typed_data import TypedData


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


def test_shallow_round_trip():
    original = LeafClass1(value="some-value")
    typed_data = TypedData.from_instance(original)
    as_json = typed_data.model_dump_json()
    assert (
        as_json
        == """{"type":"test_typed_data.LeafClass1","data":{"value":"some-value"}}"""
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
  "type": "test_typed_data.TopLevel",
  "data": {
    "poly": {
      "elements": [
        {
          "type": "test_typed_data.LeafClass1",
          "data": {
            "value": "l1"
          }
        },
        {
          "type": "test_typed_data.LeafClass2",
          "data": {
            "value": "l2"
          }
        },
        {
          "type": "test_typed_data.DeepLeaf",
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
  "type": "test_typed_data.TopLevel",
  "data": {
    "poly": {
      "elements": [
        {
          "type": "test_typed_data.PolymorphicList",
          "data": {
            "elements": [
              {
                "type": "test_typed_data.LeafClass1",
                "data": {
                  "value": "l1"
                }
              }
            ]
          }
        },
        {
          "type": "test_typed_data.LeafClass2",
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
        == "Cannot convert test_typed_data.LeafClass1 to test_typed_data.LeafClass2."
    )
