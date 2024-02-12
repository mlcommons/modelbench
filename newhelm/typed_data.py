from typing import Any, Dict, Type, TypeVar
from typing_extensions import Self
from pydantic import BaseModel


_BaseModelType = TypeVar("_BaseModelType", bound=BaseModel)


class TypedData(BaseModel):
    """This is a generic container that allows Pydantic to do polymorphic serialization.

    This is useful in situations where you have an unknown set of classes that could be
    used in a particular field.
    """

    type: str
    data: Dict[str, Any]

    @classmethod
    def from_instance(cls, obj: BaseModel) -> Self:
        """Convert the object into a TypedData instance."""
        return cls(type=TypedData._get_type(obj.__class__), data=obj.model_dump())

    def to_instance(self, instance_cls: Type[_BaseModelType]) -> _BaseModelType:
        """Convert this data back into its original type."""
        instance_cls_type = TypedData._get_type(instance_cls)
        assert (
            instance_cls_type == self.type
        ), f"Cannot convert {self.type} to {instance_cls_type}."
        return instance_cls.model_validate(self.data)

    @staticmethod
    def _get_type(instance_cls: Type[BaseModel]) -> str:
        return f"{instance_cls.__module__}.{instance_cls.__qualname__}"
