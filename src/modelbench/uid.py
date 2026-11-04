import re
from enum import Enum
from typing import Any

import casefy

_HYPHENS = re.compile(r"-+")
_SELF_PREFIX = "self."


def _clean_string(value: Any) -> str:
    if isinstance(value, Enum):
        value = str(value.value).lower()
    s = _HYPHENS.sub("_", str(value))
    if s.lower() != s:
        return casefy.snakecase(s)
    return s


class HasUid:
    """
    A mixin class that gives an object an AISafety UID.

    Add it to your object's parent class list and then add a _uid_definition
    class variable that specifies your UID.

        class MySimpleObject(ABC, HasUid):
            _uid_definition = {"name": "simple", "version": "0.5"}

    That will result in a uid of "simple-0.5".

    Your UID values can include literals, properties, function references, or
    class references, all of which will get rendered automatically. Due to the
    specifics of python, you can't refer to a function or object before it
    exists, so make sure the UID definition is after the reference. For example:

        class MyDynamicObject(ABC, HasUid):
            def name(self):
                return "bob"
            _uid_definition = {"name": name, "version": "0.5"}

    Then calling MyDynamicObject().uid will return "bob-0.5".

    If you'd like to refer to the class currently being defined, you'll need to
    use the special value "class": "self", like this:

        class ClassyObject(ABC, HasUid):
            _uid_definition = {"class": "self", "version": "0.5"}

    This object's UID would be "classy_object-0.5".

    To refer to a property, prefix it with self:

        class IceCream(HasUid):
            def __init__(self):
                self.flavor = "chocolate"

            _uid_definition = {"class": "self", "flavor": "self.flavor"}

    This object's UID would be "ice_cream-chocolate".
    """

    @property
    def uid_definition(self) -> dict[str, Any]:
        if not hasattr(self.__class__, "_uid_definition"):
            raise AttributeError("classes with HasUid must define _uid_definition")
        return self.__class__._uid_definition

    def _as_string(self, key: str, value: Any) -> str:
        if key == "class" and value == "self":
            return _clean_string(self.__class__.__name__)
        if isinstance(value, type):
            return _clean_string(value.__name__)
        if isinstance(value, classmethod):
            return _clean_string(value.__wrapped__(self.__class__))
        if callable(value):
            return _clean_string(value(self))
        if isinstance(value, str) and value.startswith(_SELF_PREFIX):
            attr_name = value.removeprefix(_SELF_PREFIX)
            try:
                attr_value = getattr(self, attr_name)
            except AttributeError as e:
                raise AttributeError(
                    f"{self.__class__.__name__} uid key {key!r} refers to missing attribute {attr_name!r}"
                ) from e
            return _clean_string(attr_value)
        return _clean_string(value)

    @property
    def uid(self) -> str:
        return "-".join(self.uid_dict.values())

    @property
    def uid_dict(self) -> dict[str, str]:
        return {k: self._as_string(k, v) for k, v in self.uid_definition.items()}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.uid})"
