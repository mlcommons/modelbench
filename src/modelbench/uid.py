import re
from enum import Enum

import casefy


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
        class IceCream:
            def __init__(self):
                self.flavor="chocolate"
        _uid_definition = {"class": "self", "flavor": "self.flavor"}

    This object's UID would be "ice_cream-chocolate"
    """

    @property
    def uid_definition(self) -> dict:
        if not hasattr(self.__class__, "_uid_definition"):
            raise AttributeError("classes with HasUid must define _uid_definition")
        return self.__class__._uid_definition

    def _as_string(self, k, o):
        def clean_string(s):
            if isinstance(s, Enum):
                s = s.value.lower()
            s = re.sub("[-]+", "_", s)
            if s.lower() != s:
                return casefy.snakecase(s)
            else:
                return s

        if k == "class" and o == "self":
            return clean_string(self.__class__.__name__)
        if isinstance(o, type):
            return clean_string(o.__name__)
        if isinstance(o, classmethod):
            return clean_string(str(o.__wrapped__(self.__class__)))
        if callable(o):
            return clean_string(str(o(self)))
        if o.startswith("self."):
            return clean_string(self.__getattribute__(o[5:]))
        return clean_string(str(o))

    @property
    def uid(self) -> str:
        return "-".join(self._as_string(k, v) for k, v in self.uid_definition.items())

    @property
    def uid_dict(self) -> dict:
        return {k: self._as_string(k, v) for k, v in self.uid_definition.items()}

    def __str__(self):
        return f"{self.__class__.__name__}({self.uid})"
