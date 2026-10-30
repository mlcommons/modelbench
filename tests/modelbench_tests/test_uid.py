from enum import Enum

import pytest

from modelbench.uid import HasUid


class HasStaticUid(HasUid, object):
    _uid_definition = {"name": "static", "version": "1.1"}


class HasPropertyInUid(HasUid, object):
    _uid_definition = {"name": "self.name"}

    def __init__(self, name):
        self.name = name


class HasInstanceMethodInUid(HasUid, object):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def name(self):
        return self._name

    _uid_definition = {"name": name}


class HasClassMethodInUid(HasUid, object):
    @classmethod
    def name(cls):
        return "a_class_specific_name"

    _uid_definition = {"name": name}


class HasOwnClassInUid(HasUid, object):
    _uid_definition = {"class": "self", "version": "1.2"}


def test_mixin_static():
    assert HasStaticUid().uid == "static-1.1"


def test_mixin_property():
    assert HasPropertyInUid("fnord").uid == "fnord"


def test_mixin_instance_method():
    assert HasInstanceMethodInUid("fnord").uid == "fnord"


def test_mixin_class_method():
    # class methods behave differently than normal methods
    assert HasClassMethodInUid().uid == "a_class_specific_name"


def test_mixin_class():
    assert HasOwnClassInUid().uid == "has_own_class_in_uid-1.2"


def test_mixin_case():
    assert HasInstanceMethodInUid("lower").uid == "lower"
    assert HasInstanceMethodInUid("lower_with_underscore").uid == "lower_with_underscore"
    assert HasInstanceMethodInUid("lower-with-dash").uid == "lower_with_dash"
    assert HasInstanceMethodInUid("UPPER").uid == "upper"
    assert HasInstanceMethodInUid("MixedCase").uid == "mixed_case"


class HasNumericLiteralInUid(HasUid, object):
    _uid_definition = {"name": "numeric", "version": 1}


class HasMissingPropertyInUid(HasUid, object):
    _uid_definition = {"name": "self.missing"}


class IntEnum(Enum):
    ONE = 1


def test_mixin_numeric_literal():
    assert HasNumericLiteralInUid().uid == "numeric-1"


def test_mixin_non_string_property():
    assert HasPropertyInUid(42).uid == "42"


def test_mixin_non_string_enum():
    assert HasPropertyInUid(IntEnum.ONE).uid == "1"


def test_mixin_missing_property():
    with pytest.raises(AttributeError, match="uid key 'name' refers to missing attribute 'missing'"):
        _ = HasMissingPropertyInUid().uid
