import pytest
from newhelm.base_test import BaseTest
from newhelm.record_init import InitializationRecord
from newhelm.sut_capabilities import AcceptsTextPrompt
from newhelm.test_decorator import assert_is_test, newhelm_test


@newhelm_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SomeTest(BaseTest):
    def __init__(self, uid, arg1):
        self.uid = uid
        self.arg1 = arg1


def test_basic():
    result = SomeTest(1234, 2)
    assert result.uid == 1234
    assert result.arg1 == 2
    assert result.requires_sut_capabilities == [AcceptsTextPrompt]
    assert result._newhelm_test
    assert_is_test(result)


class NoDecorator(BaseTest):
    def __init__(self, uid, arg1):
        self.uid = uid
        self.arg1 = arg1


def test_no_decorator():
    result = NoDecorator(1234, 2)
    with pytest.raises(AssertionError) as err_info:
        assert_is_test(result)
    assert str(err_info.value) == "NoDecorator should be decorated with @newhelm_test."


@newhelm_test(requires_sut_capabilities=[])
class ChildTestCallsSuper(SomeTest):
    def __init__(self, uid, arg1, arg2):
        super().__init__(uid, arg1)
        self.arg2 = arg2


def test_child_calls_super():
    result = ChildTestCallsSuper(1234, 2, 3)
    assert result.uid == 1234
    assert result._newhelm_test
    assert result.initialization_record == InitializationRecord(
        module="tests.test_test_decorator",
        class_name="ChildTestCallsSuper",
        args=[1234, 2, 3],
        kwargs={},
    )


@newhelm_test(requires_sut_capabilities=[])
class ChildTestNoSuper(SomeTest):
    def __init__(self, uid, arg1, arg2):
        self.uid = uid
        self.arg1 = arg1
        self.arg2 = arg2


def test_child_no_super():
    result = ChildTestNoSuper(1234, 2, 3)
    assert result.uid == 1234
    assert result._newhelm_test
    assert result.initialization_record == InitializationRecord(
        module="tests.test_test_decorator",
        class_name="ChildTestNoSuper",
        args=[1234, 2, 3],
        kwargs={},
    )


@newhelm_test(requires_sut_capabilities=[])
class ChildTestNoInit(SomeTest):
    pass


def test_child_init():
    result = ChildTestNoInit(1234, 2)
    assert result.uid == 1234
    assert result._newhelm_test
    assert result.initialization_record == InitializationRecord(
        module="tests.test_test_decorator",
        class_name="ChildTestNoInit",
        args=[1234, 2],
        kwargs={},
    )


def test_bad_signature():
    with pytest.raises(AssertionError) as err_info:
        # Exception happens without even constructing an instance.
        @newhelm_test(requires_sut_capabilities=[])
        class ChildBadSignature(SomeTest):
            def __init__(self, arg1, uid):
                self.uid = uid
                self.arg1 = arg1

    assert "All Tests must have UID as the first parameter." in str(err_info.value)
