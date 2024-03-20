import pytest
from newhelm.record_init import InitializationRecord
from newhelm.sut import SUT
from newhelm.sut_decorator import newhelm_sut


@newhelm_sut()
class SomeSUT(SUT):
    def __init__(self, uid, arg1):
        self.uid = uid
        self.arg1 = arg1


def test_basic():
    result = SomeSUT(1234, 2)
    assert result.uid == 1234
    assert result.arg1 == 2
    assert result._newhelm_sut


@newhelm_sut()
class ChildSUTCallsSuper(SomeSUT):
    def __init__(self, uid, arg1, arg2):
        super().__init__(uid, arg1)
        self.arg2 = arg2


def test_child_calls_super():
    result = ChildSUTCallsSuper(1234, 2, 3)
    assert result.uid == 1234
    assert result._newhelm_sut
    assert result.initialization_record == InitializationRecord(
        module="tests.test_sut_decorator",
        class_name="ChildSUTCallsSuper",
        args=[1234, 2, 3],
        kwargs={},
    )


@newhelm_sut()
class ChildSUTNoSuper(SomeSUT):
    def __init__(self, uid, arg1, arg2):
        self.uid = uid
        self.arg1 = arg1
        self.arg2 = arg2


def test_child_no_super():
    result = ChildSUTNoSuper(1234, 2, 3)
    assert result.uid == 1234
    assert result._newhelm_sut
    assert result.initialization_record == InitializationRecord(
        module="tests.test_sut_decorator",
        class_name="ChildSUTNoSuper",
        args=[1234, 2, 3],
        kwargs={},
    )


@newhelm_sut()
class ChildSUTNoInit(SomeSUT):
    pass


def test_child_init():
    result = ChildSUTNoInit(1234, 2)
    assert result.uid == 1234
    assert result._newhelm_sut
    assert result.initialization_record == InitializationRecord(
        module="tests.test_sut_decorator",
        class_name="ChildSUTNoInit",
        args=[1234, 2],
        kwargs={},
    )


def test_bad_signature():
    with pytest.raises(AssertionError) as err_info:
        # Exception happens without even constructing an instance.
        @newhelm_sut()
        class ChildBadSignature(SomeSUT):
            def __init__(self, arg1, uid):
                self.uid = uid
                self.arg1 = arg1

    assert "All SUTs must have UID as the first parameter." in str(err_info.value)
