import pytest
from newhelm.base_test import BaseTest, TestMetadata
from newhelm.sut import SUT
from newhelm.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from newhelm.sut_capabilities_verification import (
    MissingSUTCapabilities,
    assert_sut_capabilities,
    get_capable_suts,
    sut_is_capable,
)
from newhelm.sut_decorator import newhelm_sut
from newhelm.test_decorator import newhelm_test


@newhelm_test(requires_sut_capabilities=[])
class NoReqsTest(BaseTest):
    def get_metadata(self) -> TestMetadata:
        raise NotImplementedError()


@newhelm_test(requires_sut_capabilities=[AcceptsTextPrompt])
class HasReqsTest(BaseTest):
    def get_metadata(self) -> TestMetadata:
        raise NotImplementedError()


@newhelm_test(requires_sut_capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class HasMultipleReqsTest(BaseTest):
    def get_metadata(self) -> TestMetadata:
        raise NotImplementedError()


@newhelm_sut(capabilities=[])
class NoReqsSUT(SUT):
    pass


@newhelm_sut(capabilities=[AcceptsTextPrompt])
class HasReqsSUT(SUT):
    pass


@newhelm_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class HasMultipleReqsSUT(SUT):
    pass


def test_assert_sut_capabilities_neither():
    assert_sut_capabilities(sut=NoReqsSUT("sut-uid"), test=NoReqsTest("test-uid"))


def test_assert_sut_capabilities_extras():
    assert_sut_capabilities(sut=HasReqsSUT("sut-uid"), test=NoReqsTest("test-uid"))


def test_assert_sut_capabilities_both():
    assert_sut_capabilities(sut=HasReqsSUT("sut-uid"), test=HasReqsTest("test-uid"))


def test_assert_sut_capabilities_missing():
    with pytest.raises(MissingSUTCapabilities) as err_info:
        assert_sut_capabilities(sut=NoReqsSUT("sut-uid"), test=HasReqsTest("test-uid"))
    assert str(err_info.value) == (
        "Test test-uid cannot run on sut-uid because it requires "
        "the following capabilities: ['AcceptsTextPrompt']."
    )


def test_assert_sut_capabilities_multiple_missing():
    with pytest.raises(MissingSUTCapabilities) as err_info:
        assert_sut_capabilities(
            sut=NoReqsSUT("sut-uid"), test=HasMultipleReqsTest("test-uid")
        )
    assert str(err_info.value) == (
        "Test test-uid cannot run on sut-uid because it requires "
        "the following capabilities: ['AcceptsTextPrompt', 'AcceptsChatPrompt']."
    )


def test_assert_sut_capabilities_only_missing():
    with pytest.raises(MissingSUTCapabilities) as err_info:
        assert_sut_capabilities(
            sut=HasReqsSUT("sut-uid"), test=HasMultipleReqsTest("test-uid")
        )
    assert str(err_info.value) == (
        "Test test-uid cannot run on sut-uid because it requires "
        "the following capabilities: ['AcceptsChatPrompt']."
    )


def test_sut_is_capable():
    assert (
        sut_is_capable(sut=NoReqsSUT("some-sut"), test=NoReqsTest("some-test")) == True
    )
    assert (
        sut_is_capable(sut=NoReqsSUT("some-sut"), test=HasReqsTest("some-test"))
        == False
    )


def test_get_capable_suts():
    none = NoReqsSUT("no-reqs")
    some = HasReqsSUT("has-reqs")
    multiple = HasMultipleReqsSUT("multiple-reqs")
    result = get_capable_suts(HasReqsTest("some-test"), [none, some, multiple])
    assert result == [some, multiple]
