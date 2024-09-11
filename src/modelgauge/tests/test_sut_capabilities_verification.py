import pytest
from modelgauge.base_test import BaseTest
from modelgauge.sut import SUT
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_capabilities_verification import (
    MissingSUTCapabilities,
    assert_sut_capabilities,
    get_capable_suts,
    sut_is_capable,
)
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.test_decorator import modelgauge_test


@modelgauge_test(requires_sut_capabilities=[])
class NoReqsTest(BaseTest):
    pass


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class HasReqsTest(BaseTest):
    pass


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class HasMultipleReqsTest(BaseTest):
    pass


@modelgauge_sut(capabilities=[])
class NoReqsSUT(SUT):
    pass


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class HasReqsSUT(SUT):
    pass


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
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
