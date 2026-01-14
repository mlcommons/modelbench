import pytest
from modelgauge.sut import SUT
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_capabilities_verification import (
    MissingMultipleSUTsCapabilities,
    MissingSUTCapabilities,
    assert_multiple_suts_capabilities,
    assert_sut_capabilities,
    get_capable_suts,
    sut_is_capable,
)
from modelgauge.sut_decorator import modelgauge_sut


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
    assert_sut_capabilities(sut=NoReqsSUT("sut-uid"), required_capabilities=[])


def test_assert_sut_capabilities_extras():
    assert_sut_capabilities(sut=HasReqsSUT("sut-uid"), required_capabilities=[])


def test_assert_sut_capabilities_both():
    assert_sut_capabilities(sut=HasReqsSUT("sut-uid"), required_capabilities=[AcceptsTextPrompt])


def test_assert_sut_capabilities_missing():
    with pytest.raises(MissingSUTCapabilities) as err_info:
        assert_sut_capabilities(sut=NoReqsSUT("sut-uid"), required_capabilities=[AcceptsTextPrompt])
    assert str(err_info.value) == ("sut-uid is missing the following required capabilities: ['AcceptsTextPrompt'].")


def test_assert_sut_capabilities_multiple_missing():
    with pytest.raises(MissingSUTCapabilities) as err_info:
        assert_sut_capabilities(sut=NoReqsSUT("sut-uid"), required_capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
    assert str(err_info.value) == (
        "sut-uid is missing the following required capabilities: ['AcceptsTextPrompt', 'AcceptsChatPrompt']."
    )


def test_assert_sut_capabilities_only_missing():
    with pytest.raises(MissingSUTCapabilities) as err_info:
        assert_sut_capabilities(sut=HasReqsSUT("sut-uid"), required_capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
    assert str(err_info.value) == ("sut-uid is missing the following required capabilities: ['AcceptsChatPrompt'].")


def test_assert_multiple_suts_capabilities_neither_missing():
    assert_multiple_suts_capabilities(suts=[NoReqsSUT("sut-1"), NoReqsSUT("sut-2")], required_capabilities=[])


def test_assert_multiple_suts_capabilities_one_missing():
    with pytest.raises(MissingSUTCapabilities) as err_info:
        assert_multiple_suts_capabilities(
            suts=[HasReqsSUT("sut-1"), NoReqsSUT("sut-2")], required_capabilities=[AcceptsTextPrompt]
        )
    assert str(err_info.value) == ("sut-2 is missing the following required capabilities: ['AcceptsTextPrompt'].")


def test_assert_multiple_suts_capabilities_both_missing():
    with pytest.raises(MissingMultipleSUTsCapabilities) as err_info:
        assert_multiple_suts_capabilities(
            suts=[NoReqsSUT("sut-1"), NoReqsSUT("sut-2")], required_capabilities=[AcceptsTextPrompt]
        )
    assert str(err_info.value) == (
        "Multiple SUTs are missing required capabilities:\nsut-1 is missing the following required capabilities: ['AcceptsTextPrompt'].\nsut-2 is missing the following required capabilities: ['AcceptsTextPrompt']."
    )


def test_sut_is_capable():
    assert sut_is_capable(sut=NoReqsSUT("some-sut"), required_capabilities=[]) == True
    assert sut_is_capable(sut=NoReqsSUT("some-sut"), required_capabilities=[AcceptsTextPrompt]) == False


def test_get_capable_suts():
    none = NoReqsSUT("no-reqs")
    some = HasReqsSUT("has-reqs")
    multiple = HasMultipleReqsSUT("multiple-reqs")
    result = get_capable_suts([none, some, multiple], [AcceptsTextPrompt])
    assert result == [some, multiple]
