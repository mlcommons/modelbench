import pytest
from modelgauge.not_implemented import not_implemented
from modelgauge.prompt import ChatPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.sut import SUT, PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
    ProducesPerTokenLogProbabilities,
)
from modelgauge.sut_decorator import assert_is_sut, modelgauge_sut


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class SomeSUT(SUT):
    def __init__(self, uid, arg1):
        self.uid = uid
        self.arg1 = arg1


def test_basic():
    result = SomeSUT(1234, 2)
    assert result.uid == 1234
    assert result.arg1 == 2
    assert result.capabilities == [AcceptsTextPrompt]
    assert result._modelgauge_sut
    assert_is_sut(result)


class NoDecorator(SUT):
    def __init__(self, uid, arg1):
        self.uid = uid
        self.arg1 = arg1


def test_no_decorator():
    result = NoDecorator(1234, 2)
    with pytest.raises(AssertionError) as err_info:
        assert_is_sut(result)
    assert (
        str(err_info.value) == "NoDecorator should be decorated with @modelgauge_sut."
    )


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class ChildSUTCallsSuper(SomeSUT):
    def __init__(self, uid, arg1, arg2):
        super().__init__(uid, arg1)
        self.arg2 = arg2


def test_child_calls_super():
    result = ChildSUTCallsSuper(1234, 2, 3)
    assert result.uid == 1234
    assert result._modelgauge_sut
    assert result.initialization_record == InitializationRecord(
        module="tests.test_sut_decorator",
        class_name="ChildSUTCallsSuper",
        args=[1234, 2, 3],
        kwargs={},
    )


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class ChildSUTNoSuper(SomeSUT):
    def __init__(self, uid, arg1, arg2):
        self.uid = uid
        self.arg1 = arg1
        self.arg2 = arg2


def test_child_no_super():
    result = ChildSUTNoSuper(1234, 2, 3)
    assert result.uid == 1234
    assert result._modelgauge_sut
    assert result.initialization_record == InitializationRecord(
        module="tests.test_sut_decorator",
        class_name="ChildSUTNoSuper",
        args=[1234, 2, 3],
        kwargs={},
    )


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class ChildSUTNoInit(SomeSUT):
    pass


def test_child_init():
    result = ChildSUTNoInit(1234, 2)
    assert result.uid == 1234
    assert result._modelgauge_sut
    assert result.initialization_record == InitializationRecord(
        module="tests.test_sut_decorator",
        class_name="ChildSUTNoInit",
        args=[1234, 2],
        kwargs={},
    )


def test_bad_signature():
    with pytest.raises(AssertionError) as err_info:
        # Exception happens without even constructing an instance.
        @modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
        class ChildBadSignature(SomeSUT):
            def __init__(self, arg1, uid):
                self.uid = uid
                self.arg1 = arg1

    assert "All SUTs must have UID as the first parameter." in str(err_info.value)


class SomePromptResponseSUT(PromptResponseSUT):
    # Define abstract methods to make subclasses easier to make.
    def translate_text_prompt(self, prompt):
        pass

    def evaluate(self, request):
        pass

    def translate_response(self, request, response):
        pass


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class LogprobsNoCapabilitiesNotSet(SomePromptResponseSUT):
    def translate_response(self, request, response):
        return SUTResponse(completions=[SUTCompletion(text="some-text")])


def test_logprobs_no_capabilities_not_set():
    sut = LogprobsNoCapabilitiesNotSet("some-sut")
    # Mostly here to ensure no exceptions
    assert sut.translate_response(None, None).completions[0].text == "some-text"


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class LogprobsNoCapabilitiesAndSet(SomePromptResponseSUT):
    def translate_response(self, request, response):
        return SUTResponse(
            completions=[SUTCompletion(text="some-text", top_logprobs=[])]
        )


def test_logprobs_no_capabilities_and_set():
    sut = LogprobsNoCapabilitiesAndSet("some-sut")
    with pytest.raises(AssertionError) as err_info:
        sut.translate_response(None, None)
    assert (
        "LogprobsNoCapabilitiesAndSet does not list capability ProducesPerTokenLogProbabilities"
        in str(err_info.value)
    )


@modelgauge_sut(capabilities=[ProducesPerTokenLogProbabilities, AcceptsTextPrompt])
class LogprobsHasCapabilitiesNotSet(SomePromptResponseSUT):
    def translate_response(self, request, response):
        return SUTResponse(completions=[SUTCompletion(text="some-text")])


def test_logprobs_has_capabilities_not_set():
    sut = LogprobsHasCapabilitiesNotSet("some-sut")
    # This is allowed because SUTOption might not be set
    assert sut.translate_response(None, None).completions[0].text == "some-text"


@modelgauge_sut(capabilities=[ProducesPerTokenLogProbabilities, AcceptsTextPrompt])
class LogprobsHasCapabilitiesAndSet(SomePromptResponseSUT):
    def translate_response(self, request, response):
        return SUTResponse(
            completions=[SUTCompletion(text="some-text", top_logprobs=[])]
        )


def test_logprobs_has_capabilities_and_set():
    sut = LogprobsHasCapabilitiesAndSet("some-sut")
    assert sut.translate_response(None, None).completions[0].text == "some-text"


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class LogprobsInheritsSet(LogprobsHasCapabilitiesAndSet):
    pass


def test_logprobs_inherits_set():
    sut = LogprobsInheritsSet("some-sut")
    with pytest.raises(AssertionError) as err_info:
        sut.translate_response(None, None)
    assert (
        "LogprobsInheritsSet does not list capability ProducesPerTokenLogProbabilities"
        in str(err_info.value)
    )


def test_both_capabilities_both_implemented():
    @modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
    class BothCapabilitiesBothImplmented(SomePromptResponseSUT):
        def translate_text_prompt(self, prompt):
            pass

        def translate_chat_prompt(self, prompt):
            pass

    # Verify you can make an instance
    BothCapabilitiesBothImplmented("some-sut")


def test_chat_capabilities_not_implemented():
    with pytest.raises(AssertionError) as err_info:

        @modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
        class ChatCapabilitiesNotImplemented(SomePromptResponseSUT):
            def translate_text_prompt(self, prompt):
                pass

    assert str(err_info.value) == (
        "ChatCapabilitiesNotImplemented says it AcceptsChatPrompt, "
        "but it does not implement translate_chat_prompt."
    )


def test_chat_capabilities_not_implemented_override():
    with pytest.raises(AssertionError) as err_info:

        @modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
        class ChatCapabilitiesNotImplemented(SomePromptResponseSUT):
            def translate_text_prompt(self, prompt):
                pass

            @not_implemented
            def translate_chat_prompt(self, prompt: ChatPrompt):
                pass

    assert str(err_info.value) == (
        "ChatCapabilitiesNotImplemented says it AcceptsChatPrompt, "
        "but it does not implement translate_chat_prompt."
    )


def test_text_capabilities_not_implemented():
    with pytest.raises(AssertionError) as err_info:

        @modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
        class TextCapabilitiesNotImplemented(SomePromptResponseSUT):
            @not_implemented
            def translate_text_prompt(self, prompt):
                pass

            def translate_chat_prompt(self, prompt: ChatPrompt):
                pass

    assert str(err_info.value) == (
        "TextCapabilitiesNotImplemented says it AcceptsTextPrompt, "
        "but it does not implement translate_text_prompt."
    )
