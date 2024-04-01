import pytest
from newhelm.record_init import InitializationRecord
from newhelm.sut import SUT, PromptResponseSUT, SUTCompletion, SUTResponse
from newhelm.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
    ProducesPerTokenLogProbabilities,
)
from newhelm.sut_decorator import assert_is_sut, newhelm_sut


@newhelm_sut(capabilities=[AcceptsTextPrompt])
class SomeSUT(SUT):
    def __init__(self, uid, arg1):
        self.uid = uid
        self.arg1 = arg1


def test_basic():
    result = SomeSUT(1234, 2)
    assert result.uid == 1234
    assert result.arg1 == 2
    assert result.capabilities == [AcceptsTextPrompt]
    assert result._newhelm_sut
    assert_is_sut(result)


class NoDecorator(SUT):
    def __init__(self, uid, arg1):
        self.uid = uid
        self.arg1 = arg1


def test_no_decorator():
    result = NoDecorator(1234, 2)
    with pytest.raises(AssertionError) as err_info:
        assert_is_sut(result)
    assert str(err_info.value) == "NoDecorator should be decorated with @newhelm_sut."


@newhelm_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
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


@newhelm_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
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


@newhelm_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
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
        @newhelm_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
        class ChildBadSignature(SomeSUT):
            def __init__(self, arg1, uid):
                self.uid = uid
                self.arg1 = arg1

    assert "All SUTs must have UID as the first parameter." in str(err_info.value)


class SomePromptResponseSUT(PromptResponseSUT):
    # Define abstract methods to make subclasses easier to make.
    def translate_text_prompt(self, prompt):
        pass

    def translate_chat_prompt(self, prompt):
        pass

    def evaluate(self, request):
        pass

    def translate_response(self, request, response):
        pass


@newhelm_sut(capabilities=[AcceptsTextPrompt])
class LogprobsNoCapNotSet(SomePromptResponseSUT):
    def translate_response(self, request, response):
        return SUTResponse(completions=[SUTCompletion(text="some-text")])


def test_logprobs_no_cap_not_set():
    sut = LogprobsNoCapNotSet("some-sut")
    # Mostly here to ensure no exceptions
    assert sut.translate_response(None, None).completions[0].text == "some-text"


@newhelm_sut(capabilities=[AcceptsTextPrompt])
class LogprobsNoCapAndSet(SomePromptResponseSUT):
    def translate_response(self, request, response):
        return SUTResponse(
            completions=[SUTCompletion(text="some-text", top_logprobs=[])]
        )


def test_logprobs_no_cap_and_set():
    sut = LogprobsNoCapAndSet("some-sut")
    with pytest.raises(AssertionError) as err_info:
        sut.translate_response(None, None)
    assert (
        "LogprobsNoCapAndSet does not list capability ProducesPerTokenLogProbabilities"
        in str(err_info.value)
    )


@newhelm_sut(capabilities=[ProducesPerTokenLogProbabilities, AcceptsTextPrompt])
class LogprobsHasCapNotSet(SomePromptResponseSUT):
    def translate_response(self, request, response):
        return SUTResponse(completions=[SUTCompletion(text="some-text")])


def test_logprobs_has_cap_not_set():
    sut = LogprobsHasCapNotSet("some-sut")
    # This is allowed because SUTOption might not be set
    assert sut.translate_response(None, None).completions[0].text == "some-text"


@newhelm_sut(capabilities=[ProducesPerTokenLogProbabilities, AcceptsTextPrompt])
class LogprobsHasCapAndSet(SomePromptResponseSUT):
    def translate_response(self, request, response):
        return SUTResponse(
            completions=[SUTCompletion(text="some-text", top_logprobs=[])]
        )


def test_logprobs_has_cap_and_set():
    sut = LogprobsHasCapAndSet("some-sut")
    assert sut.translate_response(None, None).completions[0].text == "some-text"


@newhelm_sut(capabilities=[AcceptsTextPrompt])
class LogprobsInheritsSet(LogprobsHasCapAndSet):
    pass


def test_logprobs_inherits_set():
    sut = LogprobsInheritsSet("some-sut")
    with pytest.raises(AssertionError) as err_info:
        sut.translate_response(None, None)
    assert (
        "LogprobsInheritsSet does not list capability ProducesPerTokenLogProbabilities"
        in str(err_info.value)
    )
