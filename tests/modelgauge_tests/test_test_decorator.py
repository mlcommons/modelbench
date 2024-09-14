import pytest
from modelgauge.base_test import BaseTest, PromptResponseTest
from modelgauge.prompt import ChatPrompt, SUTOptions, TextPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.single_turn_prompt_response import PromptWithContext, TestItem
from modelgauge.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
    ProducesPerTokenLogProbabilities,
)
from modelgauge.test_decorator import assert_is_test, modelgauge_test


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class SomeTest(BaseTest):
    def __init__(self, uid, arg1):
        self.uid = uid
        self.arg1 = arg1


def test_basic():
    result = SomeTest(1234, 2)
    assert result.uid == 1234
    assert result.arg1 == 2
    assert result.requires_sut_capabilities == [AcceptsTextPrompt]
    assert result._modelgauge_test
    assert_is_test(result)


class NoDecorator(BaseTest):
    def __init__(self, uid, arg1):
        self.uid = uid
        self.arg1 = arg1


def test_no_decorator():
    result = NoDecorator(1234, 2)
    with pytest.raises(AssertionError) as err_info:
        assert_is_test(result)
    assert (
        str(err_info.value) == "NoDecorator should be decorated with @modelgauge_test."
    )


@modelgauge_test(requires_sut_capabilities=[])
class ChildTestCallsSuper(SomeTest):
    def __init__(self, uid, arg1, arg2):
        super().__init__(uid, arg1)
        self.arg2 = arg2


def test_child_calls_super():
    result = ChildTestCallsSuper(1234, 2, 3)
    assert result.uid == 1234
    assert result._modelgauge_test
    assert result.initialization_record == InitializationRecord(
        module="tests.test_test_decorator",
        class_name="ChildTestCallsSuper",
        args=[1234, 2, 3],
        kwargs={},
    )


@modelgauge_test(requires_sut_capabilities=[])
class ChildTestNoSuper(SomeTest):
    def __init__(self, uid, arg1, arg2):
        self.uid = uid
        self.arg1 = arg1
        self.arg2 = arg2


def test_child_no_super():
    result = ChildTestNoSuper(1234, 2, 3)
    assert result.uid == 1234
    assert result._modelgauge_test
    assert result.initialization_record == InitializationRecord(
        module="tests.test_test_decorator",
        class_name="ChildTestNoSuper",
        args=[1234, 2, 3],
        kwargs={},
    )


@modelgauge_test(requires_sut_capabilities=[])
class ChildTestNoInit(SomeTest):
    pass


def test_child_init():
    result = ChildTestNoInit(1234, 2)
    assert result.uid == 1234
    assert result._modelgauge_test
    assert result.initialization_record == InitializationRecord(
        module="tests.test_test_decorator",
        class_name="ChildTestNoInit",
        args=[1234, 2],
        kwargs={},
    )


def test_bad_signature():
    with pytest.raises(AssertionError) as err_info:
        # Exception happens without even constructing an instance.
        @modelgauge_test(requires_sut_capabilities=[])
        class ChildBadSignature(SomeTest):
            def __init__(self, arg1, uid):
                self.uid = uid
                self.arg1 = arg1

    assert "All Tests must have UID as the first parameter." in str(err_info.value)


class SomePromptResponseTest(PromptResponseTest):
    # Define all the abstract methods to make other subclasses easier
    def get_dependencies(self):
        pass

    def make_test_items(self, dependency_helper):
        pass

    def get_annotators(self):
        pass

    def measure_quality(self, item):
        pass

    def aggregate_measurements(self, items):
        pass


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class LogprobsNotRequiredNotRequested(SomePromptResponseTest):
    def make_test_items(self, dependency_helper):
        return [
            TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(text="some-text"), source_id=None
                    )
                ]
            )
        ]


def test_logprobs_not_required_not_requested():
    test = LogprobsNotRequiredNotRequested("some-test")
    # Mostly check that no error is raised
    assert len(test.make_test_items(None)) == 1


@modelgauge_test(
    requires_sut_capabilities=[ProducesPerTokenLogProbabilities, AcceptsTextPrompt]
)
class LogprobsRequiredNotRequested(SomePromptResponseTest):
    def make_test_items(self, dependency_helper):
        return [
            TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(text="some-text"), source_id=None
                    )
                ]
            )
        ]


def test_logprobs_required_not_requested():
    test = LogprobsRequiredNotRequested("some-test")
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(None)
    assert "LogprobsRequiredNotRequested lists ProducesPerTokenLogProbabilities" in str(
        err_info.value
    )


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class LogprobsNotRequiredAndRequested(SomePromptResponseTest):
    def make_test_items(self, dependency_helper):
        return [
            TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(
                            text="some-text", options=SUTOptions(top_logprobs=1)
                        ),
                        source_id=None,
                    )
                ]
            )
        ]


def test_logprobs_not_required_and_requested():
    test = LogprobsNotRequiredAndRequested("some-test")
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(None)
    assert (
        "LogprobsNotRequiredAndRequested specified the SUT option top_logprobs"
        in str(err_info.value)
    )


@modelgauge_test(
    requires_sut_capabilities=[ProducesPerTokenLogProbabilities, AcceptsTextPrompt]
)
class LogprobsRequiredAndRequested(SomePromptResponseTest):
    def make_test_items(self, dependency_helper):
        return [
            TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(
                            text="some-text", options=SUTOptions(top_logprobs=1)
                        ),
                        source_id=None,
                    )
                ]
            )
        ]


def test_logprobs_required_and_requested():
    test = LogprobsRequiredAndRequested("some-test")
    # Mostly check that no error is raised
    assert len(test.make_test_items(None)) == 1


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class LogprobsInheritsRequested(LogprobsRequiredAndRequested):
    pass


def test_logprobs_inherits_requested():
    test = LogprobsInheritsRequested("some-test")
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(None)
    assert "LogprobsInheritsRequested specified the SUT option top_logprobs" in str(
        err_info.value
    )


@modelgauge_test(
    requires_sut_capabilities=[ProducesPerTokenLogProbabilities, AcceptsTextPrompt]
)
class LogprobsInheritsNotRequested(LogprobsNotRequiredNotRequested):
    pass


def test_logprobs_inherits_not_requested():
    test = LogprobsInheritsNotRequested("some-test")
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(None)
    assert "LogprobsInheritsNotRequested lists ProducesPerTokenLogProbabilities" in str(
        err_info.value
    )


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class MakeTextRequireText(SomePromptResponseTest):
    def make_test_items(self, dependency_helper):
        return [
            TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(text="some-text"), source_id=None
                    )
                ]
            )
        ]


def test_make_text_require_text():
    test = MakeTextRequireText("some-test")
    # Mostly check that no error is raised
    assert len(test.make_test_items(None)) == 1


@modelgauge_test(requires_sut_capabilities=[])
class MakeTextRequireNone(SomePromptResponseTest):
    def make_test_items(self, dependency_helper):
        return [
            TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(text="some-text"), source_id=None
                    )
                ]
            )
        ]


def test_make_text_require_none():
    test = MakeTextRequireNone("some-test")
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(None)
    assert str(err_info.value) == (
        "MakeTextRequireNone produces TextPrompt but does not "
        "requires_sut_capabilities AcceptsTextPrompt."
    )


@modelgauge_test(requires_sut_capabilities=[])
class MakeChatRequireNone(SomePromptResponseTest):
    def make_test_items(self, dependency_helper):
        return [
            TestItem(
                prompts=[
                    PromptWithContext(prompt=ChatPrompt(messages=[]), source_id=None)
                ]
            )
        ]


def test_make_chat_require_none():
    test = MakeChatRequireNone("some-test")
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(None)
    assert str(err_info.value) == (
        "MakeChatRequireNone produces ChatPrompt but does not "
        "requires_sut_capabilities AcceptsChatPrompt."
    )


@modelgauge_test(requires_sut_capabilities=[AcceptsChatPrompt])
class MakeTextRequireChat(SomePromptResponseTest):
    def make_test_items(self, dependency_helper):
        return [
            TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(text="some-text"), source_id=None
                    )
                ]
            )
        ]


def test_make_text_require_chat():
    test = MakeTextRequireChat("some-test")
    with pytest.raises(AssertionError) as err_info:
        test.make_test_items(None)
    assert str(err_info.value) == (
        "MakeTextRequireChat produces TextPrompt but does not "
        "requires_sut_capabilities AcceptsTextPrompt."
    )


@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class MakeTextRequireBoth(SomePromptResponseTest):
    def make_test_items(self, dependency_helper):
        return [
            TestItem(
                prompts=[
                    PromptWithContext(
                        prompt=TextPrompt(text="some-text"), source_id=None
                    )
                ]
            )
        ]


def test_make_text_require_both():
    test = MakeTextRequireBoth("some-test")
    # This is allowed in case the class conditionally makes chat prompts.
    assert len(test.make_test_items(None)) == 1
