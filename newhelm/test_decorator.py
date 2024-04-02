import inspect
from dataclasses import dataclass
from functools import wraps
from newhelm.base_test import BaseTest, PromptResponseTest
from newhelm.dependency_helper import DependencyHelper
from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.record_init import add_initialization_record
from newhelm.single_turn_prompt_response import TestItem
from newhelm.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
    ProducesPerTokenLogProbabilities,
    SUTCapability,
)
from typing import List, Sequence, Type


def newhelm_test(requires_sut_capabilities: Sequence[Type[SUTCapability]]):
    """Decorator providing common behavior and hooks for all NewHELM Tests."""

    def inner(cls):
        assert issubclass(
            cls, BaseTest
        ), "Decorator can only be applied to classes that inherit from BaseTest."
        cls.requires_sut_capabilities = requires_sut_capabilities
        cls.__init__ = _wrap_init(cls.__init__)
        if issubclass(cls, PromptResponseTest):
            _override_make_test_items(cls)
        cls._newhelm_test = True
        return cls

    return inner


def assert_is_test(obj):
    if not getattr(obj, "_newhelm_test", False):
        raise AssertionError(
            f"{obj.__class__.__name__} should be decorated with @newhelm_test."
        )


def _wrap_init(init):
    """Wrap the Test __init__ function to verify it behaves as expected."""

    if hasattr(init, "_newhelm_wrapped"):
        # Already wrapped, no need to do any work.
        return init

    _validate_init_signature(init)

    @wraps(init)
    def wrapped_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        add_initialization_record(self, *args, **kwargs)

    wrapped_init._newhelm_wrapped = True
    return wrapped_init


def _validate_init_signature(init):
    params = list(inspect.signature(init).parameters.values())
    assert params[1].name == "uid", "All Tests must have UID as the first parameter."


@dataclass
class PromptTypeHandling:
    """Helper class for verifying the handling of a prompt type."""

    prompt_type: Type
    capability: Type[SUTCapability]
    test_obj: PromptResponseTest
    produces: bool = False

    def update_producing(self, prompt):
        self.produces |= isinstance(prompt, self.prompt_type)

    def assert_handled(self):
        required = self.capability in self.test_obj.requires_sut_capabilities
        test_name = self.test_obj.__class__.__name__
        prompt_type_name = self.prompt_type.__name__
        capability_name = self.capability.__name__
        if self.produces and not required:
            raise AssertionError(
                f"{test_name} produces {prompt_type_name} but does not requires_sut_capabilities {capability_name}."
            )
        # Tests may conditionally produce a prompt type, so requirements are a superset.


def _override_make_test_items(cls: Type[PromptResponseTest]) -> None:
    """Wrap the Test make_test_items function to verify it behaves as expected."""

    original = cls.make_test_items

    if hasattr(original, "_newhelm_wrapped"):
        # Already wrapped, no need to do any work.
        return

    @wraps(original)
    def inner(self, dependency_helper: DependencyHelper) -> List[TestItem]:
        items: List[TestItem] = original(self, dependency_helper)
        requires_logprobs = (
            ProducesPerTokenLogProbabilities in self.requires_sut_capabilities
        )
        prompt_types = [
            PromptTypeHandling(
                prompt_type=TextPrompt,
                capability=AcceptsTextPrompt,
                test_obj=self,
            ),
            PromptTypeHandling(
                prompt_type=ChatPrompt,
                capability=AcceptsChatPrompt,
                test_obj=self,
            ),
        ]
        any_request_logprobs = False
        for item in items:
            for prompt in item.prompts:
                any_request_logprobs |= prompt.prompt.options.top_logprobs is not None
                for prompt_type in prompt_types:
                    prompt_type.update_producing(prompt.prompt)

        if any_request_logprobs and not requires_logprobs:
            raise AssertionError(
                f"{self.__class__.__name__} specified the SUT option top_logprobs, "
                f"but did not list ProducesPerTokenLogProbabilities as a "
                f"required capability. If it doesn't actually need top_logprobs, "
                f"remove setting the option."
            )

        if not any_request_logprobs and requires_logprobs:
            raise AssertionError(
                f"{self.__class__.__name__} lists ProducesPerTokenLogProbabilities "
                f"as required, but did not request the SUT option top_logprobs. "
                f"If it doesn't actually need top_logprobs, remove specifying the capability."
            )

        for prompt_type in prompt_types:
            prompt_type.assert_handled()
        return items

    inner._newhelm_wrapped = True  # type: ignore [attr-defined]
    cls.make_test_items = inner  # type: ignore [method-assign]
