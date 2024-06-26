import inspect
from functools import wraps
from modelgauge.not_implemented import is_not_implemented
from modelgauge.record_init import add_initialization_record
from modelgauge.sut import SUT, PromptResponseSUT, SUTResponse
from modelgauge.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
    ProducesPerTokenLogProbabilities,
    SUTCapability,
)
from typing import Sequence, Type


def modelgauge_sut(capabilities: Sequence[Type[SUTCapability]]):
    """Decorator providing common behavior and hooks for all ModelGauge SUTs.

    Args:
       capabilities: List of capabilities being reported by the SUT.
    """

    def inner(cls):
        assert issubclass(
            cls, SUT
        ), "Decorator can only be applied to classes that inherit from SUT."
        cls.capabilities = capabilities
        cls.__init__ = _wrap_init(cls.__init__)
        if issubclass(cls, PromptResponseSUT):
            _assert_prompt_types(cls)
            _override_translate_response(cls)
        cls._modelgauge_sut = True
        return cls

    return inner


def assert_is_sut(obj):
    """Raise AssertionError if obj is not decorated with @modelgauge_sut."""
    if not getattr(obj, "_modelgauge_sut", False):
        raise AssertionError(
            f"{obj.__class__.__name__} should be decorated with @modelgauge_sut."
        )


def _wrap_init(init):
    """Wrap the SUT __init__ function to verify it behaves as expected."""

    if hasattr(init, "_modelgauge_wrapped"):
        # Already wrapped, no need to do any work.
        return init

    _validate_init_signature(init)

    @wraps(init)
    def wrapped_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        add_initialization_record(self, *args, **kwargs)

    wrapped_init._modelgauge_wrapped = True
    return wrapped_init


def _validate_init_signature(init):
    params = list(inspect.signature(init).parameters.values())
    assert params[1].name == "uid", "All SUTs must have UID as the first parameter."


def _override_translate_response(cls: Type[PromptResponseSUT]) -> None:
    """Wrap the SUT translate_response function to verify it behaves as expected."""

    original = cls.translate_response

    if hasattr(original, "_modelgauge_wrapped"):
        # Already wrapped, no need to do any work.
        return

    @wraps(original)
    def inner(self, request, response) -> SUTResponse:
        response = original(self, request, response)
        logprob_capable = ProducesPerTokenLogProbabilities in self.capabilities
        logprob_produced = False
        for completion in response.completions:
            logprob_produced |= completion.top_logprobs is not None
        if not logprob_capable and logprob_produced:
            raise AssertionError(
                f"{self.__class__.__name__} does not list capability "
                f"ProducesPerTokenLogProbabilities, but it sets the top_logprobs field."
            )
        # We can't assert the other way, as if the SUTOption isn't set, the SUT may
        # not return top_logprobs.
        return response

    inner._modelgauge_wrapped = True  # type: ignore [attr-defined]
    cls.translate_response = inner  # type: ignore [method-assign]


def _assert_prompt_types(cls: Type[PromptResponseSUT]):
    _assert_prompt_type(cls, AcceptsTextPrompt, cls.translate_text_prompt)
    _assert_prompt_type(cls, AcceptsChatPrompt, cls.translate_chat_prompt)


def _assert_prompt_type(cls, capability, method):
    accepts_type = capability in cls.capabilities
    implements_type = not is_not_implemented(method)
    if accepts_type and not implements_type:
        raise AssertionError(
            f"{cls.__name__} says it {capability.__name__}, but it does not implement {method.__name__}."
        )
    if not accepts_type and implements_type:
        raise AssertionError(
            f"{cls.__name__} implements {method.__name__}, but it does not say it {capability.__name__}."
        )
