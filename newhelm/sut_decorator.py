from functools import wraps
import inspect
from typing import Sequence, Type
from newhelm.record_init import add_initialization_record
from newhelm.sut import SUT, PromptResponseSUT, SUTResponse
from newhelm.sut_capabilities import ProducesPerTokenLogProbabilities, SUTCapability


def newhelm_sut(capabilities: Sequence[Type[SUTCapability]]):
    """Decorator providing common behavior and hooks for all NewHELM SUTs."""

    def inner(cls):
        assert issubclass(
            cls, SUT
        ), "Decorator can only be applied to classes that inherit from SUT."
        cls.capabilities = capabilities
        cls.__init__ = _wrap_init(cls.__init__)
        if issubclass(cls, PromptResponseSUT):
            _override_translate_response(cls)
        cls._newhelm_sut = True
        return cls

    return inner


def assert_is_sut(obj):
    if not getattr(obj, "_newhelm_sut", False):
        raise AssertionError(
            f"{obj.__class__.__name__} should be decorated with @newhelm_sut."
        )


def _wrap_init(init):
    """Wrap the SUT __init__ function to verify it behaves as expected."""

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
    assert params[1].name == "uid", "All SUTs must have UID as the first parameter."


def _override_translate_response(cls: Type[PromptResponseSUT]) -> None:
    """Wrap the SUT translate_response function to verify it behaves as expected."""

    original = cls.translate_response

    if hasattr(original, "_newhelm_wrapped"):
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

    inner._newhelm_wrapped = True  # type: ignore [attr-defined]
    cls.translate_response = inner  # type: ignore [method-assign]
