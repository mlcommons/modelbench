from pydantic import BaseModel
from newhelm.prompt import ChatPrompt, TextPrompt
from newhelm.sut import SUTCompletion, SUTResponse, PromptResponseSUT
from newhelm.sut_decorator import newhelm_sut
from newhelm.sut_registry import SUTS


class DemoConstantRequest(BaseModel):
    """This SUT just returns whatever you configured"""

    configured_response: str


class DemoConstantResponse(BaseModel):
    """This SUT is only capable of returning the configured text."""

    configured_response: str


@newhelm_sut()
class DemoConstantSUT(PromptResponseSUT[DemoConstantRequest, DemoConstantResponse]):
    """This SUT allows you to configure the response it will always give."""

    def __init__(self, uid: str, response_text: str):
        super().__init__(uid)
        self.response_text = response_text

    def translate_text_prompt(self, prompt: TextPrompt) -> DemoConstantRequest:
        return DemoConstantRequest(configured_response=self.response_text)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> DemoConstantRequest:
        return DemoConstantRequest(configured_response=self.response_text)

    def evaluate(self, request: DemoConstantRequest) -> DemoConstantResponse:
        assert self.response_text == request.configured_response
        return DemoConstantResponse(configured_response=request.configured_response)

    def translate_response(
        self, request: DemoConstantRequest, response: DemoConstantResponse
    ) -> SUTResponse:
        return SUTResponse(
            completions=[SUTCompletion(text=response.configured_response)]
        )


# Everything after the class name gets passed to the class.
SUTS.register(DemoConstantSUT, "demo_always_angry", "I hate you!")
# You can use kwargs if you want.
SUTS.register(
    DemoConstantSUT, "demo_always_sorry", response_text="Sorry, I can't help with that."
)
