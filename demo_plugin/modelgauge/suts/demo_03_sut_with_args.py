from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from pydantic import BaseModel


class DemoConstantRequest(BaseModel):
    """This SUT just returns whatever you configured"""

    configured_response: str


class DemoConstantResponse(BaseModel):
    """This SUT is only capable of returning the configured text."""

    configured_response: str


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
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
