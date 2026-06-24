from pydantic import BaseModel

from modelgauge.prompt import TextPromptWithMetadata
from modelgauge.sut import SUTResponse


class AnnotationRequest(BaseModel):
    annotator: str
    prompt: str = ""
    response: str
    metadata: dict = {}  # optional metadata to pass along with the request; used for benchmarking

    def to_prompt(self) -> TextPromptWithMetadata:
        return TextPromptWithMetadata(text=self.prompt, metadata=self.metadata)

    def to_response(self) -> SUTResponse:
        return SUTResponse(text=self.response)
