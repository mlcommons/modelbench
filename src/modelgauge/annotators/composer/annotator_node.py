from typing import Sequence

from modelgauge.annotator import Annotator
from modelgauge.annotators.composer.context import EvalContext, NodeOutput
from modelgauge.annotators.composer.nodes import Enricher
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse


class AnnotatorNode(Enricher):
    def __init__(self, name: str, annotator: Annotator, routes: Sequence[str]):
        super().__init__(name=name, routes=routes)
        self.annotator = annotator

    def run(self, ctx: EvalContext) -> NodeOutput:
        prompt = TextPrompt(text=ctx.prompt)
        response = SUTResponse(text=ctx.response)
        request = self.annotator.translate_prompt(prompt, response)
        raw = self.annotator.annotate(request)
        annotation = self.annotator.translate_response(request, raw)
        return self.build_output(annotation, ctx)
