from typing import Sequence

from modelgauge.annotator import Annotator
from modelgauge.annotators.composer.context import EvalContext, NodeOutput
from modelgauge.annotators.composer.nodes import Enricher
from modelgauge.sut import SUTResponse


class AnnotatorNode(Enricher):
    def __init__(self, name: str, annotator: Annotator, routes: Sequence[str]):
        super().__init__(name=name, routes=routes)
        self.annotator = annotator

    def run(self, ctx: EvalContext) -> NodeOutput:
        annotation = self.annotator.process(ctx.to_test_item(), SUTResponse(text=ctx.response))
        return self.build_output(annotation, ctx)
