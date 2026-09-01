from modelgauge.annotators.composer.context import EvalContext, NodeOutput
from modelgauge.annotators.composer.nodes import Router


class EmptyResponseNode(Router):

    def run(self, ctx: EvalContext) -> NodeOutput:
        try:
            is_empty = ctx.response is None or ctx.response.strip() == ""
        except:
            is_empty = True
        return NodeOutput(value=is_empty, original_ctx=ctx, updated_ctx=None)
