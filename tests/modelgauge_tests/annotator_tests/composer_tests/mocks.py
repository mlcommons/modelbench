from modelgauge.annotators.composed_annotator import Safety
from modelgauge.annotators.composer.context import EvalContext, NodeOutput
from modelgauge.annotators.composer.cost import CostInfo
from modelgauge.annotators.composer.nodes import (
    Arbiter,
    CacheableNodeMixin,
    Enricher,
    Gate,
    LLMCostMixin,
)
from modelgauge.annotators.composer.verdict import Verdict


def context_token_count(ctx: EvalContext) -> int:
    return len(ctx.prompt.split() + ctx.response.split())


class FailingNode(Enricher):

    def run(self, ctx: EvalContext) -> NodeOutput:
        raise RuntimeError("I'm afraid I can't do that, Dave.")


class PassthroughGate(Gate, LLMCostMixin):
    ROUTE_TO_TAKE: bool

    def run(self, ctx: EvalContext) -> NodeOutput:
        return self.build_output(self.ROUTE_TO_TAKE, ctx)

    def input_tokens(self, ctx: EvalContext) -> int:
        return context_token_count(ctx)

    def output_tokens(self, ctx: EvalContext) -> int:
        return 1


class AlwaysTrue(PassthroughGate):
    ROUTE_TO_TAKE = True


class AlwaysTrueCacheable(AlwaysTrue, CacheableNodeMixin):
    """Always-true gate that participates in Composer node-level disk caching."""

    run_count = 0

    def run(self, ctx: EvalContext) -> NodeOutput:
        type(self).run_count += 1
        return super().run(ctx)


class AlwaysFalse(PassthroughGate):
    ROUTE_TO_TAKE = False

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            input_cost_per_token=0.1,
            output_cost_per_token=0.2,
            fixed_cost=0.1,
            latency_seconds=0.1,
        )


class PromptLengthGate(Gate):
    def run(self, ctx: EvalContext) -> NodeOutput:
        return self.build_output(len(ctx.prompt) % 2 == 0, ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.2,
            latency_seconds=0.2,
        )


class Caser(Enricher, LLMCostMixin):
    def input_tokens(self, ctx: EvalContext) -> int:
        return len(ctx.response.split())

    def output_tokens(self, ctx: EvalContext) -> int:
        return len(ctx.response.split())


class LowerCaser(Caser):
    """Enriches by returning the response lowercased."""

    def run(self, ctx: EvalContext) -> NodeOutput:
        output = ctx.response.lower()
        return self.build_output(
            output,
            ctx,
            updated_ctx=ctx.with_response(output),
        )

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            input_cost_per_token=0.3,
            output_cost_per_token=0.4,
            fixed_cost=0.3,
            latency_seconds=0.3,
        )


class UpperCaser(Caser):
    """Enriches by returning the response uppercased."""

    def run(self, ctx: EvalContext) -> NodeOutput:
        output = ctx.response.upper()
        return self.build_output(
            output,
            ctx,
            updated_ctx=ctx.with_response(output),
        )

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            input_cost_per_token=0.4,
            output_cost_per_token=0.5,
            fixed_cost=0.4,
            latency_seconds=0.4,
        )


class LLMEnricher(Enricher, LLMCostMixin):

    def run(self, ctx: EvalContext) -> NodeOutput:
        return self.build_output(ctx.response, ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            input_cost_per_token=0.5,
            output_cost_per_token=0.6,
            fixed_cost=0.5,
            latency_seconds=0.5,
        )

    def input_tokens(self, ctx: EvalContext) -> int:
        return context_token_count(ctx)

    def output_tokens(self, ctx: EvalContext) -> int:
        return context_token_count(ctx)


class NoOpEnricher(Enricher):
    """Passes context through without changing it."""

    def run(self, ctx: EvalContext) -> NodeOutput:
        return self.build_output(None, ctx)


class FixedScorer(Enricher):
    """Returns a fixed float score regardless of context."""

    def __init__(self, name: str, value: float, **kwargs):
        super().__init__(name, **kwargs)
        self.value = value

    def run(self, ctx: EvalContext) -> NodeOutput:
        return self.build_output(self.value, ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.6,
            latency_seconds=0.6,
        )


class LowerCaseScorer(Enricher):
    """Scores based on the percentage of lowercase characters in the response."""

    def run(self, ctx: EvalContext) -> NodeOutput:
        if not ctx.response:
            return self.build_output(0.0, ctx)
        num_lower = sum(1 for c in ctx.response if c.islower())
        return self.build_output(num_lower / len(ctx.response), ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.7,
            latency_seconds=0.7,
        )


class UpperCaseScorer(Enricher):
    """Scores based on the percentage of uppercase characters in the response."""

    def run(self, ctx: EvalContext) -> NodeOutput:
        if not ctx.response:
            return self.build_output(0.0, ctx)
        num_upper = sum(1 for c in ctx.response if c.isupper())
        return self.build_output(num_upper / len(ctx.response), ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.8,
            latency_seconds=0.8,
        )


class AlwaysUnsafe(Arbiter):
    def run(self, ctx: EvalContext) -> NodeOutput:
        return self.build_output(Safety(is_safe=False), ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=0.9,
            latency_seconds=0.9,
        )

    @property
    def verdict_type(self) -> type:
        return Safety


class AlwaysSafe(Arbiter):
    def run(self, ctx: EvalContext) -> NodeOutput:
        return self.build_output(Safety(is_safe=True), ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=1.0,
            latency_seconds=1.0,
        )

    @property
    def verdict_type(self) -> type:
        return Safety


class ThresholdArbiter(Arbiter):
    def __init__(self, name: str, threshold: float, **kwargs):
        super().__init__(name, **kwargs)
        self.threshold = threshold

    def run(self, ctx: EvalContext) -> NodeOutput:
        score_outputs = ctx.parent_outputs()
        scores = [output.value for output in score_outputs]
        score = sum(scores) / len(scores)
        return self.build_output(Safety(is_safe=score < self.threshold), ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=1.1,
            latency_seconds=1.1,
        )

    @property
    def verdict_type(self) -> type:
        return Safety


class UnexpectedOutput(Verdict):
    @property
    def name(self) -> str:
        return "UNEXPECTED_OUTPUT"


class UnexpectedArbiter(Arbiter):
    """An arbiter that returns an output not declared in outputs()."""

    def run(self, ctx: EvalContext) -> NodeOutput:
        return self.build_output(UnexpectedOutput(), ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=1.2,
            latency_seconds=1.2,
        )

    @property
    def verdict_type(self) -> type:
        return UnexpectedOutput


class BadArbiter(Arbiter):
    """An arbiter that violates the contract by returning a non-Output value."""

    def run(self, ctx: EvalContext) -> NodeOutput:
        return self.build_output("safe", ctx)

    @property
    def cost(self) -> CostInfo:
        return CostInfo(
            fixed_cost=1.3,
            latency_seconds=1.3,
        )

    @property
    def verdict_type(self) -> type:
        return Safety
