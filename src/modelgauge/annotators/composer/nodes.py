"""
Node types for the Composer pipeline.

Class hierarchy:

    ComposerNode (ABC)
    ├── Gate       (binary test; routes on True/False)
    ├── Enricher   (produces arbitary output; routes forward unconditionally)
    └── Arbiter    (produces Output)
    Output         (terminal; carries a verdict value)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from modelgauge.annotators.composer.context import EvalContext, NodeOutput
from modelgauge.annotators.composer.cost import CostInfo, RealizedCost
from modelgauge.annotators.composer.verdict import Verdict


class ComposerNode(ABC):
    def __init__(
        self,
        name: str,
        routes_true: Optional[Sequence[str | Verdict]] = None,
        routes_false: Optional[Sequence[str | Verdict]] = None,
        routes: Optional[Sequence[str | Verdict]] = None,
    ) -> None:
        self.name = name
        self._routes_true: tuple[str | Verdict, ...] = tuple(routes_true or [])
        self._routes_false: tuple[str | Verdict, ...] = tuple(routes_false or [])
        self._routes: tuple[str | Verdict, ...] = tuple(routes or [])
        self.validate()

    @property
    def routes_true(self) -> tuple[str | Verdict, ...]:
        return self._routes_true

    @property
    def routes_false(self) -> tuple[str | Verdict, ...]:
        return self._routes_false

    @property
    def routes(self) -> tuple[str | Verdict, ...]:
        return self._routes

    @abstractmethod
    def run(self, ctx: EvalContext) -> NodeOutput:
        """Execute the node and return its output and realized cost."""
        raise NotImplementedError  # pragma: no cover

    def build_output(
        self,
        value: Any,
        ctx: EvalContext,
        updated_ctx: Optional[EvalContext] = None,
    ) -> NodeOutput:
        """Helper method for building a NodeOutput with the node's realized cost
        when the cost doesn't have to be computed concurrently with the output value.

        This helper assumes the context is not updated.
        """
        return NodeOutput(
            value=value,
            realized_cost=self.realized_cost(ctx),
            original_ctx=ctx,
            updated_ctx=updated_ctx,
        )

    @property
    def cost(self) -> CostInfo:
        """Override this to represent the cost of running this node."""
        return CostInfo()

    def realized_cost(self, ctx: EvalContext) -> RealizedCost:
        """Base realized cost when ctx doesn't affect (see LLMNodeMixin for context-aware cost)."""
        return RealizedCost(
            fixed_cost=self.cost.fixed_cost,
            latency_seconds=self.cost.latency_seconds,
        )

    def __repr__(self) -> str:
        return f"{self.name!r}: ({self.__class__.__name__})"

    @staticmethod
    def format_output(output: Any) -> str:
        """Convenience method to format the node's output for debugging/visualization."""
        if isinstance(output, float):
            return f"{output:.3g}"
        s = str(output)
        return s if len(s) <= 30 else s[:27] + "..."

    def all_routes(self) -> list[str | Verdict]:
        """Return a list of all route targets from this node."""
        return [*self.routes_true, *self.routes_false, *self.routes]

    def next_nodes(self, output_value: Any) -> tuple[str | Verdict, ...]:
        """Given the node's output value, return the tuple of next node names to activate."""
        if isinstance(self, Gate):
            return self.routes_true if output_value else self.routes_false
        else:
            return self.routes

    def validate(self) -> None:
        """Validate that the node's routing configuration is consistent with its type."""
        # validate that routes with Verdicts only have one Verdict
        for route_list in [self.routes_true, self.routes_false, self.routes]:
            output_routes = [r for r in route_list if isinstance(r, Verdict)]
            if len(output_routes) > 1:
                raise ValueError(f"{self!r} has multiple Verdict routes {output_routes}, which is not allowed.")


class CacheableNodeMixin(ComposerNode, ABC):
    """Mixin for nodes whose outputs should be cached."""

    def cache_key(self, ctx: EvalContext) -> int:
        return ctx.hash()


class LLMCostMixin(ComposerNode):
    """Mixin for nodes that involve LLM calls, to simplify cost calculation."""

    @abstractmethod
    def input_tokens(self, ctx: EvalContext) -> int:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def output_tokens(self, ctx: EvalContext) -> int:
        raise NotImplementedError  # pragma: no cover

    def realized_cost(self, ctx: EvalContext) -> RealizedCost:
        return RealizedCost(
            input_token_cost=self.input_tokens(ctx) * self.cost.input_cost_per_token,
            output_token_cost=self.output_tokens(ctx) * self.cost.output_cost_per_token,
            fixed_cost=self.cost.fixed_cost,
            latency_seconds=self.cost.latency_seconds,
        )


def _validate_binary_routes(node: ComposerNode) -> None:
    if not node.routes_true or not node.routes_false:
        raise ValueError(f"{node!r} requires both routes_true and routes_false")
    if node.routes:
        raise ValueError(f"{node!r} should not have routes= (use routes_true= / routes_false=)")


def _validate_unary_routes(node: ComposerNode) -> None:
    if not node.routes:
        raise ValueError(f"{node!r} requires routes=")
    if node.routes_true or node.routes_false:
        raise ValueError(f"{node!r} should not have routes_true= / routes_false= (use routes=)")


def _validate_terminal(node: ComposerNode) -> None:
    if node.routes_true or node.routes_false or node.routes:
        raise ValueError(f"{node!r} is terminal and cannot have routing kwargs")


class Gate(ComposerNode):
    """Binary test node."""

    def validate(self) -> None:
        super().validate()
        _validate_binary_routes(self)


class Enricher(ComposerNode):
    """Context transformation node."""

    def validate(self) -> None:
        super().validate()
        _validate_unary_routes(self)


class Arbiter(ComposerNode):
    """Takes context and returns a Verdict indicating the final verdict (based on routes)."""

    def validate(self) -> None:
        super().validate()
        _validate_terminal(self)

    @property
    @abstractmethod
    def verdict_type(self) -> type:
        """Return the expected type of the Verdict's value for validation."""
        raise NotImplementedError  # pragma: no cover
