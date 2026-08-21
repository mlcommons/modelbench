"""
Node types for the Composer pipeline.

Class hierarchy:

    ComposerNode (ABC)
    ├── Router     (routes to other nodes based on run output)
    ├── Enricher   (produces arbitary output; routes forward unconditionally)
    └── Arbiter    (produces Output)
    Output         (terminal; carries a verdict value)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from modelgauge.annotators.composer.context import EvalContext, NodeOutput
from modelgauge.annotators.composer.cost import CostInfo, RealizedCost
from modelgauge.annotators.composer.verdict import Verdict


class ComposerNode(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.validate()

    @abstractmethod
    def all_routes(self) -> list[str | Verdict]:
        """Return a list of all possible route targets from this node."""
        raise NotImplementedError

    @abstractmethod
    def all_route_paths(self) -> list[list[str | Verdict]]:
        """Return a list of possible route paths, separated by branches."""
        raise NotImplementedError

    @abstractmethod
    def next_nodes(self, output_value: Any) -> tuple[str | Verdict, ...]:
        """Given the node's output value, return the tuple of next node names to activate."""
        raise NotImplementedError

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

    def validate(self) -> None:
        """Validate that the node's routing configuration is consistent with its type."""
        # validate that routes with Verdicts only have one Verdict
        for route_list in self.all_route_paths():
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


class NoRouteError(Exception):
    """Error raised when no route is found for a given output value."""

    def __init__(self, node_name: str, output_value: Any) -> None:
        super().__init__(f"Node {node_name} could not find a route for output value {output_value} in its route map.")


class Router(ComposerNode):
    """
    Node that routes to other nodes based on the output value.
    If the output value is not in the route map, the default_route is used. If no default_route is provided, an error is raised.
    """

    def __init__(
        self,
        name: str,
        route_map: Dict[bool | str, Sequence[str | Verdict]],
        default_route: Optional[Sequence[str | Verdict]] = None,
    ) -> None:
        self.route_map: Dict[bool | str, tuple[str | Verdict, ...]] = {k: tuple(v) for k, v in route_map.items()}
        self.default_route: Optional[tuple[str | Verdict, ...]] = (
            tuple(default_route) if default_route is not None else None
        )
        super().__init__(name)

    def all_routes(self) -> list[str | Verdict]:
        return [target for path in self.all_route_paths() for target in path]

    def all_route_paths(self) -> list[list[str | Verdict]]:
        routes = [list(routes) for routes in self.route_map.values()]
        if self.default_route is not None:
            routes.append(list(self.default_route))
        return routes

    def next_nodes(self, output_value: Any) -> tuple[str | Verdict, ...]:
        if output_value not in self.route_map:
            if self.default_route is None:
                raise NoRouteError(self.name, output_value)
            return self.default_route
        return self.route_map[output_value]


class Enricher(ComposerNode):
    """Context transformation node."""

    def __init__(
        self,
        name: str,
        routes: Sequence[str | Verdict],
    ) -> None:
        self.routes: tuple[str | Verdict, ...] = tuple(routes)
        super().__init__(name)

    def all_routes(self) -> list[str | Verdict]:
        return list(self.routes)

    def all_route_paths(self) -> list[list[str | Verdict]]:
        return [self.all_routes()]

    def next_nodes(self, output_value: Any) -> tuple[str | Verdict, ...]:
        return self.routes


class Arbiter(ComposerNode):
    """Terminal node. Takes context and returns a Verdict indicating the final verdict (based on routes)."""

    def all_routes(self) -> list[str | Verdict]:
        return []

    def all_route_paths(self) -> list[list[str | Verdict]]:
        return []

    def next_nodes(self, output_value: Any) -> tuple[str | Verdict, ...]:
        return ()

    @property
    @abstractmethod
    def verdict_type(self) -> type:
        """Return the expected type of the Verdict's value for validation."""
        raise NotImplementedError  # pragma: no cover
