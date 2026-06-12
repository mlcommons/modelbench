from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from modelgauge.annotators.composer.cost import RealizedCost


@dataclass
class NodeOutput:
    value: Any
    original_ctx: EvalContext
    realized_cost: RealizedCost = field(default_factory=RealizedCost)
    updated_ctx: Optional[EvalContext] = None

    def to_dict(self) -> dict:
        return {
            "value": str(self.value),
            "realized_cost": self.realized_cost.to_dict(),
            "updated_ctx": self.updated_ctx.to_dict() if self.updated_ctx else None,
            "original_ctx": self.original_ctx.to_dict(),
        }


class EvalContext:
    """Context state passed around during DAG execution."""

    def __init__(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.prompt = prompt
        self.response = response
        self.metadata = metadata or {}
        self._parent_outputs: dict[str, NodeOutput] = {}

    def with_parent_outputs(self, outputs: dict[str, NodeOutput]) -> EvalContext:
        updated_ctx = None
        for node_output in outputs.values():
            if node_output.updated_ctx:
                if updated_ctx and node_output.updated_ctx != updated_ctx:
                    # confirm prompt/response are the same
                    if (
                        node_output.updated_ctx.prompt != updated_ctx.prompt
                        or node_output.updated_ctx.response != updated_ctx.response
                    ):
                        raise ValueError(
                            "If context is updated, all parent outputs must have the same updated prompt/response."
                        )
                    updated_ctx = updated_ctx.with_metadata_updates(
                        node_output.updated_ctx.metadata,
                        raise_on_conflict=True,
                    )
                elif not updated_ctx:
                    updated_ctx = node_output.updated_ctx
        if updated_ctx:
            ctx = EvalContext(
                prompt=updated_ctx.prompt,
                response=updated_ctx.response,
                metadata=updated_ctx.metadata,
            )
        else:
            ctx = EvalContext(
                prompt=self.prompt,
                response=self.response,
                metadata=self.metadata,
            )
        ctx._parent_outputs = outputs
        return ctx

    def parent_outputs(self) -> list[NodeOutput]:
        """Return the NodeOutput for a specific node, or None if it was skipped."""
        return list(self._parent_outputs.values())

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata,
        }

    def with_prompt(self, new_prompt: str) -> EvalContext:
        return EvalContext(
            prompt=new_prompt,
            response=self.response,
            metadata=self.metadata,
        )

    def with_response(self, new_response: str) -> EvalContext:
        return EvalContext(
            prompt=self.prompt,
            response=new_response,
            metadata=self.metadata,
        )

    def with_metadata(self, new_metadata: dict[str, Any]) -> EvalContext:
        """
        Return a new EvalContext with the provided metadata replacing the
        original metadata.
        """
        return EvalContext(
            prompt=self.prompt,
            response=self.response,
            metadata=new_metadata,
        )

    def with_metadata_updates(self, updates: dict[str, Any], raise_on_conflict: bool = False) -> EvalContext:
        """
        Return a new EvalContext with the original metadata updated with the
        provided updates.

        If raise_on_conflict is True, raises a ValueError if there are
        overlapping keys with conflicting values between the original metadata
        and the updates with different values, to prevent unintentional
        overwrites.
        """
        if raise_on_conflict and any(k in self.metadata and self.metadata[k] != v for k, v in updates.items()):
            raise ValueError("Metadata updates cannot have overlapping keys with different values.")
        new_metadata = self.metadata.copy()
        new_metadata.update(updates)
        return EvalContext(
            prompt=self.prompt,
            response=self.response,
            metadata=new_metadata,
        )

    def with_updates(
        self,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> EvalContext:
        return EvalContext(
            prompt=prompt or self.prompt,
            response=response or self.response,
            metadata=metadata or self.metadata,
        )

    def hash(self):
        return hash(
            (
                self.prompt,
                self.response,
                json.dumps(self.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            )
        )

    def __eq__(self, value) -> bool:
        if not isinstance(value, EvalContext):
            return False
        return self.prompt == value.prompt and self.response == value.response and self.metadata == value.metadata
