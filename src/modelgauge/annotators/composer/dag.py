"""Composer implementation."""

import collections
import functools
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Optional

import pandas as pd  # type: ignore
from airrlogger.log_config import get_logger
from tqdm import tqdm

from modelbench.cache import DiskCache, MBCache, NullCache
from modelgauge.annotators.composer.context import EvalContext, NodeOutput
from modelgauge.annotators.composer.cost import CostInfo, RealizedCost
from modelgauge.annotators.composer.nodes import (
    Arbiter,
    CacheableNodeMixin,
    ComposerNode,
    Gate,
)
from modelgauge.annotators.composer.verdict import Verdict

logger = get_logger(__name__)


def requires_validate_and_build(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._validate_and_build()
        return method(self, *args, **kwargs)

    return wrapper


@dataclass
class _DAGOutput:
    node_outputs: dict[str, NodeOutput]
    total_cost: RealizedCost

    def to_dict(self, skip_cost=False) -> dict:
        d = {
            "node_outputs": {k: v.to_dict(skip_cost=skip_cost) for k, v in self.node_outputs.items()},
        }
        if not skip_cost:
            d["total_cost"] = self.total_cost.to_dict()
        return d


@dataclass
class SuccessfulDAGOutput(_DAGOutput):
    verdict: Verdict

    def to_dict(self, skip_cost=False) -> dict:
        d = super().to_dict(skip_cost=skip_cost)
        d["verdict"] = self.verdict.name
        return d


@dataclass
class FailedDAGOutput(_DAGOutput):
    error: Exception


class NodeExecutionError(Exception):
    def __init__(self, node_name: str, original_error: Exception):
        formatted = traceback.format_exception(original_error)
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Error executing node '{node_name}': {''.join(formatted)}")


class ComposerColumnNames:
    def __init__(
        self,
        composer_name: Optional[str] = None,
        output_col_name: Optional[str] = None,
        error_col_name: Optional[str] = None,
        dag_run_col_name: Optional[str] = None,
        cost_col_name: Optional[str] = None,
    ):
        if (
            any(
                not name
                for name in [
                    output_col_name,
                    error_col_name,
                    dag_run_col_name,
                    cost_col_name,
                ]
            )
            and composer_name is None
        ):
            raise ValueError(
                "If any of the column names are not provided, composer_name must be provided to generate default column names."
            )

        self.output_col = output_col_name or f"{composer_name}_output"
        self.error_col = error_col_name or f"{composer_name}_error"
        self.dag_run_col = dag_run_col_name or f"{composer_name}_dag_run"
        self.cost_col = cost_col_name or f"{composer_name}_dag_cost"


class Composer:
    """DAG of ComposerNodes.

    Usage:

        refusal_gate     = MyRefusalGate("RefusalGate", routes_true=[Score(value=1)], routes_false=["NonRefusal"])
        eval_non_refusal = MyNonRefusalEvaluator("NonRefusal", routes=["Arbiter"])
        arbiter          = MyArbiter("Arbiter")

        dag = (
            Composer("refusal_gated_safety_evaluator", verdict_type=Safety)
            .add_node(refusal_gate)
            .add_node(eval_non_refusal)
            .add_node(arbiter)
        )

        # run single
        result = dag.run(prompt_uid="123", prompt="...", response="...")
        # run batch
        results_df = dag.run_dataframe(df)
    """

    def __init__(
        self,
        name: str,
        verdict_type: type,
        cache_path: Optional[Path] = None,
        col_names: Optional[ComposerColumnNames] = None,
    ) -> None:
        self.name = name
        self._nodes: dict[str, ComposerNode] = {}
        self._root_nodes: list[str] = []
        self._ordered: list[str] = []
        self._validated: bool = False
        self._predecessors: dict[str, list[str]] = collections.defaultdict(list)
        if not issubclass(verdict_type, Verdict):
            raise ValueError("verdict_type must be a subclass of Verdict.")
        self._verdict_type = verdict_type
        self._cache_path = cache_path
        self._node_caches: dict[str, MBCache] = {}
        self._col_names = col_names or ComposerColumnNames(composer_name=name)

    @property
    def verdict_type(self) -> type:
        return self._verdict_type

    @property
    def df_output_col(self) -> str:
        return self._col_names.output_col

    @property
    def df_error_col(self) -> str:
        return self._col_names.error_col

    @property
    def df_dag_run_col(self) -> str:
        return self._col_names.dag_run_col

    @property
    def df_cost_col(self) -> str:
        return self._col_names.cost_col

    def add_node(
        self,
        node: ComposerNode,
    ) -> "Composer":
        """Register a node with its routes."""

        if node.name in self._nodes:
            raise ValueError(f"A different node named {node.name} is already registered.")
        self._nodes[node.name] = node
        self._validated = False
        if isinstance(node, CacheableNodeMixin):
            self._node_caches[node.name] = DiskCache(self._cache_path / node.name) if self._cache_path else NullCache()
        return self

    def _validate_and_build(self) -> None:
        """
        Validate the DAG:
        - All routes reference registered nodes or instances of the output type.
        - No cycles.
        - All paths lead to an instance of the output type.

        Build:
        - _predecessors: dict mapping node name to list of parent node names (for context during execution)
        - _root_nodes: list of node names with no incoming routes (starting points)
        - _ordered: list of node names in topological order (valid execution order)
        """
        # skip validation if we've already done it and the DAG hasn't changed
        if self._validated:
            return

        # check that all route targets reference registered nodes or instances
        # of the output type, and that all Arbiters have compatible output types
        for node_name, node in self._nodes.items():
            if isinstance(node, Arbiter):
                if not issubclass(node.verdict_type, self.verdict_type):
                    raise ValueError(
                        f"Node {node_name} is an Arbiter with verdict_type {node.verdict_type.__name__}, which is not compatible with the DAG's verdict_type {self.verdict_type.__name__}."
                    )
            for target in node.all_routes():
                if target not in self._nodes and not isinstance(target, self.verdict_type):
                    raise ValueError(f"Node {node_name} routes to unregistered node {target} or incompatible output.")

        # check for cycles (kahn's algorithm)
        all_routes = {name: node.all_routes() for name, node in self._nodes.items()}
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for routes in all_routes.values():
            for route in routes:
                if isinstance(route, Verdict):
                    continue
                in_degree[route] += 1

        root_nodes = [n for n in self._nodes if in_degree[n] == 0]
        queue = collections.deque(root_nodes)
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for child in all_routes.get(current, []):
                if isinstance(child, Verdict):
                    continue
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(self._nodes):
            nodes_in_cycle = set(self._nodes) - set(ordered)
            raise ValueError(f"DAG contains a cycle. Nodes in cycle: {nodes_in_cycle}")

        # build predecessors
        for name, node in self._nodes.items():
            for target in node.all_routes():
                if isinstance(target, Verdict):
                    continue
                self._predecessors[target].append(name)

        self._validated = True
        self._root_nodes = root_nodes
        self._ordered = ordered

    def _run_node(self, node: ComposerNode, ctx: EvalContext) -> NodeOutput:
        if isinstance(node, CacheableNodeMixin):
            key = node.cache_key(ctx)
            if key in self._node_caches[node.name]:
                return self._node_caches[node.name][key]
            else:
                output = node.run(ctx)
                self._node_caches[node.name][key] = output
                return output
        else:
            return node.run(ctx)

    def _run_traced(self, ctx: EvalContext) -> tuple[SuccessfulDAGOutput | FailedDAGOutput, set[tuple[str, str]]]:
        """Execute the DAG and return (final verdict, node outputs, realized costs, traversed edges)."""
        node_outputs: dict[str, NodeOutput] = {}
        traversed_edges: set[tuple[str, str]] = set()
        total_cost = RealizedCost()
        reachable: set[str] = set(self._root_nodes)
        for node_name in self._ordered:
            if node_name not in reachable:
                continue
            try:
                ctx = ctx.with_parent_outputs(
                    {pred: node_outputs[pred] for pred in self._predecessors[node_name] if pred in node_outputs}
                )
                node = self._nodes[node_name]
                output = self._run_node(node, ctx)
            except Exception as e:
                wrapped_error = NodeExecutionError(node_name, e)
                return (
                    FailedDAGOutput(
                        node_outputs=node_outputs,
                        total_cost=total_cost,
                        error=wrapped_error,
                    ),
                    traversed_edges,
                )
            node_outputs[node_name] = output
            total_cost += output.realized_cost
            if isinstance(output.value, Verdict):
                traversed_edges.add((node_name, output.value.name))
                dag_output = SuccessfulDAGOutput(
                    verdict=output.value,
                    node_outputs=node_outputs,
                    total_cost=total_cost,
                )
                return dag_output, traversed_edges
            for target in node.next_nodes(output.value):
                t = target if isinstance(target, str) else target.name
                traversed_edges.add((node_name, t))
                if isinstance(target, Verdict):
                    return (
                        SuccessfulDAGOutput(
                            verdict=target,
                            node_outputs=node_outputs,
                            total_cost=total_cost,
                        ),
                        traversed_edges,
                    )
                reachable.add(t)
        raise ValueError("DAG execution completed without reaching a Verdict node.")

    @requires_validate_and_build
    def run(self, ctx: EvalContext) -> SuccessfulDAGOutput | FailedDAGOutput:
        """Execute the DAG on a single prompt/response and get the output,
        node outputs, and overall realized cost."""
        dag_output, _ = self._run_traced(ctx)
        return dag_output

    @requires_validate_and_build
    def run_dataframe(
        self,
        df: pd.DataFrame,
        prompt_col: str = "prompt",
        response_col: str = "response",
        metadata_col: Optional[str] = None,
        metadata_cols: Optional[list[str] | bool] = None,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """
        Run the DAG over every row of a DataFrame.

        Each row in the input DataFrame is converted into an EvalContext using the
        prompt and response columns.

        If a metadata column is provided, its value is parsed as JSON and included in the context.
        If metadata_cols are provided, the metadata is constructed as a dictionary from their values.

        Parameters:
            df: Input DataFrame containing prompt/response rows.
            prompt_col: Column name for the prompt text.
            response_col: Column name for the response text.
            metadata_col: Optional column name containing JSON-encoded metadata.
            metadata_cols: Optional list of metadata column names (currently unused).
                If True, all columns other than prompt_col and response_col are treated as metadata columns.
            n_jobs: Number of worker threads to use when evaluating rows.

        Returns:
            DataFrame with the original columns and appended result columns.
        """

        def _extract_metadata_json(row: pd.Series) -> dict[str, Any]:
            if metadata_col:
                row_val = row[metadata_col]
                if row_val:
                    try:
                        return json.loads(row_val)
                    except Exception as e:
                        logger.warning("Failed to parse json metadata in row. Proceeding with no metadata.")
                        logger.debug(f"Metadata parsing error: {e}")
            return {}

        def _extract_metadata_cols(row: pd.Series) -> dict[str, Any]:
            cols: list[str] = []
            if metadata_cols is True:
                cols = [col for col in df.columns if col not in {prompt_col, response_col}]
            elif isinstance(metadata_cols, list):
                cols = metadata_cols
            return row[cols].to_dict() if cols else {}  # type: ignore

        if metadata_col is not None and metadata_cols is not None:
            raise ValueError("Cannot specify both metadata_col and metadata_cols.")
        elif metadata_col:
            metadata_extractor = _extract_metadata_json
        elif metadata_cols:
            metadata_extractor = _extract_metadata_cols
        else:
            metadata_extractor = None

        def _run_row(row: pd.Series) -> SuccessfulDAGOutput | FailedDAGOutput:
            ctx = EvalContext(
                prompt=str(row[prompt_col]),
                response=str(row[response_col]),
                metadata=metadata_extractor(row) if metadata_extractor else None,
            )
            return self.run(ctx)

        rows = [row for _, row in df.iterrows()]

        if n_jobs == 1:
            records = [_run_row(row) for row in tqdm(rows, desc=self.name)]
        else:
            max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                records = list(tqdm(executor.map(_run_row, rows), total=len(rows), desc=self.name))

        result_df = pd.DataFrame(
            {
                self.df_output_col: [r.verdict.name if isinstance(r, SuccessfulDAGOutput) else None for r in records],
                self.df_error_col: [str(r.error) if isinstance(r, FailedDAGOutput) else None for r in records],
                self.df_dag_run_col: [json.dumps({k: v.to_dict() for k, v in r.node_outputs.items()}) for r in records],
                self.df_cost_col: [json.dumps(r.total_cost.to_dict()) for r in records],
            },
            index=df.index,
        )
        return pd.concat([df, result_df], axis=1)

    @requires_validate_and_build
    def potential_costs(self) -> dict[str, CostInfo]:
        """Run the DAG on all terminal paths and report total costs per path."""
        gates = [name for name, node in self._nodes.items() if isinstance(node, Gate)]
        path_costs: dict[str, CostInfo] = {}

        for combo in product([True, False], repeat=len(gates)):
            gate_outcomes = dict(zip(gates, combo))
            reachable: set[str] = set(self._root_nodes)
            path: list[str] = []
            total = CostInfo()

            for node_name in self._ordered:
                if node_name not in reachable:
                    continue
                node = self._nodes[node_name]
                total += node.cost
                path.append(node_name)
                if isinstance(node, Gate):
                    targets = node.routes_true if gate_outcomes[node_name] else node.routes_false
                elif isinstance(node, Arbiter):
                    targets = []  # type: ignore
                else:
                    targets = node.routes
                for target in targets:
                    if not isinstance(target, Verdict):
                        reachable.add(target if isinstance(target, str) else target.name)

            base_path = " -> ".join(path)
            path_costs[f"{base_path} -> Out ({self.verdict_type.__name__})"] = total

        return path_costs

    def _visualize(
        self,
        node_outputs: Optional[dict[str, Any]] = None,
        traversed_edges: Optional[set[tuple[str, str]]] = None,
        final_output: Optional[Verdict] = None,
        ctx: Optional[EvalContext] = None,
    ):  # pragma: no cover
        """Render the DAG as a PNG image. In a Jupyter notebook the image is displayed inline.

        When node_outputs/traversed_edges/final_output are provided (via visualize_run),
        the hot path is highlighted and each node shows its output value.

        NOTE: this helper method is vibe-coded and provided as-is.
        """
        import graphviz  # type: ignore
        from IPython.display import Image

        traced = node_outputs is not None

        _NODE_STYLES: dict[type, dict] = {
            Gate: {"shape": "diamond", "style": "filled", "fillcolor": "#ffe082"},
            Arbiter: {"shape": "hexagon", "style": "filled", "fillcolor": "#e1bee7"},
            Verdict: {
                "shape": "rectangle",
                "style": "filled,rounded",
                "fillcolor": "#dcedc8",
            },
        }
        _OUTPUT_TYPE_STYLE = {
            "shape": "rectangle",
            "style": "filled,rounded,dashed",
            "fillcolor": "#dcedc8",
        }
        _DEFAULT_STYLE = {
            "shape": "rectangle",
            "style": "filled",
            "fillcolor": "#eeeeee",
        }
        _DIM = {
            "style": "filled",
            "fillcolor": "#f0f0f0",
            "color": "#bbbbbb",
            "fontcolor": "#aaaaaa",
        }

        _NODE_W, _NODE_H = 1.5, 0.5  # inches, fixed for all nodes

        def _fontsize(label: str, max_fs: float = 11.0, min_fs: float = 7.0, fill: float = 0.8) -> str:
            """Scale font size so the longest line fits within _NODE_W.

            fill: fraction of the node width usable for text. Shapes like diamonds,
            hexagons, and parallelograms have less usable area than rectangles, so
            pass a smaller fill value for those.
            """
            longest = max((len(line) for line in label.split("\n")), default=1)
            # approx: each char ≈ 0.55 × fontsize points
            fs = (_NODE_W * 72 * fill) / (longest * 0.55)
            return f"{max(min_fs, min(max_fs, fs)):.1f}"

        dot = graphviz.Digraph(name=self.name)
        dot.attr(
            label=self.name,
            labelloc="t",
            fontsize="13",
            fontname="Helvetica",
            rankdir="LR",
            ranksep="0.5",
            nodesep="0.4",
        )
        dot.attr(
            "node",
            fontname="Helvetica",
            fontsize="11",
            width=str(_NODE_W),
            height=str(_NODE_H),
            fixedsize="true",
        )
        dot.attr("edge", fontname="Helvetica", fontsize="9")

        # implicit input node pinned to the left
        top = graphviz.Digraph()
        top.attr(rank="min")

        def _truncate(s: str, n: int = 24) -> str:
            return s if len(s) <= n else s[: n - 1] + "…"

        if ctx is not None:
            input_label = f"p: {_truncate(ctx.prompt)}\nr: {_truncate(ctx.response)}"
        else:
            input_label = "prompt\nresponse"
        top.node(
            "__input__",
            input_label,
            shape="parallelogram",
            style="filled",
            fillcolor="#b2dfdb",
            color="#4db6ac",
            fontcolor="#00695c",
            fontsize=_fontsize(input_label, fill=0.45),
        )
        dot.subgraph(top)

        # collect Verdict instances directly referenced in routes (from non-Arbiter nodes)
        direct_verdicts: dict[str, Verdict] = {}
        has_arbiter = any(isinstance(n, Arbiter) for n in self._nodes.values())
        for node in self._nodes.values():
            if not isinstance(node, Arbiter):
                for target in node.all_routes():
                    if isinstance(target, Verdict):
                        direct_verdicts[target.name] = target

        # whether the final output came from a direct route or an arbiter
        final_from_direct = traced and final_output in direct_verdicts.values()

        bottom = graphviz.Digraph()
        bottom.attr(rank="max")

        # individual nodes for directly-routed Verdict instances, shown with their repr
        for out_name, out_inst in direct_verdicts.items():
            attrs = dict(_NODE_STYLES[Verdict])
            if traced:
                if out_inst is final_output:
                    attrs["penwidth"] = "2.5"
                else:
                    attrs = dict(_DIM, shape="rectangle", style="filled,rounded")
            bottom.node(out_name, repr(out_inst), fontsize=_fontsize(repr(out_inst)), **attrs)

        # synthetic output type node for Arbiters
        if has_arbiter:
            output_node_id = f"__output_{self.verdict_type.__name__}__"
            output_label = f"{self.verdict_type.__name__} (?)"
            attrs = dict(_OUTPUT_TYPE_STYLE)
            if traced:
                if not final_from_direct and final_output is not None:
                    attrs = dict(_NODE_STYLES[Verdict])
                    attrs["penwidth"] = "2.5"
                    output_label = repr(final_output)
                elif final_from_direct:
                    attrs = dict(_DIM, shape="rectangle", style="filled,rounded")
            bottom.node(output_node_id, output_label, fontsize=_fontsize(output_label), **attrs)

        dot.subgraph(bottom)

        # processing nodes
        for node_name, node in self._nodes.items():
            base_style = next(
                (s for t, s in _NODE_STYLES.items() if isinstance(node, t)),
                _DEFAULT_STYLE,
            )
            node_was_active = (node_outputs is not None and node_name in node_outputs) or (
                traversed_edges is not None and any(src == node_name for src, _ in traversed_edges)
            )
            if traced and not node_was_active:
                attrs = dict(_DIM, shape=base_style.get("shape", "box"))
                label = node_name
            else:
                attrs = dict(base_style)
                if traced:
                    raw = node_outputs[node_name]  # type: ignore[index]
                    label = f"{node_name}\n{node.format_output(raw.value)}"
                    attrs["penwidth"] = "2.5"
                else:
                    label = node_name
            _fill = 0.45 if isinstance(node, Gate) else 0.65 if isinstance(node, Arbiter) else 0.8
            dot.node(node_name, label, fontsize=_fontsize(label, fill=_fill), **attrs)

        # edges from implicit input to root nodes
        for root in self._root_nodes:
            dot.edge("__input__", root, color="#888888")

        # edges between processing nodes
        for node_name, node in self._nodes.items():
            if isinstance(node, Gate):
                for target in node.routes_true:
                    t = target if isinstance(target, str) else target.name
                    hot = not traced or (node_name, t) in traversed_edges  # type: ignore[operator]
                    dot.edge(
                        node_name,
                        t,
                        label=" True",
                        color="#2e7d32" if hot else "#cccccc",
                        fontcolor="#2e7d32" if hot else "#cccccc",
                        penwidth="2" if hot and traced else "1",
                    )
                for target in node.routes_false:
                    t = target if isinstance(target, str) else target.name
                    hot = not traced or (node_name, t) in traversed_edges  # type: ignore[operator]
                    dot.edge(
                        node_name,
                        t,
                        label=" False",
                        color="#c62828" if hot else "#cccccc",
                        fontcolor="#c62828" if hot else "#cccccc",
                        penwidth="2" if hot and traced else "1",
                    )
            elif isinstance(node, Arbiter):
                output_node_id = f"__output_{self.verdict_type.__name__}__"
                hot = not traced or node_name in (node_outputs or {})
                dot.edge(
                    node_name,
                    output_node_id,
                    color="#555555" if hot else "#cccccc",
                    penwidth="2" if hot and traced else "1",
                )
            else:
                for target in node.routes:
                    t = target if isinstance(target, str) else target.name
                    hot = not traced or (node_name, t) in traversed_edges  # type: ignore[operator]
                    edge_label = ""
                    if traced and hot and node_name in (node_outputs or {}):
                        edge_label = f" {node.format_output(node_outputs[node_name].value)}"  # type: ignore[index]
                    dot.edge(
                        node_name,
                        t,
                        label=edge_label,
                        color="#555555" if hot else "#cccccc",
                        fontcolor="#555555" if hot else "#cccccc",
                        penwidth="2" if hot and traced else "1",
                    )

        try:
            return Image(dot.pipe(format="png"))
        except graphviz.ExecutableNotFound as e:
            raise RuntimeError(
                "Graphviz system binaries not found. Install them with:\n"
                "  macOS:  brew install graphviz\n"
                "  Ubuntu: apt-get install graphviz\n"
                "  conda:  conda install graphviz"
            ) from e

    @requires_validate_and_build
    def visualize(self):  # pragma: no cover
        """Render the DAG structure as a PNG image (inline in Jupyter notebooks).

        The graph flows left to right. Node shapes and colors:
          - Input          — teal parallelogram (implicit; represents the prompt/response pair)
          - Gate           — amber diamond; edges labelled "True" (green) / "False" (red)
          - Enricher       — light grey rectangle; edges are unlabelled
          - Arbiter        — light purple hexagon; edge labelled with the output type name
          - Output (direct instance)   — soft green rounded rectangle, solid border;
                                         label is repr(output)
          - Output (type placeholder)  — soft green rounded rectangle, dashed border;
                                         label is the class name; shown when the DAG contains
                                         an Arbiter whose concrete value is only known at runtime

        Raises:
            RuntimeError: if the Graphviz system binaries are not installed.
        """
        return self._visualize()

    @requires_validate_and_build
    def visualize_run(self, ctx: EvalContext):  # pragma: no cover
        """Run the DAG on ctx and return a visualization with the executed path highlighted.

        Identical layout to visualize(), with the following additions:
          - Active nodes are bolded and show their output value beneath the node name.
          - Inactive nodes are greyed out.
        """
        dag_output, traversed_edges = self._run_traced(ctx)
        return self._visualize(
            node_outputs=dag_output.node_outputs,
            traversed_edges=traversed_edges,
            final_output=(dag_output.verdict if isinstance(dag_output, SuccessfulDAGOutput) else None),
            ctx=ctx,
        )
