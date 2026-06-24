from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotator import Annotator, SUTResponse, TextPrompt
from modelgauge.annotators.composer.context import EvalContext
from modelgauge.annotators.composer.dag import Composer, SuccessfulDAGOutput
from modelgauge.annotators.composer.nodes import Arbiter, CacheableNodeMixin, NodeOutput
from modelgauge.annotators.composer.verdict import Verdict
from modelgauge.prompt import ChatPrompt, TextPrompt, TextPromptWithMetadata
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import SUTResponse


class DAGAnnotator(Annotator):
    """Annotator that executes a DAG."""

    def __init__(self, uid: str, dag: Composer) -> None:
        super().__init__(uid)
        self.dag = dag

    def translate_prompt(
        self,
        prompt: TextPrompt | ChatPrompt,
        response: SUTResponse,
    ) -> EvalContext:
        prompt_str = prompt.text if isinstance(prompt, TextPrompt) else format_chat(prompt)
        metadata = prompt.metadata if isinstance(prompt, TextPromptWithMetadata) else {}

        return EvalContext(
            prompt=prompt_str,
            response=response.text,
            metadata=metadata,
        )

    def annotate(self, annotation_request: EvalContext) -> SuccessfulDAGOutput:
        dag_output = self.dag.run(annotation_request)
        if isinstance(dag_output, SuccessfulDAGOutput):
            return dag_output
        else:
            raise dag_output.error


class Safety(Verdict):

    def __init__(self, is_safe: bool) -> None:
        self.is_safe = is_safe

    @property
    def name(self) -> str:
        return "SAFE" if self.is_safe else "UNSAFE"


class SafetyArbiter(Arbiter):
    @property
    def verdict_type(self) -> type:
        return Safety


class SafetyDAGAnnotator(DAGAnnotator):
    """Implementation of DAGAnnotator that produces a SafetyAnnotation."""

    def __init__(self, uid: str, dag: Composer) -> None:
        super().__init__(uid, dag)
        if not issubclass(dag.verdict_type, Safety):
            raise ValueError("All outputs of the DAG must be of type Safety.")

    def translate_response(
        self,
        request: EvalContext,
        response: SuccessfulDAGOutput,
    ) -> SafetyAnnotation:
        """Map DAGResult verdict to a SafetyAnnotation (is_safe bool)."""
        assert isinstance(response.verdict, Safety), "Safety DAG output verdict must be of type Safety."
        return SafetyAnnotation(
            is_safe=response.verdict.is_safe,
            is_valid=True,
            metadata=response.to_dict(skip_cost=True),
        )


class AnnotatorArbiter(SafetyArbiter, CacheableNodeMixin):
    """Arbiter that outputs SAFE or UNSAFE based on the output of a (safety) Annotator."""

    def __init__(self, name: str, annotator: Annotator) -> None:
        super().__init__(name=name)
        self.annotator = annotator

    def run(self, ctx: EvalContext) -> NodeOutput:
        prompt = TextPrompt(text=ctx.prompt)
        response = SUTResponse(text=ctx.response)
        annotation = self.annotator.process(prompt, response)
        val = Safety(is_safe=annotation.is_safe)
        return NodeOutput(value=val, original_ctx=ctx)
