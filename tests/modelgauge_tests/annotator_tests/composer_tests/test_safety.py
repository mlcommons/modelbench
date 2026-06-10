import pytest
from modelgauge_tests.annotator_tests.composer_tests.mocks import FailingNode

from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotators.composed_annotator import (
    AnnotatorArbiter,
    Safety,
    SafetyDAGAnnotator,
)
from modelgauge.annotators.composer.dag import (
    Composer,
    FailedDAGOutput,
    NodeExecutionError,
)
from modelgauge.annotators.composer.verdict import Verdict
from modelgauge.annotators.demo_annotator import DemoYBadAnnotator
from modelgauge.prompt import TextPrompt
from modelgauge.sut import SUTResponse


def test_safety_annotator_arbiter(sample_ctx):
    annotator = DemoYBadAnnotator("demo_annotator")
    arbiter = AnnotatorArbiter(name="demo_arbiter", annotator=annotator)
    output = arbiter.run(sample_ctx)
    assert output.value.is_safe
    assert isinstance(output.value, Safety)
    assert arbiter.verdict_type == Safety


def test_safety_dag_run(simple_dag, sample_ctx):
    safety_annotator = SafetyDAGAnnotator("safety", simple_dag)
    output = safety_annotator.process(
        prompt=TextPrompt(text=sample_ctx.prompt),
        response=SUTResponse(text=sample_ctx.response),
    )
    assert not output.is_safe
    assert isinstance(output, SafetyAnnotation)


def test_safety_dag_with_bad_verdict_type():
    with pytest.raises(
        ValueError,
        match="All outputs of the DAG must be of type Safety.",
    ):
        SafetyDAGAnnotator("bad_dag", Composer("bad_dag", verdict_type=Verdict))


def test_safety_dag_with_bad_node(sample_ctx, threshold_arbiter):
    failing_node = FailingNode(name="failing_node", routes=["threshold_arbiter"])
    dag = (
        Composer(
            "bad_node_dag",
            verdict_type=Safety,
        )
        .add_node(failing_node)
        .add_node(threshold_arbiter)
    )
    dag_output = dag.run(sample_ctx)
    assert isinstance(dag_output, FailedDAGOutput)
    assert str(dag_output.error.original_error) == "I'm afraid I can't do that, Dave."

    dag_annotator = SafetyDAGAnnotator("safety_annotator", dag)
    with pytest.raises(NodeExecutionError, match=r"Error executing node 'failing_node'") as e:
        dag_annotator.process(
            prompt=TextPrompt(text=sample_ctx.prompt),
            response=SUTResponse(text=sample_ctx.response),
        )
        assert type(e.value.original_error) == type(dag_output.error)
        assert "Traceback" in str(e.value)
