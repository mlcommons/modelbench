import itertools
import jsonlines
import pytest
import time
from unittest.mock import MagicMock

from modelgauge.annotation_pipeline import (
    AnnotatorSource,
    AnnotatorAssigner,
    AnnotatorWorkers,
    AnnotatorSink,
    EnsembleVoter,
    JsonlAnnotatorOutput,
)
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.dataset import PromptResponseDataset
from modelgauge.data_schema import DEFAULT_PROMPT_RESPONSE_SCHEMA as PROMPT_RESPONSE_SCHEMA
from modelgauge.pipeline import Pipeline
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import (
    PromptOutput,
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
)
from modelgauge.single_turn_prompt_response import SUTInteraction, TestItem
from modelgauge.sut import SUTResponse
from modelgauge_tests.fake_annotator import (
    FakeAnnotation,
    FakeAnnotator,
    FakeAnnotatorRequest,
)
from modelgauge_tests.fake_sut import FakeSUT
from modelgauge_tests.test_prompt_pipeline import FakePromptInput


class FakeAnnotatorInput:
    def __init__(self, items: list[dict], delay=None):
        super().__init__()
        self.items = items
        self.delay = itertools.cycle(delay or [0])

    def __iter__(self):
        for row in self.items:
            time.sleep(next(self.delay))
            prompt = TestItem(
                prompt=TextPrompt(text=row[PROMPT_RESPONSE_SCHEMA.prompt_text]),
                source_id=row[PROMPT_RESPONSE_SCHEMA.prompt_uid],
                context=row,
            )
            response = SUTResponse(text=row[PROMPT_RESPONSE_SCHEMA.sut_response])
            yield SUTInteraction(prompt, row[PROMPT_RESPONSE_SCHEMA.sut_uid], response)


class FakeAnnotatorOutput(PromptOutput):
    def __init__(self):
        self.output = {}

    def write(self, item, annotations):
        self.output[item] = annotations


def make_sut_interaction(source_id, prompt, sut_uid, response):
    return SUTInteraction(
        TestItem(source_id=source_id, prompt=TextPrompt(text=prompt)),
        sut_uid,
        SUTResponse(text=response),
    )


def sut_interactions_is_equal(a, b):
    """Equality check that ignores the prompt's context attribute."""
    return (
        a.prompt.source_id == b.prompt.source_id
        and a.prompt.prompt.text == b.prompt.prompt.text
        and a.sut_uid == b.sut_uid
        and a.response == b.response
    )


def test_csv_annotator_input(tmp_path):
    file_path = tmp_path / "input.csv"
    file_path.write_text(
        f'{PROMPT_RESPONSE_SCHEMA.prompt_uid},{PROMPT_RESPONSE_SCHEMA.prompt_text},{PROMPT_RESPONSE_SCHEMA.sut_uid},{PROMPT_RESPONSE_SCHEMA.sut_response}\n"1","a","s","b"'
    )
    input = PromptResponseDataset(file_path, mode="r")

    assert len(input) == 1
    item: SUTInteraction = next(iter(input))
    assert sut_interactions_is_equal(item, make_sut_interaction("1", "a", "s", "b"))


def test_json_annotator_output(tmp_path):
    file_path = tmp_path / "output.jsonl"
    with JsonlAnnotatorOutput(file_path) as output:
        output.write(make_sut_interaction("1", "a", "sut1", "b"), {"fake": "x"})
        output.write(make_sut_interaction("2", "c", "sut2", "d"), {"fake": "y"})

    with jsonlines.open(file_path) as reader:
        items: list[dict] = [i for i in reader]
        assert len(items) == 2
        assert items[0] == {
            PROMPT_RESPONSE_SCHEMA.prompt_uid: "1",
            PROMPT_RESPONSE_SCHEMA.prompt_text: "a",
            PROMPT_RESPONSE_SCHEMA.sut_uid: "sut1",
            PROMPT_RESPONSE_SCHEMA.sut_response: "b",
            "Annotations": {"fake": "x"},
        }
        assert items[1] == {
            PROMPT_RESPONSE_SCHEMA.prompt_uid: "2",
            PROMPT_RESPONSE_SCHEMA.prompt_text: "c",
            PROMPT_RESPONSE_SCHEMA.sut_uid: "sut2",
            PROMPT_RESPONSE_SCHEMA.sut_response: "d",
            "Annotations": {"fake": "y"},
        }


def test_json_annotator_output_different_annotation_types(tmp_path):
    file_path = tmp_path / "output.jsonl"
    annotations = {
        "fake1": {"sut_text": "a"},
        "fake2": {"sut_text": "b", "num": 0},
        "fake3": "c",
    }
    with JsonlAnnotatorOutput(file_path) as output:
        output.write(make_sut_interaction("1", "a", "s", "b"), annotations)

    with jsonlines.open(file_path) as reader:
        assert reader.read()["Annotations"] == annotations


@pytest.mark.parametrize("output_fname", ["output.csv", "output.json"])
def test_csv_annotator_output_invalid(tmp_path, output_fname):
    file_path = tmp_path / output_fname
    with pytest.raises(AssertionError, match=f"Invalid output file {file_path}. Must be of type JSONL."):
        JsonlAnnotatorOutput(file_path)


@pytest.fixture
def annotators():
    annotator_pydantic = FakeAnnotator("annotator_pydantic")
    annotator_dict = FakeAnnotator("annotator_dict")
    # Return the same annotation but as a dict.
    annotator_dict.translate_response = MagicMock(
        side_effect=lambda *args: annotator_pydantic.translate_response(*args).model_dump()
    )
    annotator_dummy = FakeAnnotator("dummy")
    annotator_dummy.translate_response = MagicMock(return_value="d")
    return {"annotator_pydantic": annotator_pydantic, "annotator_dict": annotator_dict, "dummy": annotator_dummy}


@pytest.mark.parametrize(
    "annotator_uid,annotation",
    [
        ("annotator_pydantic", FakeAnnotation(sut_text="response")),
        ("annotator_dict", {"sut_text": "response"}),
        ("dummy", "d"),
    ],
)
def test_annotator_worker_normal(annotators, annotator_uid, annotation):
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    w = AnnotatorWorkers(annotators)
    result = w.handle_item((sut_interaction, annotator_uid))

    assert result[0] == sut_interaction
    assert result[1] == annotator_uid
    assert result[2] == annotation


def test_annotator_worker_cache_simple(annotators, tmp_path):
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    w = AnnotatorWorkers(annotators, cache_path=tmp_path)

    # Tests that first call invokes the annotator and the second call uses the cache.
    assert annotators["annotator_pydantic"].annotate_calls == 0
    for _ in range(2):
        _, _, annotation = w.handle_item((sut_interaction, "annotator_pydantic"))
        assert annotation == FakeAnnotation(sut_text="response")
        assert annotators["annotator_pydantic"].annotate_calls == 1


def test_annotator_worker_unique_responses(annotators, tmp_path):
    """Different responses have different cache keys for annotator with response-based requests."""
    w = AnnotatorWorkers(annotators, cache_path=tmp_path)

    assert annotators["annotator_pydantic"].annotate_calls == 0
    w.handle_item((make_sut_interaction("", "", "", "response 1"), "annotator_pydantic"))
    assert annotators["annotator_pydantic"].annotate_calls == 1
    w.handle_item((make_sut_interaction("", "", "", "response 2"), "annotator_pydantic"))
    assert annotators["annotator_pydantic"].annotate_calls == 2

    # Non-response SUT interaction attributes do not affect the cache key.
    w.handle_item((make_sut_interaction("2", "2", "2", "response 2"), "annotator_pydantic"))
    assert annotators["annotator_pydantic"].annotate_calls == 2


def test_annotator_worker_cache_unique_prompts(tmp_path):
    """Different prompts have different cache keys for annotator with prompt-based requests."""

    annotator = FakeAnnotator("a")
    annotator.translate_request = MagicMock(
        side_effect=lambda prompt, response: FakeAnnotatorRequest(text=prompt.prompt.text)
    )
    w = AnnotatorWorkers({"a": annotator}, cache_path=tmp_path)

    # Different prompt texts.
    assert annotator.annotate_calls == 0
    w.handle_item((make_sut_interaction("", "prompt 1", "", ""), "a"))
    assert annotator.annotate_calls == 1
    w.handle_item((make_sut_interaction("", "prompt 2", "", ""), "a"))
    assert annotator.annotate_calls == 2


def test_annotator_worker_cache_different_annotators(annotators, tmp_path):
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    w = AnnotatorWorkers(annotators, cache_path=tmp_path)

    assert annotators["annotator_pydantic"].annotate_calls == 0
    assert annotators["annotator_dict"].annotate_calls == 0

    w.handle_item((sut_interaction, "annotator_pydantic"))
    assert annotators["annotator_pydantic"].annotate_calls == 1
    assert annotators["annotator_dict"].annotate_calls == 0

    w.handle_item((sut_interaction, "annotator_dict"))
    assert annotators["annotator_pydantic"].annotate_calls == 1
    assert annotators["annotator_dict"].annotate_calls == 1


def test_annotator_worker_retries_until_success():
    num_exceptions = 3
    mock = MagicMock()
    exceptions = [Exception() for _ in range(num_exceptions)]
    mock.side_effect = exceptions + [FakeAnnotation(sut_text="response")]
    annotator = FakeAnnotator("fake-annotator")
    annotator.annotate = mock

    w = AnnotatorWorkers({"fake-annotator": annotator})
    w.sleep_time = 0.01
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    result = w.handle_item((sut_interaction, "fake-annotator"))

    assert mock.call_count == num_exceptions + 1
    assert (sut_interaction, "fake-annotator", FakeAnnotation(sut_text="response")) == result


class FakeEnsemble(AnnotatorSet):
    annotators = []

    def __init__(self, annotators):
        self.annotators = annotators

    def evaluate(self, item):
        return {"ensemble_vote": 1.0}


@pytest.mark.parametrize("annotator_uid", ["annotator_pydantic", "random"])
def test_ensemble_worker_puts_all_items(annotator_uid):
    ensemble = FakeEnsemble(annotators=["annotator_pydantic"])
    w = EnsembleVoter(ensemble)
    assert w._queue.qsize() == 0

    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    annotation = FakeAnnotation(sut_text="response")
    w.handle_item((sut_interaction, annotator_uid, annotation))

    assert w._queue.qsize() > 0
    item = w._queue.get()

    assert item[0] == sut_interaction
    assert item[1] == annotator_uid
    assert item[2] == annotation


def test_ensemble_worker_computes_ensemble_with_all_annotators():
    ensemble = FakeEnsemble(annotators=["annotator_pydantic", "dummy"])
    w = EnsembleVoter(ensemble)
    assert w._queue.qsize() == 0

    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    annotation = FakeAnnotation(sut_text="response")
    w.handle_item((sut_interaction, "annotator_pydantic", annotation))
    assert w._queue.qsize() == 1  # Should just pass the first annotation through

    w.handle_item((sut_interaction, "dummy", annotation))
    assert w._queue.qsize() == 3  # Should pass second annotation + final ensemble annotation
    w._queue.get()
    w._queue.get()
    item = w._queue.get()

    assert item[0] == sut_interaction
    assert item[1] == "ensemble"
    assert item[2] == {"ensemble_vote": 1.0}


def test_full_run(annotators):
    input = FakeAnnotatorInput(
        [
            {
                PROMPT_RESPONSE_SCHEMA.prompt_uid: "1",
                PROMPT_RESPONSE_SCHEMA.prompt_text: "a",
                PROMPT_RESPONSE_SCHEMA.sut_response: "b",
                PROMPT_RESPONSE_SCHEMA.sut_uid: "s",
            },
            {
                PROMPT_RESPONSE_SCHEMA.prompt_uid: "2",
                PROMPT_RESPONSE_SCHEMA.prompt_text: "c",
                PROMPT_RESPONSE_SCHEMA.sut_response: "d",
                PROMPT_RESPONSE_SCHEMA.sut_uid: "s",
            },
        ]
    )
    output = FakeAnnotatorOutput()
    p = Pipeline(
        AnnotatorSource(input),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=1),
        AnnotatorSink(annotators, output),
        debug=True,
    )
    p.run()

    assert len(output.output) == len(input.items)
    interactions = sorted(list(output.output.keys()), key=lambda o: o.prompt.source_id)
    assert sut_interactions_is_equal(interactions[0], make_sut_interaction("1", "a", "s", "b"))
    assert output.output[interactions[0]] == {
        "annotator_pydantic": {"sut_text": "b"},
        "annotator_dict": {"sut_text": "b"},
        "dummy": "d",
    }
    assert sut_interactions_is_equal(interactions[1], make_sut_interaction("2", "c", "s", "d"))
    assert output.output[interactions[1]] == {
        "annotator_pydantic": {"sut_text": "d"},
        "annotator_dict": {"sut_text": "d"},
        "dummy": "d",
    }


def test_full_run_with_ensemble(annotators):
    input = FakeAnnotatorInput(
        [
            {
                PROMPT_RESPONSE_SCHEMA.prompt_uid: "1",
                PROMPT_RESPONSE_SCHEMA.prompt_text: "a",
                PROMPT_RESPONSE_SCHEMA.sut_response: "b",
                PROMPT_RESPONSE_SCHEMA.sut_uid: "s",
            },
            {
                PROMPT_RESPONSE_SCHEMA.prompt_uid: "2",
                PROMPT_RESPONSE_SCHEMA.prompt_text: "c",
                PROMPT_RESPONSE_SCHEMA.sut_response: "d",
                PROMPT_RESPONSE_SCHEMA.sut_uid: "s",
            },
        ]
    )
    output = FakeAnnotatorOutput()
    p = Pipeline(
        AnnotatorSource(input),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=1),
        EnsembleVoter(FakeEnsemble(["annotator_pydantic", "annotator_dict"])),
        AnnotatorSink(annotators, output, ensemble=True),
        debug=False,
    )
    p.run()

    assert len(output.output) == len(input.items)
    interactions = sorted(list(output.output.keys()), key=lambda o: o.prompt.source_id)
    assert output.output[interactions[0]] == {
        "annotator_pydantic": {"sut_text": "b"},
        "annotator_dict": {"sut_text": "b"},
        "dummy": "d",
        "ensemble": {"ensemble_vote": 1.0},
    }


@pytest.mark.parametrize(
    "sut_worker_count,annotator_worker_count",
    [(1, 1), (2, 2), (8, 8), (1, 5), (5, 1), (3, 9), (9, 3)],
)
def test_prompt_response_annotation_pipeline(annotators, sut_worker_count, annotator_worker_count):
    input = FakePromptInput(
        [
            {PROMPT_RESPONSE_SCHEMA.prompt_uid: "1", PROMPT_RESPONSE_SCHEMA.prompt_text: "a"},
            {PROMPT_RESPONSE_SCHEMA.prompt_uid: "2", PROMPT_RESPONSE_SCHEMA.prompt_text: "b"},
        ]
    )
    output = FakeAnnotatorOutput()

    suts = {"sut1": FakeSUT("sut1"), "sut2": FakeSUT("sut2")}
    p = Pipeline(
        PromptSource(input),
        PromptSutAssigner(suts),
        PromptSutWorkers(suts, workers=sut_worker_count),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=annotator_worker_count),
        AnnotatorSink(annotators, output),
    )
    p.run()

    assert len(output.output) == len(input.items) * len(suts)
    interactions = sorted(list(output.output.keys()), key=lambda o: (o.prompt.source_id, o.sut_uid))
    for interaction, prompt_sut in zip(interactions, itertools.product(input.items, suts)):
        prompt, sut = prompt_sut
        assert sut_interactions_is_equal(
            interaction,
            make_sut_interaction(
                prompt[PROMPT_RESPONSE_SCHEMA.prompt_uid],
                prompt[PROMPT_RESPONSE_SCHEMA.prompt_text],
                sut,
                prompt[PROMPT_RESPONSE_SCHEMA.prompt_text],
            ),
        )
        annotation = {"sut_text": prompt[PROMPT_RESPONSE_SCHEMA.prompt_text]}
        assert output.output[interaction] == {
            "annotator_pydantic": annotation,
            "annotator_dict": annotation,
            "dummy": "d",
        }
