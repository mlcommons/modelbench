import itertools
import jsonlines
import pytest
import time
from unittest.mock import MagicMock

from modelgauge.annotation_pipeline import (
    SutInteraction,
    AnnotatorInput,
    AnnotatorSource,
    AnnotatorAssigner,
    AnnotatorWorkers,
    AnnotatorSink,
    CsvAnnotatorInput,
    JsonlAnnotatorOutput,
)
from modelgauge.pipeline import Pipeline
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import (
    PromptOutput,
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
)
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from tests.fake_annotator import (
    FakeAnnotation,
    FakeAnnotator,
)
from tests.fake_sut import FakeSUT
from tests.test_prompt_pipeline import FakePromptInput


class FakeAnnotatorInput(AnnotatorInput):
    def __init__(self, items: list[dict], delay=None):
        super().__init__()
        self.items = items
        self.delay = itertools.cycle(delay or [0])

    def __iter__(self):
        for row in self.items:
            time.sleep(next(self.delay))
            prompt = PromptWithContext(
                prompt=TextPrompt(text=row["Prompt"]),
                source_id=row["UID"],
                context=row,
            )
            response = SUTCompletion(text=row["Response"])
            yield SutInteraction(prompt, row["SUT"], response)


class FakeAnnotatorOutput(PromptOutput):
    def __init__(self):
        self.output = {}

    def write(self, item, annotations):
        self.output[item] = annotations


def make_sut_interaction(source_id, prompt, sut_uid, response):
    return SutInteraction(
        PromptWithContext(source_id=source_id, prompt=TextPrompt(text=prompt)),
        sut_uid,
        SUTCompletion(text=response),
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
    file_path.write_text('UID,Prompt,SUT,Response\n"1","a","s","b"')
    input = CsvAnnotatorInput(file_path)

    assert len(input) == 1
    item: SutInteraction = next(iter(input))
    assert sut_interactions_is_equal(item, make_sut_interaction("1", "a", "s", "b"))


@pytest.mark.parametrize(
    "header",
    [
        "Prompt,UID,Extra,Response,Response\n",
        "UID,Prompt,SUT\n",
        "Extra,Response,Extra\n",
    ],
)
def test_csv_annotator_input_invalid_columns(tmp_path, header):
    file_path = tmp_path / "input.csv"
    file_path.write_text(header)
    with pytest.raises(
        AssertionError,
        match="Invalid input file. Must have columns: UID, Prompt, SUT, Response.",
    ):
        CsvAnnotatorInput(file_path)


def test_json_annotator_output(tmp_path):
    file_path = tmp_path / "output.jsonl"
    with JsonlAnnotatorOutput(file_path) as output:
        output.write(make_sut_interaction("1", "a", "sut1", "b"), {"fake": "x"})
        output.write(make_sut_interaction("2", "c", "sut2", "d"), {"fake": "y"})

    with jsonlines.open(file_path) as reader:
        items: list[dict] = [i for i in reader]
        assert len(items) == 2
        assert items[0] == {
            "UID": "1",
            "Prompt": "a",
            "SUT": "sut1",
            "Response": "b",
            "Annotations": {"fake": "x"},
        }
        assert items[1] == {
            "UID": "2",
            "Prompt": "c",
            "SUT": "sut2",
            "Response": "d",
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
    with pytest.raises(
        AssertionError, match=f"Invalid output file {file_path}. Must be of type JSONL."
    ):
        JsonlAnnotatorOutput(file_path)


@pytest.fixture
def annotators():
    annotator_pydantic = FakeAnnotator()
    annotator_dict = FakeAnnotator()
    # Return the same annotation but as a dict.
    annotator_dict.translate_response = MagicMock(
        side_effect=lambda *args: annotator_pydantic.translate_response(
            *args
        ).model_dump()
    )
    annotator_dummy = FakeAnnotator()
    annotator_dummy.translate_response = MagicMock(return_value="d")
    return {
        "annotator_pydantic": annotator_pydantic,
        "annotator_dict": annotator_dict,
        "dummy": annotator_dummy,
    }


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
    w.handle_item(
        (make_sut_interaction("", "", "", "response 1"), "annotator_pydantic")
    )
    assert annotators["annotator_pydantic"].annotate_calls == 1
    w.handle_item(
        (make_sut_interaction("", "", "", "response 2"), "annotator_pydantic")
    )
    assert annotators["annotator_pydantic"].annotate_calls == 2

    # Non-response SUT interaction attributes do not affect the cache key.
    w.handle_item(
        (make_sut_interaction("2", "2", "2", "response 2"), "annotator_pydantic")
    )
    assert annotators["annotator_pydantic"].annotate_calls == 2


def test_annotator_worker_cache_unique_prompts(tmp_path):
    """Different prompts have different cache keys for annotator with prompt-based requests."""

    annotator = FakeAnnotator()
    annotator.translate_request = MagicMock(
        side_effect=lambda prompt, response: {"prompt": prompt, "text": response}
    )
    w = AnnotatorWorkers({"a": annotator}, cache_path=tmp_path)

    # Different prompt texts.
    assert annotator.annotate_calls == 0
    w.handle_item((make_sut_interaction("", "prompt 1", "", ""), "a"))
    assert annotator.annotate_calls == 1
    w.handle_item((make_sut_interaction("", "prompt 2", "", ""), "a"))
    assert annotator.annotate_calls == 2

    # Different SUT options for same prompt text.
    sut_interaction = make_sut_interaction("", "prompt 1", "", "")
    sut_interaction.prompt.prompt.options.max_tokens += 1
    w.handle_item((sut_interaction, "a"))
    assert annotator.annotate_calls == 3


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


def test_full_run(annotators):
    input = FakeAnnotatorInput(
        [
            {"UID": "1", "Prompt": "a", "Response": "b", "SUT": "s"},
            {"UID": "2", "Prompt": "c", "Response": "d", "SUT": "s"},
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
    assert sut_interactions_is_equal(
        interactions[0], make_sut_interaction("1", "a", "s", "b")
    )
    assert output.output[interactions[0]] == {
        "annotator_pydantic": {"sut_text": "b"},
        "annotator_dict": {"sut_text": "b"},
        "dummy": "d",
    }
    assert sut_interactions_is_equal(
        interactions[1], make_sut_interaction("2", "c", "s", "d")
    )
    assert output.output[interactions[1]] == {
        "annotator_pydantic": {"sut_text": "d"},
        "annotator_dict": {"sut_text": "d"},
        "dummy": "d",
    }


@pytest.mark.parametrize(
    "sut_worker_count,annotator_worker_count",
    [(1, 1), (2, 2), (8, 8), (1, 5), (5, 1), (3, 9), (9, 3)],
)
def test_prompt_response_annotation_pipeline(
    annotators, sut_worker_count, annotator_worker_count
):
    input = FakePromptInput(
        [
            {"UID": "1", "Text": "a"},
            {"UID": "2", "Text": "b"},
        ]
    )
    output = FakeAnnotatorOutput()

    suts = {"sut1": FakeSUT(), "sut2": FakeSUT()}
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
    interactions = sorted(
        list(output.output.keys()), key=lambda o: (o.prompt.source_id, o.sut_uid)
    )
    for interaction, prompt_sut in zip(
        interactions, itertools.product(input.items, suts)
    ):
        prompt, sut = prompt_sut
        assert sut_interactions_is_equal(
            interaction,
            make_sut_interaction(prompt["UID"], prompt["Text"], sut, prompt["Text"]),
        )
        annotation = {"sut_text": prompt["Text"]}
        assert output.output[interaction] == {
            "annotator_pydantic": annotation,
            "annotator_dict": annotation,
            "dummy": "d",
        }
