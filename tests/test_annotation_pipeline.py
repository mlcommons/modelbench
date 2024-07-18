import itertools
import jsonlines
import pytest
import time

from modelgauge.annotation_pipeline import (
    AnnotatorInputSample,
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
from modelgauge.prompt_pipeline import PromptOutput
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from tests.fake_annotator import (
    FakeAnnotation,
    FakeAnnotator,
)


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
            yield AnnotatorInputSample(prompt, row["SUT"], response)


class FakeAnnotatorOutput(PromptOutput):
    def __init__(self):
        self.output = []

    def write(self, item, annotations):
        self.output.append((item, annotations))


@pytest.fixture
def annotators():
    return {"fake1": FakeAnnotator(), "fake2": FakeAnnotator()}


def make_input_sample(source_id, prompt, sut_uid, response):
    return AnnotatorInputSample(
        PromptWithContext(source_id=source_id, prompt=TextPrompt(text=prompt)),
        sut_uid,
        SUTCompletion(text=response),
    )


def test_csv_annotator_input(tmp_path):
    file_path = tmp_path / "input.csv"
    file_path.write_text('UID,Prompt,SUT,Response\n"1","a","sut_uid","b"')
    input = CsvAnnotatorInput(file_path)

    assert len(input) == 1
    item: AnnotatorInputSample = next(iter(input))
    assert item.prompt.prompt == TextPrompt(text="a")
    assert item.prompt.source_id == "1"
    assert item.sut_uid == "sut_uid"
    assert item.response == SUTCompletion(text="b")


def test_json_annotator_output(tmp_path, annotators):
    file_path = tmp_path / "output.jsonl"
    input_sample1 = make_input_sample("1", "a", "sut_uid1", "b")
    input_sample2 = make_input_sample("2", "c", "sut_uid2", "d")
    with JsonlAnnotatorOutput(file_path, annotators) as output:
        output.write(input_sample1, {"fake1": "a1", "fake2": "a2"})
        output.write(input_sample2, {"fake1": "a3", "fake2": "a4"})

    with jsonlines.open(file_path) as reader:
        items: list[dict] = [i for i in reader]
        assert len(items) == 2
        assert items[0] == {
            "UID": "1",
            "Prompt": "a",
            "SUT": "sut_uid1",
            "Response": "b",
            "Annotations": {"fake1": "a1", "fake2": "a2"},
        }
        assert items[1] == {
            "UID": "2",
            "Prompt": "c",
            "SUT": "sut_uid2",
            "Response": "d",
            "Annotations": {"fake1": "a3", "fake2": "a4"},
        }


def test_json_annotator_output_dict_annotation(tmp_path, annotators):
    file_path = tmp_path / "output.jsonl"

    with JsonlAnnotatorOutput(file_path, annotators) as output:
        output.write(
            make_input_sample("1", "a", "sut_uid1", "b"),
            {
                "fake1": FakeAnnotation(sut_text="a1").model_dump(),
                "fake2": FakeAnnotation(sut_text="a2").model_dump(),
            },
        )
    with jsonlines.open(file_path) as reader:
        items: list[dict] = [i for i in reader]
        assert items[0]["Annotations"] == {
            "fake1": {"sut_text": "a1"},
            "fake2": {"sut_text": "a2"},
        }


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
    assert sorted([r[0].prompt.source_id for r in output.output]) == [
        i["UID"] for i in input.items
    ]
    assert sorted([r[0].response.text for r in output.output]) == [
        i["Response"] for i in input.items
    ]
    row1 = output.output[0]
    assert "fake1" in row1[1]
    assert "fake2" in row1[1]
    row2 = output.output[1]
    assert "fake1" in row2[1]
    assert "fake2" in row2[1]


#
def test_progress(annotators):
    input = FakeAnnotatorInput(
        [
            {"UID": "1", "Prompt": "a", "Response": "b", "SUT": "s"},
            {"UID": "2", "Prompt": "c", "Response": "d", "SUT": "s"},
        ]
    )
    output = FakeAnnotatorOutput()

    def track_progress(data):
        progress_items.append(data.copy())

    p = Pipeline(
        AnnotatorSource(input),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=1),
        AnnotatorSink(annotators, output),
        progress_callback=track_progress,
    )
    progress_items = []

    p.run()

    assert progress_items[0]["completed"] == 0
    assert progress_items[-1]["completed"] == 4
