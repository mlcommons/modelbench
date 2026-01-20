import itertools
import json
import pytest
import time
from unittest.mock import MagicMock

from modelgauge.annotation import SafetyAnnotation
from modelgauge.annotation_pipeline import (
    AnnotatorSource,
    AnnotatorAssigner,
    AnnotatorWorkers,
    AnnotatorSink,
)
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.dataset import AnnotationDataset, PromptResponseDataset
from modelgauge.data_schema import PromptResponseSchema
from modelgauge.ensemble_annotator import EnsembleAnnotator
from modelgauge.ensemble_strategies import ENSEMBLE_STRATEGIES
from modelgauge.pipeline import Pipeline
from modelgauge.prompt import TextPrompt
from modelgauge.prompt_pipeline import (
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
)
from modelgauge.single_turn_prompt_response import AnnotatedSUTInteraction, SUTInteraction, TestItem
from modelgauge.sut import SUTResponse
from modelgauge_tests.fake_annotator import (
    FakeAnnotatorRequest,
    FakeSafetyAnnotator,
)
from modelgauge_tests.fake_sut import FakeSUT
from modelgauge_tests.fake_ensemble_strategy import FakeEnsembleStrategy
from modelgauge_tests.test_prompt_pipeline import FakePromptInput


PROMPT_RESPONSE_SCHEMA = PromptResponseSchema.default()


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


class FakeAnnotatorOutput(AnnotationDataset):
    def __init__(self, path: str):
        self.output = []
        super().__init__(path, "w")

    def write(self, item):
        self.output.append(self.item_to_row(item))


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


@pytest.fixture
def annotators():
    annotator_pydantic = FakeSafetyAnnotator("annotator_pydantic")
    annotator_dict = FakeSafetyAnnotator("annotator_dict")
    # Return the same annotation but as a dict.
    annotator_dict.translate_response = MagicMock(
        side_effect=lambda *args: annotator_pydantic.translate_response(*args).model_dump(exclude_none=True)
    )
    annotator_dummy = FakeSafetyAnnotator("dummy")
    annotator_dummy.translate_response = MagicMock(return_value="d")
    return {"annotator_pydantic": annotator_pydantic, "annotator_dict": annotator_dict, "dummy": annotator_dummy}


@pytest.fixture
def ensemble_annotator():
    ENSEMBLE_STRATEGIES["fake"] = FakeEnsembleStrategy()

    ANNOTATORS.register(FakeSafetyAnnotator, "annotator_safety")
    annotator_ensemble = EnsembleAnnotator("annotator_ensemble", ["annotator_safety"], "fake")
    return {
        "annotator_ensemble": annotator_ensemble,
    }


@pytest.mark.parametrize(
    "annotator_uid,annotation",
    [
        ("annotator_pydantic", SafetyAnnotation(is_safe=True)),
        ("annotator_dict", {"is_safe": True, "is_valid": True}),
        ("dummy", "d"),
    ],
)
def test_annotator_worker_normal(annotators, annotator_uid, annotation):
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    w = AnnotatorWorkers(annotators)
    result = w.handle_item((sut_interaction, annotator_uid))
    assert isinstance(result, AnnotatedSUTInteraction)
    assert result.sut_interaction == sut_interaction
    assert result.annotator_uid == annotator_uid
    assert result.annotation == annotation


def test_annotator_worker_cache_simple(annotators, tmp_path):
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    w = AnnotatorWorkers(annotators, cache_path=tmp_path)

    # Tests that first call invokes the annotator and the second call uses the cache.
    assert annotators["annotator_pydantic"].annotate_calls == 0
    for _ in range(2):
        result = w.handle_item((sut_interaction, "annotator_pydantic"))
        assert result.annotation == SafetyAnnotation(is_safe=True)
        assert annotators["annotator_pydantic"].annotate_calls == 1


def test_annotator_worker_cache_simple_ensemble(ensemble_annotator, tmp_path):
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    w = AnnotatorWorkers(ensemble_annotator, cache_path=tmp_path)
    # Tests that first call invokes the annotator and the second call uses the cache.
    inner_annotator = ensemble_annotator["annotator_ensemble"].annotators["annotator_safety"]

    assert inner_annotator.annotate_calls == 0
    _ = w.handle_item((sut_interaction, "annotator_ensemble"))
    assert inner_annotator.annotate_calls == 1
    _ = w.handle_item((sut_interaction, "annotator_ensemble"))
    assert inner_annotator.annotate_calls == 1  # Still 1, so second call used the cache.


def test_annotator_worker_unique_responses(annotators, tmp_path):
    """Different responses have different cache keys for annotator with response-based requests."""
    w = AnnotatorWorkers(annotators, cache_path=tmp_path)

    assert annotators["annotator_pydantic"].annotate_calls == 0
    w.handle_item((make_sut_interaction("", "", "", "response 1"), "annotator_pydantic"))
    assert annotators["annotator_pydantic"].annotate_calls == 1
    w.handle_item((make_sut_interaction("", "", "", "response 2"), "annotator_pydantic"))
    assert annotators["annotator_pydantic"].annotate_calls == 2

    # New prompt id does affect the cache key.
    w.handle_item((make_sut_interaction("2", "2", "2", "response 2"), "annotator_pydantic"))
    assert annotators["annotator_pydantic"].annotate_calls == 3


def test_annotator_worker_cache_unique_prompts(tmp_path):
    """Different prompts have different cache keys for annotator with prompt-based requests."""

    annotator = FakeSafetyAnnotator("a")
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
    mock.side_effect = exceptions + [SafetyAnnotation(is_safe=True)]
    annotator = FakeSafetyAnnotator("fake-annotator")
    annotator.annotate = mock

    w = AnnotatorWorkers({"fake-annotator": annotator})
    w.sleep_time = 0.01
    sut_interaction = make_sut_interaction("1", "prompt", "sut", "response")
    result = w.handle_item((sut_interaction, "fake-annotator"))

    assert mock.call_count == num_exceptions + 1
    assert result.sut_interaction == sut_interaction
    assert result.annotator_uid == "fake-annotator"
    assert result.annotation == SafetyAnnotation(is_safe=True)


def test_full_run(annotators, tmp_path):
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
    output = FakeAnnotatorOutput(tmp_path / "output.csv")
    p = Pipeline(
        AnnotatorSource(input),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=1),
        AnnotatorSink(output),
    )
    p.run()

    assert len(output.output) == len(input.items) * len(annotators)
    items = sorted(output.output, key=lambda o: (o[0], o[4]))  # Sort by prompt_uid, annotator_uid

    # First 3 items are same sut interaction
    assert sut_interactions_is_equal(make_sut_interaction(*items[0][:4]), make_sut_interaction("1", "a", "s", "b"))
    assert sut_interactions_is_equal(make_sut_interaction(*items[1][:4]), make_sut_interaction("1", "a", "s", "b"))
    assert sut_interactions_is_equal(make_sut_interaction(*items[2][:4]), make_sut_interaction("1", "a", "s", "b"))
    assert items[0][4] == "annotator_dict"
    assert items[1][4] == "annotator_pydantic"
    assert items[2][4] == "dummy"
    assert items[0][5] == '{"is_safe": true, "is_valid": true}'
    assert items[1][5] == '{"is_safe": true, "is_valid": true}'
    assert items[2][5] == '"d"'

    # Second 3 items are same sut interaction
    assert sut_interactions_is_equal(make_sut_interaction(*items[3][:4]), make_sut_interaction("2", "c", "s", "d"))
    assert sut_interactions_is_equal(make_sut_interaction(*items[4][:4]), make_sut_interaction("2", "c", "s", "d"))
    assert sut_interactions_is_equal(make_sut_interaction(*items[5][:4]), make_sut_interaction("2", "c", "s", "d"))
    assert items[3][4] == "annotator_dict"
    assert items[4][4] == "annotator_pydantic"
    assert items[5][4] == "dummy"
    assert items[3][5] == '{"is_safe": true, "is_valid": true}'
    assert items[4][5] == '{"is_safe": true, "is_valid": true}'
    assert items[5][5] == '"d"'


@pytest.mark.parametrize(
    "sut_worker_count,annotator_worker_count",
    [(1, 1), (2, 2), (8, 8), (1, 5), (5, 1)],
)
def test_prompt_response_annotation_pipeline(annotators, sut_worker_count, annotator_worker_count, tmp_path):
    input = FakePromptInput(
        [
            {PROMPT_RESPONSE_SCHEMA.prompt_uid: "1", PROMPT_RESPONSE_SCHEMA.prompt_text: "a"},
            {PROMPT_RESPONSE_SCHEMA.prompt_uid: "2", PROMPT_RESPONSE_SCHEMA.prompt_text: "b"},
        ]
    )
    output = FakeAnnotatorOutput(tmp_path / "output.csv")

    suts = {"sut1": FakeSUT("sut1"), "sut2": FakeSUT("sut2")}
    p = Pipeline(
        PromptSource(input),
        PromptSutAssigner(suts),
        PromptSutWorkers(suts, workers=sut_worker_count),
        AnnotatorAssigner(annotators),
        AnnotatorWorkers(annotators, workers=annotator_worker_count),
        AnnotatorSink(output),
    )
    p.run()

    assert len(output.output) == len(input.items) * len(suts) * len(annotators)

    rows = sorted(output.output, key=lambda row: (row[0], row[2], row[4]))  # Sort by prompt_uid, sut_uid, annotator_uid

    # Group rows by prompt and sut
    current_idx = 0
    for prompt in input.items:
        for sut in suts:
            # For each prompt-sut combination, we should have one row per annotator
            for annotator_name in ["annotator_dict", "annotator_pydantic", "dummy"]:
                row = rows[current_idx]
                # Check prompt fields
                assert row[0] == prompt[PROMPT_RESPONSE_SCHEMA.prompt_uid]  # prompt_uid
                assert row[1] == prompt[PROMPT_RESPONSE_SCHEMA.prompt_text]  # prompt_text
                # Check SUT fields
                assert row[2] == sut  # sut_uid
                assert row[3] == prompt[PROMPT_RESPONSE_SCHEMA.prompt_text]  # sut_response (FakeSUT echoes prompt)
                # Check annotator fields
                assert row[4] == annotator_name  # annotator_uid
                # Check annotation content
                if annotator_name == "dummy":
                    assert row[5] == '"d"'  # dummy annotator returns "d"
                else:
                    # Both dict and pydantic annotators return the same structure
                    expected_annotation = {"is_safe": True, "is_valid": True}
                    assert row[5] == json.dumps(expected_annotation)
                current_idx += 1
