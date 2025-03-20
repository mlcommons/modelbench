import pytest
import re

from modelgauge.annotation_pipeline import (
    AnnotatorAssigner,
    AnnotatorSink,
    AnnotatorSource,
    AnnotatorWorkers,
    CsvAnnotatorInput,
)
from modelgauge.pipeline_runner import AnnotatorRunner, PromptPlusAnnotatorRunner, PromptRunner
from modelgauge.prompt_pipeline import (
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
    PromptSink,
    CsvPromptInput,
    CsvPromptOutput,
)
from modelgauge.sut import SUTOptions
from modelgauge_tests.fake_annotator import FakeAnnotator
from modelgauge_tests.fake_sut import FakeSUT, FakeSUTResponse, FakeSUTRequest


NUM_PROMPTS = 3  # Number of prompts in the prompts file


class AlwaysFailingAnnotator(FakeAnnotator):
    def annotate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        raise Exception("I don't wanna annotate")


class SometimesFailingAnnotator(FakeAnnotator):
    """Fails to annotate on even-numbered requests."""

    def __init__(self, uid):
        super().__init__(uid)
        self.annotate_count = 0

    def annotate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        self.annotate_count += 1
        if self.annotate_count % 2 == 0:
            raise Exception("I don't wanna annotate")
        super().annotate(request)


class AlwaysFailingSUT(FakeSUT):
    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        raise Exception("I don't wanna respond")


class SometimesFailingSUT(FakeSUT):
    """Fails to evaluate on even-numbered requests."""

    def __init__(self, uid):
        super().__init__(uid)
        self.eval_count = 0

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        self.eval_count += 1
        if self.eval_count % 2 == 0:
            raise Exception("I don't wanna respond")
        super().evaluate(request)


@pytest.fixture(scope="session")
def prompts_file(tmp_path_factory):
    """Sample file with 3 prompts for testing."""
    file = tmp_path_factory.mktemp("data") / "prompts.csv"
    with open(file, "w") as f:
        text = "UID,Text\n"
        for i in range(NUM_PROMPTS):
            text += f"p{i},Prompt {i}\n"
        f.write(text)
    return file


@pytest.fixture
def annotators():
    return {
        "annotator1": FakeAnnotator("annotator1"),
        "annotator2": FakeAnnotator("annotator2"),
        "annotator3": FakeAnnotator("annotator3"),
    }


@pytest.fixture
def some_bad_annotators():
    return {
        "good_annotator": FakeAnnotator("good_annotator"),
        "bad_annotator": SometimesFailingAnnotator("bad_annotator"),
        "very_bad_annotator": AlwaysFailingAnnotator("very_bad_annotator"),
    }


@pytest.fixture
def suts():
    return {"sut1": FakeSUT("sut1"), "sut2": FakeSUT("sut2")}


@pytest.fixture
def some_bad_suts():
    return {
        "good_sut": FakeSUT("good_sut"),
        "bad_sut": SometimesFailingSUT("bad_sut"),
        "very_bad_sut": AlwaysFailingSUT("very_bad_sut"),
    }


# Some helper functions to test functionality that is common across runner types
def assert_basic_sut_metadata(metadata):
    """For runs that used the basic suts fixture."""
    assert metadata["suts"] == [
        {
            "uid": "sut1",
            "initialization_record": {
                "args": ["sut1"],
                "class_name": "FakeSUT",
                "kwargs": {},
                "module": "modelgauge_tests.fake_sut",
            },
            "sut_options": {"max_tokens": 100},
        },
        {
            "uid": "sut2",
            "initialization_record": {
                "args": ["sut2"],
                "class_name": "FakeSUT",
                "kwargs": {},
                "module": "modelgauge_tests.fake_sut",
            },
            "sut_options": {"max_tokens": 100},
        },
    ]
    assert metadata["responses"] == {
        "count": 2 * NUM_PROMPTS,  # Num suts * num prompts
        "by_sut": {"sut1": {"count": NUM_PROMPTS}, "sut2": {"count": NUM_PROMPTS}},
    }


def assert_common_metadata_is_correct(metadata, runner):
    assert metadata["run_id"] == runner.run_id
    assert "started" in metadata["run_info"]
    assert "finished" in metadata["run_info"]
    assert "duration" in metadata["run_info"]


def assert_run_completes(runner):
    runner.run(progress_callback=lambda _: _, debug=False)
    output = runner.output_dir() / runner.output_file_name
    assert output.exists()


class TestPromptRunner:
    @pytest.fixture
    def runner_basic(self, tmp_path, prompts_file, suts):
        return PromptRunner(32, prompts_file, tmp_path, None, SUTOptions(), "tag", suts=suts)

    @pytest.fixture
    def runner_some_bad_suts(self, tmp_path, prompts_file, some_bad_suts):
        return PromptRunner(32, prompts_file, tmp_path, None, SUTOptions(), "tag", suts=some_bad_suts)

    @pytest.mark.parametrize(
        "sut_uids,tag,expected_tail",
        [(["s1"], None, "s1"), (["s1", "s2"], None, "s1-s2"), (["s1"], "tag", "tag-s1")],
    )
    def test_run_id(self, tmp_path, prompts_file, sut_uids, tag, expected_tail):
        runner = PromptRunner(
            32, prompts_file, tmp_path, None, SUTOptions(), tag, suts={uid: FakeSUT(uid) for uid in sut_uids}
        )
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic):
        assert runner_basic.output_dir() == tmp_path / runner_basic.run_id

    def test_pipeline_segments(self, tmp_path, prompts_file, suts):
        sut_options = SUTOptions(max_tokens=42)
        runner = PromptRunner(20, prompts_file, tmp_path, None, sut_options, None, suts=suts)
        source, sut_assigner, sut_workers, sink = runner.pipeline_segments

        assert isinstance(source, PromptSource)
        assert isinstance(source.input, CsvPromptInput)
        assert source.input.path == prompts_file

        assert isinstance(sut_assigner, PromptSutAssigner)
        assert sut_assigner.suts == suts

        assert isinstance(sut_workers, PromptSutWorkers)
        assert sut_workers.suts == suts
        assert sut_workers.sut_options == sut_options
        assert sut_workers.thread_count == 20

        assert isinstance(sink, PromptSink)
        assert sink.suts == suts
        assert isinstance(sink.writer, CsvPromptOutput)
        assert sink.writer.suts == suts

    def test_prompt_runner_num_input_items(self, runner_basic):
        assert runner_basic.num_input_items == NUM_PROMPTS

    @pytest.mark.parametrize("num_suts", [1, 2, 5])
    def test_num_total_items(self, tmp_path, prompts_file, num_suts):
        suts = {f"sut{i}": FakeSUT(f"sut{i}") for i in range(num_suts)}
        runner = PromptRunner(20, prompts_file, tmp_path, None, SUTOptions(), None, suts=suts)
        assert runner.num_total_items == NUM_PROMPTS * num_suts

    def test_run_completes(self, runner_basic):
        assert_run_completes(runner_basic)

    def test_run_completes_with_failing_sut(self, runner_some_bad_suts):
        assert_run_completes(runner_some_bad_suts)

    def test_metadata(self, runner_basic, prompts_file):
        runner_basic.run(progress_callback=lambda _: _, debug=False)
        metadata = runner_basic.metadata()

        assert_common_metadata_is_correct(metadata, runner_basic)
        assert metadata["input"] == {"source": prompts_file.name, "num_items": NUM_PROMPTS}
        assert_basic_sut_metadata(metadata)

    # TODO: Add test for metadata with runs that use bad suts.


class TestPromptPlusAnnotatorRunner:
    @pytest.fixture
    def runner_basic(self, tmp_path, prompts_file, suts, annotators):
        return PromptPlusAnnotatorRunner(
            32, prompts_file, tmp_path, None, SUTOptions(), "tag", suts=suts, annotators=annotators
        )

    @pytest.fixture
    def runner_some_bad_suts_and_annotators(self, tmp_path, prompts_file, some_bad_suts, some_bad_annotators):
        return PromptPlusAnnotatorRunner(
            32, prompts_file, tmp_path, None, SUTOptions(), "tag", suts=some_bad_suts, annotators=some_bad_annotators
        )

    @pytest.mark.parametrize(
        "annotator_uids,sut_uids,tag,expected_tail",
        [
            (["a1"], ["s1"], None, "s1-a1"),
            (["a1", "a2"], ["s1", "s2"], None, "s1-s2-a1-a2"),
            (["a1", "a2"], ["s1"], "tag", "tag-s1-a1-a2"),
        ],
    )
    def test_run_id(self, tmp_path, prompts_file, annotator_uids, sut_uids, tag, expected_tail):
        runner = PromptPlusAnnotatorRunner(
            32,
            prompts_file,
            tmp_path,
            None,
            SUTOptions(),
            tag,
            suts={uid: FakeSUT(uid) for uid in sut_uids},
            annotators={uid: FakeAnnotator(uid) for uid in annotator_uids},
        )
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic):
        assert runner_basic.output_dir() == tmp_path / runner_basic.run_id

    def test_pipeline_segments(self, tmp_path, prompts_file, suts, annotators):
        sut_options = SUTOptions(max_tokens=42)
        runner = PromptPlusAnnotatorRunner(
            20, prompts_file, tmp_path, None, sut_options, None, suts=suts, annotators=annotators
        )
        source, sut_assigner, sut_workers, annotator_assigner, annotator_workers, sink = runner.pipeline_segments

        assert isinstance(source, PromptSource)
        assert isinstance(source.input, CsvPromptInput)
        assert source.input.path == prompts_file

        assert isinstance(sut_assigner, PromptSutAssigner)
        assert sut_assigner.suts == suts

        assert isinstance(sut_workers, PromptSutWorkers)
        assert sut_workers.suts == suts
        assert sut_workers.sut_options == sut_options
        assert sut_workers.thread_count == 20

        assert isinstance(annotator_assigner, AnnotatorAssigner)
        assert annotator_assigner.annotators == annotators

        assert isinstance(annotator_workers, AnnotatorWorkers)
        assert annotator_workers.annotators == annotators
        assert annotator_workers.thread_count == 20

        assert isinstance(sink, AnnotatorSink)
        assert sink.annotators == annotators

    def test_prompt_runner_num_input_items(self, runner_basic):
        assert runner_basic.num_input_items == NUM_PROMPTS

    @pytest.mark.parametrize("num_suts,num_annotators", [(1, 1), (1, 3), (3, 1), (3, 3)])
    def test_num_total_items(self, tmp_path, prompts_file, num_suts, num_annotators):
        suts = {f"sut{i}": FakeSUT(f"sut{i}") for i in range(num_suts)}
        annotators = {f"annotator{i}": FakeAnnotator(f"annotator{i}") for i in range(num_annotators)}
        runner = PromptPlusAnnotatorRunner(
            20, prompts_file, tmp_path, None, SUTOptions(), None, suts=suts, annotators=annotators
        )
        assert runner.num_total_items == NUM_PROMPTS * num_suts * num_annotators

    def test_run_completes(self, runner_basic):
        assert_run_completes(runner_basic)

    def test_run_completes_with_failing_suts_and_annotators(self, runner_some_bad_suts_and_annotators):
        assert_run_completes(runner_some_bad_suts_and_annotators)

    def test_metadata(self, runner_basic, prompts_file, suts, annotators):
        runner_basic.run(progress_callback=lambda _: _, debug=False)
        metadata = runner_basic.metadata()

        assert_common_metadata_is_correct(metadata, runner_basic)
        assert metadata["input"] == {"source": prompts_file.name, "num_items": NUM_PROMPTS}
        assert_basic_sut_metadata(metadata)
        assert metadata["annotators"] == [{"uid": "annotator1"}, {"uid": "annotator2"}, {"uid": "annotator3"}]
        assert metadata["annotations"] == {
            "count": NUM_PROMPTS * len(suts) * len(annotators),
            "by_annotator": {
                "annotator1": {"count": NUM_PROMPTS * len(suts)},
                "annotator2": {"count": NUM_PROMPTS * len(suts)},
                "annotator3": {"count": NUM_PROMPTS * len(suts)},
            },
        }

    # TODO: Add test for metadata with runs that use bad suts and annotators.


class TestAnnotatorRunner:
    NUM_SUTS = 2  # Number of SUTs included in the input prompts_response_file

    @pytest.fixture(scope="session")
    def prompt_responses_file(self, tmp_path_factory):
        """Sample file with 2 prompts + responses from 2 SUTs for testing."""
        file = tmp_path_factory.mktemp("data") / "prompt-responses.csv"
        with open(file, "w") as f:
            text = "UID,Prompt,SUT,Response\n"
            for i in range(NUM_PROMPTS):
                text += f"p{i},Prompt {i},sut1,Response {i}\n"
                text += f"p{i},Prompt {i},sut2,Response {i}\n"
            f.write(text)
        return file

    @pytest.fixture
    def runner_basic(self, tmp_path, prompt_responses_file, annotators):
        return AnnotatorRunner(32, prompt_responses_file, tmp_path, None, None, "tag", annotators=annotators)

    @pytest.fixture
    def runner_some_bad_annotators(self, tmp_path, prompt_responses_file, some_bad_annotators):
        return AnnotatorRunner(32, prompt_responses_file, tmp_path, None, None, "tag", annotators=some_bad_annotators)

    @pytest.mark.parametrize(
        "annotator_uids,tag,expected_tail",
        [
            (["a1"], None, "a1"),
            (["a1", "a2"], None, "a1-a2"),
            (["a1", "a2"], "tag", "tag-a1-a2"),
        ],
    )
    def test_run_id(self, tmp_path, prompt_responses_file, annotator_uids, tag, expected_tail):
        runner = AnnotatorRunner(
            32,
            prompt_responses_file,
            tmp_path,
            None,
            None,
            tag,
            annotators={uid: FakeAnnotator(uid) for uid in annotator_uids},
        )
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic):
        assert runner_basic.output_dir() == tmp_path / runner_basic.run_id

    def test_pipeline_segments(self, tmp_path, prompt_responses_file, annotators):
        runner = AnnotatorRunner(20, prompt_responses_file, tmp_path, None, None, None, annotators=annotators)
        source, annotator_assigner, annotator_workers, sink = runner.pipeline_segments

        assert isinstance(source, AnnotatorSource)
        assert isinstance(source.input, CsvAnnotatorInput)
        assert source.input.path == prompt_responses_file

        assert isinstance(annotator_assigner, AnnotatorAssigner)
        assert annotator_assigner.annotators == annotators

        assert isinstance(annotator_workers, AnnotatorWorkers)
        assert annotator_workers.annotators == annotators
        assert annotator_workers.thread_count == 20

        assert isinstance(sink, AnnotatorSink)
        assert sink.annotators == annotators

    def test_prompt_runner_num_input_items(self, runner_basic):
        assert runner_basic.num_input_items == NUM_PROMPTS * self.NUM_SUTS

    @pytest.mark.parametrize("num_annotators", [1, 2, 5])
    def test_num_total_items(self, tmp_path, prompt_responses_file, num_annotators):
        annotators = {f"annotator{i}": FakeAnnotator(f"annotator{i}") for i in range(num_annotators)}
        runner = AnnotatorRunner(20, prompt_responses_file, tmp_path, None, None, None, annotators=annotators)
        assert runner.num_total_items == NUM_PROMPTS * self.NUM_SUTS * num_annotators

    def test_run_completes(self, runner_basic):
        assert_run_completes(runner_basic)

    def test_run_completes_with_annotators(self, runner_some_bad_annotators):
        assert_run_completes(runner_some_bad_annotators)

    def test_metadata(self, runner_basic, prompt_responses_file):
        runner_basic.run(progress_callback=lambda _: _, debug=False)
        metadata = runner_basic.metadata()

        assert_common_metadata_is_correct(metadata, runner_basic)
        assert metadata["input"] == {"source": prompt_responses_file.name, "num_items": NUM_PROMPTS * self.NUM_SUTS}
        assert "suts" not in metadata
        assert metadata["annotators"] == [{"uid": "annotator1"}, {"uid": "annotator2"}, {"uid": "annotator3"}]
        assert metadata["annotations"] == {
            "count": NUM_PROMPTS * self.NUM_SUTS * 3,
            "by_annotator": {
                "annotator1": {"count": NUM_PROMPTS * self.NUM_SUTS},
                "annotator2": {"count": NUM_PROMPTS * self.NUM_SUTS},
                "annotator3": {"count": NUM_PROMPTS * self.NUM_SUTS},
            },
        }

    # TODO: Add test for metadata with runs that use bad annotators.
