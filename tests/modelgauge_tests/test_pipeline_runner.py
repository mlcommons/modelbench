import pytest
import re

from modelgauge.annotation_pipeline import (
    AnnotatorAssigner,
    AnnotatorSink,
    AnnotatorSource,
    AnnotatorWorkers,
    CsvAnnotatorInput,
)
from modelgauge.annotator_set import AnnotatorSet
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
from modelgauge_tests.fake_sut import FakeSUT


NUM_PROMPTS = 3  # Number of prompts in the prompts file


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
def ensemble():
    class FakeEnsemble(AnnotatorSet):
        annotators = ["annotator1", "annotator2", "annotator3"]

        def evaluate(self, item):
            return {"ensemble_vote": 1.0}

    return FakeEnsemble()


@pytest.fixture
def suts():
    return {"sut1": FakeSUT("sut1"), "sut2": FakeSUT("sut2")}


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
        return PromptRunner(suts, 32, prompts_file, tmp_path)

    @pytest.mark.parametrize(
        "sut_uids,tag,expected_tail",
        [(["s1"], None, "s1"), (["s1", "s2"], None, "s1-s2"), (["s1"], "tag", "tag-s1")],
    )
    def test_run_id(self, tmp_path, prompts_file, sut_uids, tag, expected_tail):
        suts = {uid: FakeSUT(uid) for uid in sut_uids}
        runner = PromptRunner(suts, 32, prompts_file, tmp_path, tag=tag)
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic):
        assert runner_basic.output_dir() == tmp_path / runner_basic.run_id

    def test_pipeline_segments(self, tmp_path, prompts_file, suts):
        sut_options = SUTOptions(max_tokens=42)
        runner = PromptRunner(suts, 20, prompts_file, tmp_path, sut_options=sut_options)
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
        runner = PromptRunner(suts, 20, prompts_file, tmp_path)
        assert runner.num_total_items == NUM_PROMPTS * num_suts

    def test_run_completes(self, runner_basic):
        assert_run_completes(runner_basic)

    def test_metadata(self, runner_basic, prompts_file):
        runner_basic.run(progress_callback=lambda _: _, debug=False)
        metadata = runner_basic.metadata()

        assert_common_metadata_is_correct(metadata, runner_basic)
        assert metadata["input"] == {"source": prompts_file.name, "num_items": NUM_PROMPTS}
        assert_basic_sut_metadata(metadata)


class TestPromptPlusAnnotatorRunner:
    @pytest.fixture
    def runner_basic(self, tmp_path, prompts_file, suts, annotators):
        return PromptPlusAnnotatorRunner(suts, annotators, None, 32, prompts_file, tmp_path)

    @pytest.fixture
    def runner_ensemble(self, tmp_path, prompts_file, suts, annotators, ensemble):
        return PromptPlusAnnotatorRunner(suts, annotators, ensemble, 32, prompts_file, tmp_path)

    @pytest.mark.parametrize(
        "annotator_uids,sut_uids,tag,expected_tail",
        [
            (["a1"], ["s1"], None, "s1-a1"),
            (["a1", "a2"], ["s1", "s2"], None, "s1-s2-a1-a2"),
            (["a1", "a2"], ["s1"], "tag", "tag-s1-a1-a2"),
        ],
    )
    def test_run_id(self, tmp_path, prompts_file, annotator_uids, sut_uids, tag, expected_tail):
        suts = {uid: FakeSUT(uid) for uid in sut_uids}
        annotators = {uid: FakeAnnotator(uid) for uid in annotator_uids}
        runner = PromptPlusAnnotatorRunner(suts, annotators, None, 32, prompts_file, tmp_path, tag=tag)
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic):
        assert runner_basic.output_dir() == tmp_path / runner_basic.run_id

    def test_pipeline_segments(self, tmp_path, prompts_file, suts, annotators):
        sut_options = SUTOptions(max_tokens=42)
        runner = PromptPlusAnnotatorRunner(suts, annotators, None, 20, prompts_file, tmp_path, sut_options=sut_options)
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
        assert sink.ensemble == False

    def test_pipeline_segments_ensemble(self, runner_ensemble, annotators, ensemble):
        source, sut_assigner, sut_workers, annotator_assigner, annotator_workers, ensemble_worker, sink = (
            runner_ensemble.pipeline_segments
        )

        assert isinstance(annotator_workers, AnnotatorWorkers)
        assert annotator_workers.annotators == annotators

        assert ensemble_worker.ensemble == ensemble

        assert isinstance(sink, AnnotatorSink)
        assert sink.annotators == annotators
        assert sink.ensemble == True

    def test_runner_num_input_items(self, runner_basic):
        assert runner_basic.num_input_items == NUM_PROMPTS

    @pytest.mark.parametrize("num_suts,num_annotators", [(1, 1), (1, 3), (3, 1), (3, 3)])
    def test_num_total_items(self, tmp_path, prompts_file, num_suts, num_annotators):
        suts = {f"sut{i}": FakeSUT(f"sut{i}") for i in range(num_suts)}
        annotators = {f"annotator{i}": FakeAnnotator(f"annotator{i}") for i in range(num_annotators)}
        runner = PromptPlusAnnotatorRunner(suts, annotators, None, 20, prompts_file, tmp_path)
        assert runner.num_total_items == NUM_PROMPTS * num_suts * num_annotators

    def test_num_total_items_ensemble(self, runner_ensemble, suts):
        assert runner_ensemble.num_total_items == NUM_PROMPTS * len(suts) * len(runner_ensemble.annotators)

    def test_run_completes(self, runner_basic):
        assert_run_completes(runner_basic)

    def test_run_ensemble_completes(self, runner_ensemble):
        assert_run_completes(runner_ensemble)

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
        return AnnotatorRunner(annotators, None, 32, prompt_responses_file, tmp_path)

    @pytest.fixture
    def runner_ensemble(self, tmp_path, prompt_responses_file, annotators, ensemble):
        return AnnotatorRunner(annotators, ensemble, 32, prompt_responses_file, tmp_path)

    @pytest.mark.parametrize(
        "annotator_uids,tag,expected_tail",
        [
            (["a1"], None, "a1"),
            (["a1", "a2"], None, "a1-a2"),
            (["a1", "a2"], "tag", "tag-a1-a2"),
        ],
    )
    def test_run_id(self, tmp_path, prompt_responses_file, annotator_uids, tag, expected_tail):
        annotators = {uid: FakeAnnotator(uid) for uid in annotator_uids}
        runner = AnnotatorRunner(annotators, None, 32, prompt_responses_file, tmp_path, tag=tag)
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic):
        assert runner_basic.output_dir() == tmp_path / runner_basic.run_id

    def test_pipeline_segments(self, tmp_path, prompt_responses_file, annotators):
        runner = AnnotatorRunner(annotators, None, 20, prompt_responses_file, tmp_path)
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
        assert sink.ensemble == False

    def test_pipeline_segments_ensemble(self, runner_ensemble, annotators, ensemble):
        source, annotator_assigner, annotator_workers, ensemble_worker, sink = runner_ensemble.pipeline_segments

        assert isinstance(annotator_workers, AnnotatorWorkers)
        assert annotator_workers.annotators == annotators

        assert ensemble_worker.ensemble == ensemble

        assert isinstance(sink, AnnotatorSink)
        assert sink.annotators == annotators
        assert sink.ensemble == True

    def test_missing_ensemble_annotators_raises_error(self, tmp_path, prompt_responses_file, ensemble):
        incomplete_annotators = {"annotator1": FakeAnnotator("annotator1"), "annotator2": FakeAnnotator("annotator2")}
        with pytest.raises(ValueError, match="Ensemble annotators {'annotator3'} not found"):
            AnnotatorRunner(incomplete_annotators, ensemble, 20, prompt_responses_file, tmp_path)

    def test_runner_num_input_items(self, runner_basic):
        assert runner_basic.num_input_items == NUM_PROMPTS * self.NUM_SUTS

    @pytest.mark.parametrize("num_annotators", [1, 2, 5])
    def test_num_total_items(self, tmp_path, prompt_responses_file, num_annotators):
        annotators = {f"annotator{i}": FakeAnnotator(f"annotator{i}") for i in range(num_annotators)}
        runner = AnnotatorRunner(annotators, None, 20, prompt_responses_file, tmp_path)
        assert runner.num_total_items == NUM_PROMPTS * self.NUM_SUTS * num_annotators

    def test_num_total_items_ensemble(self, runner_ensemble):
        assert runner_ensemble.num_total_items == NUM_PROMPTS * self.NUM_SUTS * len(runner_ensemble.annotators)

    def test_run_completes(self, runner_basic):
        assert_run_completes(runner_basic)

    def test_run_ensemble_completes(self, runner_ensemble):
        assert_run_completes(runner_ensemble)

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
