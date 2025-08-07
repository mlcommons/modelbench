import pytest
import re

from modelgauge.annotation_pipeline import (
    AnnotatorAssigner,
    AnnotatorSink,
    AnnotatorSource,
    AnnotatorWorkers,
)
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.dataset import PromptDataset, PromptResponseDataset
from modelgauge.data_schema import (
    DEFAULT_PROMPT_RESPONSE_SCHEMA as PROMPT_RESPONSE_SCHEMA,
    DEFAULT_PROMPT_SCHEMA as PROMPT_SCHEMA,
)
from modelgauge.pipeline_runner import (
    AnnotatorRunner,
    EnsembleRunner,
    PromptPlusAnnotatorRunner,
    PromptPlusEnsembleRunner,
    PromptRunner,
    build_runner,
)
from modelgauge.prompt_pipeline import (
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
    PromptSink,
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
        text = f"{PROMPT_SCHEMA.prompt_uid},{PROMPT_SCHEMA.prompt_text}\n"
        for i in range(NUM_PROMPTS):
            text += f"p{i},Prompt {i}\n"
        f.write(text)
    return file


@pytest.fixture
def prompts_dataset(prompts_file):
    return PromptDataset(prompts_file)


@pytest.fixture(scope="session")
def prompt_responses_file(tmp_path_factory):
    """Sample file with 2 prompts + responses from 2 SUTs for testing."""
    file = tmp_path_factory.mktemp("data") / "prompt-responses.csv"
    with open(file, "w") as f:
        text = f"{PROMPT_RESPONSE_SCHEMA.prompt_uid},{PROMPT_RESPONSE_SCHEMA.prompt_text},{PROMPT_RESPONSE_SCHEMA.sut_uid},{PROMPT_RESPONSE_SCHEMA.sut_response}\n"
        for i in range(NUM_PROMPTS):
            text += f"p{i},Prompt {i},sut1,Response {i}\n"
            text += f"p{i},Prompt {i},sut2,Response {i}\n"
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
    def runner_basic(self, tmp_path, prompts_dataset, suts):
        return PromptRunner(suts=suts, num_workers=32, input_dataset=prompts_dataset, output_dir=tmp_path)

    @pytest.mark.parametrize(
        "sut_uids,tag,expected_tail",
        [(["s1"], None, "s1"), (["s1", "s2"], None, "s1-s2"), (["s1"], "tag", "tag-s1")],
    )
    def test_run_id(self, tmp_path, prompts_dataset, sut_uids, tag, expected_tail):
        suts = {uid: FakeSUT(uid) for uid in sut_uids}
        runner = PromptRunner(suts=suts, num_workers=32, input_dataset=prompts_dataset, output_dir=tmp_path, tag=tag)
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic):
        assert runner_basic.output_dir() == tmp_path / runner_basic.run_id

    def test_pipeline_segments(self, tmp_path, prompts_dataset, prompts_file, suts):
        sut_options = SUTOptions(max_tokens=42)
        runner = PromptRunner(
            suts=suts, num_workers=20, input_dataset=prompts_dataset, output_dir=tmp_path, sut_options=sut_options
        )
        source, sut_assigner, sut_workers, sink = runner.pipeline_segments

        assert isinstance(source, PromptSource)
        assert isinstance(source.input, PromptDataset)
        assert source.input.path == prompts_file

        assert isinstance(sut_assigner, PromptSutAssigner)
        assert sut_assigner.suts == suts

        assert isinstance(sut_workers, PromptSutWorkers)
        assert sut_workers.suts == suts
        assert sut_workers.sut_options == sut_options
        assert sut_workers.thread_count == 20

        assert isinstance(sink, PromptSink)
        assert isinstance(sink.writer, PromptResponseDataset)

    def test_prompt_runner_num_input_items(self, runner_basic):
        assert runner_basic.num_input_items == NUM_PROMPTS

    @pytest.mark.parametrize("num_suts", [1, 2, 5])
    def test_num_total_items(self, tmp_path, prompts_dataset, num_suts):
        suts = {f"sut{i}": FakeSUT(f"sut{i}") for i in range(num_suts)}
        runner = PromptRunner(suts=suts, num_workers=20, input_dataset=prompts_dataset, output_dir=tmp_path)
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
    def runner_basic(self, tmp_path, prompts_dataset, suts, annotators):
        return PromptPlusAnnotatorRunner(
            suts=suts, annotators=annotators, num_workers=32, input_dataset=prompts_dataset, output_dir=tmp_path
        )

    @pytest.fixture
    def runner_ensemble(self, tmp_path, prompts_dataset, suts, annotators, ensemble):
        return PromptPlusEnsembleRunner(
            suts=suts,
            annotators=annotators,
            ensemble=ensemble,
            num_workers=32,
            input_dataset=prompts_dataset,
            output_dir=tmp_path,
        )

    @pytest.mark.parametrize(
        "annotator_uids,sut_uids,tag,expected_tail",
        [
            (["a1"], ["s1"], None, "s1-a1"),
            (["a1", "a2"], ["s1", "s2"], None, "s1-s2-a1-a2"),
            (["a1", "a2"], ["s1"], "tag", "tag-s1-a1-a2"),
        ],
    )
    def test_run_id(self, tmp_path, prompts_dataset, annotator_uids, sut_uids, tag, expected_tail):
        suts = {uid: FakeSUT(uid) for uid in sut_uids}
        annotators = {uid: FakeAnnotator(uid) for uid in annotator_uids}
        runner = PromptPlusAnnotatorRunner(
            suts=suts,
            annotators=annotators,
            num_workers=32,
            input_dataset=prompts_dataset,
            output_dir=tmp_path,
            tag=tag,
        )
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_run_id_with_ensemble(self, tmp_path, prompts_dataset, suts, annotators, ensemble):
        # Add extra annotator
        annotators["annotator4"] = FakeAnnotator("annotator4")
        runner = PromptPlusEnsembleRunner(
            suts=suts,
            annotators=annotators,
            ensemble=ensemble,
            num_workers=32,
            input_dataset=prompts_dataset,
            output_dir=tmp_path,
        )
        assert re.match(rf"\d{{8}}-\d{{6}}-sut1-sut2-annotator4-ensemble", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic):
        assert runner_basic.output_dir() == tmp_path / runner_basic.run_id

    def test_output_dir_ensemble(self, tmp_path, runner_ensemble):
        assert runner_ensemble.output_dir() == tmp_path / runner_ensemble.run_id

    def test_pipeline_segments(self, tmp_path, prompts_dataset, prompts_file, suts, annotators):
        sut_options = SUTOptions(max_tokens=42)
        runner = PromptPlusAnnotatorRunner(
            suts=suts,
            annotators=annotators,
            num_workers=20,
            input_dataset=prompts_dataset,
            output_dir=tmp_path,
            sut_options=sut_options,
        )
        source, sut_assigner, sut_workers, annotator_assigner, annotator_workers, sink = runner.pipeline_segments

        assert isinstance(source, PromptSource)
        assert isinstance(source.input, PromptDataset)
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

    def test_pipeline_segments_ensemble(self, runner_ensemble, annotators, ensemble):
        source, sut_assigner, sut_workers, annotator_assigner, annotator_workers, ensemble_worker, sink = (
            runner_ensemble.pipeline_segments
        )

        assert isinstance(annotator_workers, AnnotatorWorkers)
        assert annotator_workers.annotators == annotators

        assert ensemble_worker.ensemble == ensemble

        assert isinstance(sink, AnnotatorSink)

    def test_runner_num_input_items(self, runner_basic):
        assert runner_basic.num_input_items == NUM_PROMPTS

    @pytest.mark.parametrize("num_suts,num_annotators", [(1, 1), (1, 3), (3, 1), (3, 3)])
    def test_num_total_items(self, tmp_path, prompts_dataset, num_suts, num_annotators):
        suts = {f"sut{i}": FakeSUT(f"sut{i}") for i in range(num_suts)}
        annotators = {f"annotator{i}": FakeAnnotator(f"annotator{i}") for i in range(num_annotators)}
        runner = PromptPlusAnnotatorRunner(
            suts=suts,
            annotators=annotators,
            num_workers=20,
            input_dataset=prompts_dataset,
            output_dir=tmp_path,
        )
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

    def test_metadata_ensemble(self, runner_ensemble, suts):
        runner_ensemble.run(progress_callback=lambda _: _, debug=False)
        metadata = runner_ensemble.metadata()

        assert_common_metadata_is_correct(metadata, runner_ensemble)
        assert metadata["ensemble"]["annotators"] == ["annotator1", "annotator2", "annotator3"]
        assert metadata["ensemble"]["num_votes"] == NUM_PROMPTS * len(suts)


class TestAnnotatorRunner:
    NUM_SUTS = 2  # Number of SUTs included in the input prompts_response_file

    @pytest.fixture
    def prompt_responses_dataset(self, prompt_responses_file):
        return PromptResponseDataset(prompt_responses_file, mode="r")

    @pytest.fixture
    def runner_basic(self, tmp_path, prompt_responses_dataset, annotators):
        return AnnotatorRunner(
            annotators=annotators, num_workers=32, input_dataset=prompt_responses_dataset, output_dir=tmp_path
        )

    @pytest.fixture
    def runner_ensemble(self, tmp_path, prompt_responses_dataset, annotators, ensemble):
        return EnsembleRunner(
            annotators=annotators,
            ensemble=ensemble,
            num_workers=32,
            input_dataset=prompt_responses_dataset,
            output_dir=tmp_path,
        )

    @pytest.mark.parametrize(
        "annotator_uids,tag,expected_tail",
        [
            (["a1"], None, "a1"),
            (["a1", "a2"], None, "a1-a2"),
            (["a1", "a2"], "tag", "tag-a1-a2"),
        ],
    )
    def test_run_id(self, tmp_path, prompt_responses_dataset, annotator_uids, tag, expected_tail):
        annotators = {uid: FakeAnnotator(uid) for uid in annotator_uids}
        runner = AnnotatorRunner(
            annotators=annotators,
            num_workers=32,
            input_dataset=prompt_responses_dataset,
            output_dir=tmp_path,
            tag=tag,
        )
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_run_id_with_ensemble(self, tmp_path, prompt_responses_dataset, annotators, ensemble):
        # Add extra annotator
        annotators["annotator4"] = FakeAnnotator("annotator4")
        runner = EnsembleRunner(
            annotators=annotators,
            ensemble=ensemble,
            num_workers=32,
            input_dataset=prompt_responses_dataset,
            output_dir=tmp_path,
        )
        assert re.match(rf"\d{{8}}-\d{{6}}-annotator4-ensemble", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic):
        assert runner_basic.output_dir() == tmp_path / runner_basic.run_id

    def test_output_dir_ensemble(self, tmp_path, runner_ensemble):
        assert runner_ensemble.output_dir() == tmp_path / runner_ensemble.run_id

    def test_pipeline_segments(self, tmp_path, prompt_responses_dataset, prompt_responses_file, annotators):
        runner = AnnotatorRunner(
            annotators=annotators, num_workers=20, input_dataset=prompt_responses_dataset, output_dir=tmp_path
        )
        source, annotator_assigner, annotator_workers, sink = runner.pipeline_segments

        assert isinstance(source, AnnotatorSource)
        assert isinstance(source.input, PromptResponseDataset)
        assert source.input.path == prompt_responses_file

        assert isinstance(annotator_assigner, AnnotatorAssigner)
        assert annotator_assigner.annotators == annotators

        assert isinstance(annotator_workers, AnnotatorWorkers)
        assert annotator_workers.annotators == annotators
        assert annotator_workers.thread_count == 20

        assert isinstance(sink, AnnotatorSink)

    def test_pipeline_segments_ensemble(self, runner_ensemble, annotators, ensemble):
        source, annotator_assigner, annotator_workers, ensemble_worker, sink = runner_ensemble.pipeline_segments

        assert isinstance(annotator_workers, AnnotatorWorkers)
        assert annotator_workers.annotators == annotators

        assert ensemble_worker.ensemble == ensemble

        assert isinstance(sink, AnnotatorSink)

    def test_missing_ensemble_annotators_raises_error(self, tmp_path, prompt_responses_dataset, ensemble):
        incomplete_annotators = {"annotator1": FakeAnnotator("annotator1"), "annotator2": FakeAnnotator("annotator2")}
        with pytest.raises(ValueError, match="Ensemble annotators {'annotator3'} not found"):
            EnsembleRunner(
                annotators=incomplete_annotators,
                ensemble=ensemble,
                num_workers=20,
                input_dataset=prompt_responses_dataset,
                output_dir=tmp_path,
            )

    def test_runner_num_input_items(self, runner_basic):
        assert runner_basic.num_input_items == NUM_PROMPTS * self.NUM_SUTS

    @pytest.mark.parametrize("num_annotators", [1, 2, 5])
    def test_num_total_items(self, tmp_path, prompt_responses_dataset, num_annotators):
        annotators = {f"annotator{i}": FakeAnnotator(f"annotator{i}") for i in range(num_annotators)}
        runner = AnnotatorRunner(
            annotators=annotators, num_workers=20, input_dataset=prompt_responses_dataset, output_dir=tmp_path
        )
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

    def test_metadata_ensemble(self, runner_ensemble):
        runner_ensemble.run(progress_callback=lambda _: _, debug=False)
        metadata = runner_ensemble.metadata()

        assert_common_metadata_is_correct(metadata, runner_ensemble)
        assert metadata["ensemble"]["annotators"] == ["annotator1", "annotator2", "annotator3"]
        assert metadata["ensemble"]["num_votes"] == NUM_PROMPTS * self.NUM_SUTS


class TestBuildRunner:
    def test_build_prompt_runner(self, prompts_file, suts, tmp_path):
        runner = build_runner(prompts_file, suts=suts, num_workers=32, output_dir=tmp_path)
        assert isinstance(runner, PromptRunner)
        assert runner.suts == suts
        assert isinstance(runner.input_dataset, PromptDataset)
        assert runner.input_dataset.path == prompts_file

    def test_build_prompt_runner_parameterized_col_names(self, suts, tmp_path):
        file_path = tmp_path / "prompts.csv"
        file_path.write_text('"a","b"\n')
        runner = build_runner(
            file_path,
            suts=suts,
            num_workers=32,
            output_dir=tmp_path,
            prompt_uid_col="a",
            prompt_text_col="b",
        )
        assert isinstance(runner.input_dataset, PromptDataset)
        assert runner.input_dataset.path == file_path
        assert runner.input_dataset.schema.prompt_uid == "a"
        assert runner.input_dataset.schema.prompt_text == "b"

    def test_build_prompt_plus_ensemble_runner(self, prompts_file, suts, annotators, ensemble, tmp_path):
        runner = build_runner(
            prompts_file, suts=suts, annotators=annotators, ensemble=ensemble, num_workers=32, output_dir=tmp_path
        )
        assert isinstance(runner, PromptPlusEnsembleRunner)
        assert runner.suts == suts
        assert runner.annotators == annotators
        assert runner.ensemble == ensemble
        assert isinstance(runner.input_dataset, PromptDataset)
        assert runner.input_dataset.path == prompts_file

    def test_build_prompt_plus_ensemble_runner_parameterized_col_names(self, suts, annotators, ensemble, tmp_path):
        file_path = tmp_path / "prompts.csv"
        file_path.write_text('"a","b"\n')
        runner = build_runner(
            file_path,
            suts=suts,
            annotators=annotators,
            ensemble=ensemble,
            num_workers=32,
            output_dir=tmp_path,
            prompt_uid_col="a",
            prompt_text_col="b",
        )
        assert isinstance(runner.input_dataset, PromptDataset)
        assert runner.input_dataset.path == file_path
        assert runner.input_dataset.schema.prompt_uid == "a"
        assert runner.input_dataset.schema.prompt_text == "b"

    def test_build_ensemble_runner(self, prompt_responses_file, annotators, ensemble, tmp_path):
        runner = build_runner(
            prompt_responses_file, annotators=annotators, ensemble=ensemble, num_workers=32, output_dir=tmp_path
        )
        assert isinstance(runner, EnsembleRunner)
        assert runner.annotators == annotators
        assert runner.ensemble == ensemble
        assert isinstance(runner.input_dataset, PromptResponseDataset)
        assert runner.input_dataset.path == prompt_responses_file

    def test_build_ensemble_runner_parameterized_col_names(self, tmp_path, annotators, ensemble):
        file_path = tmp_path / "prompt-responses.csv"
        file_path.write_text('"a","b","c","d"\n')
        runner = build_runner(
            file_path,
            annotators=annotators,
            ensemble=ensemble,
            num_workers=32,
            output_dir=tmp_path,
            prompt_uid_col="a",
            prompt_text_col="b",
            sut_uid_col="c",
            sut_response_col="d",
        )
        assert isinstance(runner.input_dataset, PromptResponseDataset)
        assert runner.input_dataset.path == file_path
        assert runner.input_dataset.schema.prompt_uid == "a"
        assert runner.input_dataset.schema.prompt_text == "b"
        assert runner.input_dataset.schema.sut_uid == "c"
        assert runner.input_dataset.schema.sut_response == "d"

    def test_build_prompt_plus_annotator_runner(self, prompts_file, suts, annotators, tmp_path):
        runner = build_runner(prompts_file, suts=suts, annotators=annotators, num_workers=32, output_dir=tmp_path)
        assert isinstance(runner, PromptPlusAnnotatorRunner)
        assert runner.suts == suts
        assert runner.annotators == annotators
        assert isinstance(runner.input_dataset, PromptDataset)
        assert runner.input_dataset.path == prompts_file

    def test_build_prompt_plus_annotator_runner_parameterized_col_names(self, suts, annotators, tmp_path):
        file_path = tmp_path / "prompts.csv"
        file_path.write_text('"a","b"\n')
        runner = build_runner(
            file_path,
            suts=suts,
            annotators=annotators,
            num_workers=32,
            output_dir=tmp_path,
            prompt_uid_col="a",
            prompt_text_col="b",
        )
        assert isinstance(runner.input_dataset, PromptDataset)
        assert runner.input_dataset.path == file_path
        assert runner.input_dataset.schema.prompt_uid == "a"
        assert runner.input_dataset.schema.prompt_text == "b"

    def test_build_annotator_runner(self, prompt_responses_file, annotators, tmp_path):
        runner = build_runner(prompt_responses_file, annotators=annotators, num_workers=32, output_dir=tmp_path)
        assert isinstance(runner, AnnotatorRunner)
        assert runner.annotators == annotators
        assert isinstance(runner.input_dataset, PromptResponseDataset)
        assert runner.input_dataset.path == prompt_responses_file

    def test_build_annotator_runner_parameterized_col_names(self, tmp_path, annotators):
        file_path = tmp_path / "prompt-responses.csv"
        file_path.write_text('"a","b","c","d"\n')
        runner = build_runner(
            file_path,
            annotators=annotators,
            num_workers=32,
            output_dir=tmp_path,
            prompt_uid_col="a",
            prompt_text_col="b",
            sut_uid_col="c",
            sut_response_col="d",
        )
        assert isinstance(runner.input_dataset, PromptResponseDataset)
        assert runner.input_dataset.path == file_path
        assert runner.input_dataset.schema.prompt_uid == "a"
        assert runner.input_dataset.schema.prompt_text == "b"
        assert runner.input_dataset.schema.sut_uid == "c"
        assert runner.input_dataset.schema.sut_response == "d"
