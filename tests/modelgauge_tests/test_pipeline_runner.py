import pytest
import re

from modelgauge.pipeline_runner import PromptRunner
from modelgauge.prompt_pipeline import (
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
    PromptSink,
    CsvPromptInput,
    CsvPromptOutput,
)
from modelgauge.sut import SUTOptions
from modelgauge_tests.fake_sut import FakeSUT, FakeSUTResponse, FakeSUTRequest


class AlwaysFailingSUT(FakeSUT):
    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        raise Exception("I don't wanna")


class SometimesFailingSUT(FakeSUT):
    """Fails to evaluate on even-numbered requests."""
    def __init__(self, uid):
        super().__init__(uid)
        self.eval_count = 0

    def evaluate(self, request: FakeSUTRequest) -> FakeSUTResponse:
        self.eval_count += 1
        if self.eval_count % 2 == 0:
            print("I don't wanna")
            raise Exception("I don't wanna")
        print(self.uid, "Eval")
        super().evaluate(request)


@pytest.fixture(scope="session")
def prompts_file(tmp_path_factory):
    """Sample file with 2 prompts for testing."""
    file = tmp_path_factory.mktemp("data") / "prompts.csv"
    with open(file, "w") as f:
        f.write("UID,Text,Ignored\np1,Say yes,ignored\np2,Refuse,ignored\n")
    return file


@pytest.fixture(scope="session")
def prompt_responses_file(tmp_path_factory):
    """Sample file with 2 prompts + responses from 1 SUT for testing."""
    file = tmp_path_factory.mktemp("data") / "prompt-responses.csv"
    with open(file, "w") as f:
        f.write("UID,Prompt,SUT,Response\np1,Say yes,demo_yes_no,Yes\np2,Refuse,demo_yes_no,No\n")
    return file


class TestPromptRunner:
    @pytest.fixture
    def runner_basic(self, tmp_path, prompts_file, suts):
        return PromptRunner(32, prompts_file, tmp_path, None, SUTOptions(), "tag", suts=suts)

    @pytest.fixture
    def runner_some_bad_suts(self, tmp_path, prompts_file):
        suts = {"good_sut": FakeSUT("good_sut"), "bad_sut": SometimesFailingSUT("bad_sut"), "very_bad_sut": AlwaysFailingSUT("very_bad_sut")}
        return PromptRunner(32, prompts_file, tmp_path, None, SUTOptions(), "tag", suts=suts)

    @pytest.fixture(scope="session")
    def suts(self):
        return {"sut1": FakeSUT("sut1"), "sut2": FakeSUT("sut2")}

    @pytest.mark.parametrize(
        "sut_uids,tag,expected_tail",
        [(["sut1"], None, "sut1"), (["sut1", "sut2"], None, "sut1-sut2"), (["sut1"], "tag", "tag-sut1")],
    )
    def test_run_id(self, tmp_path, prompts_file, sut_uids, tag, expected_tail):
        runner = PromptRunner(
            32, prompts_file, tmp_path, None, SUTOptions(), tag, suts={uid: FakeSUT(uid) for uid in sut_uids}
        )
        assert re.match(rf"\d{{8}}-\d{{6}}-{expected_tail}", runner.run_id)

    def test_output_dir(self, tmp_path, runner_basic, prompts_file):
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
        assert runner_basic.num_input_items == 2

    @pytest.mark.parametrize("num_suts", [1, 2, 5])
    def test_num_total_items(self, tmp_path, prompts_file, num_suts):
        suts = {f"sut{i}": FakeSUT(f"sut{i}") for i in range(num_suts)}
        runner = PromptRunner(20, prompts_file, tmp_path, None, SUTOptions(), None, suts=suts)
        assert runner.num_total_items == 2 * num_suts

    def test_run(self, runner_basic):
        runner_basic.run(progress_callback=lambda _: _, debug=False)
        output = runner_basic.output_dir() / runner_basic.output_file_name
        assert output.exists()

    def test_run_completes_with_failing_sut(self, runner_some_bad_suts):
        runner_some_bad_suts.run(progress_callback=lambda _: _, debug=False)
        output = runner_some_bad_suts.output_dir() / runner_some_bad_suts.output_file_name
        assert output.exists()

    def test_metadata(self, runner_basic, prompts_file):
        runner_basic.run(progress_callback=lambda _: _, debug=False)
        metadata = runner_basic.metadata()

        assert metadata["run_id"] == runner_basic.run_id
        assert "started" in metadata["run_info"]
        assert "finished" in metadata["run_info"]
        assert "duration" in metadata["run_info"]
        assert metadata["input"] == {"source": prompts_file.name, "num_items": 2}
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
            }
        ]
        assert metadata["responses"] == {"count": 4, "by_sut": {"sut1": {"count": 2}, "sut2": {"count": 2}}}

    def test_metadata_with_bad_sut(self, runner_some_bad_suts, prompts_file):
        runner_some_bad_suts.run(progress_callback=lambda _: _, debug=False)
        metadata = runner_some_bad_suts.metadata()

        assert metadata["suts"] == [
            {
                "uid": "good_sut",
                "initialization_record": {
                    "args": ["good_sut"],
                    "class_name": "FakeSUT",
                    "kwargs": {},
                    "module": "modelgauge_tests.fake_sut",
                },
                "sut_options": {"max_tokens": 100},
            },
            {
                "uid": "bad_sut",
                "initialization_record": {
                    "args": ["bad_sut"],
                    "class_name": "SometimesFailingSUT",
                    "kwargs": {},
                    "module": "modelgauge_tests.test_pipeline_runner",
                },
                "sut_options": {"max_tokens": 100},
            },
            {
                "uid": "very_bad_sut",
                "initialization_record": {
                    "args": ["very_bad_sut"],
                    "class_name": "AlwaysFailingSUT",
                    "kwargs": {},
                    "module": "modelgauge_tests.test_pipeline_runner",
                },
                "sut_options": {"max_tokens": 100},
            }
        ]
        # TODO: Add "very_bad_sut": 0
        assert metadata["responses"] == {"count": 3, "by_sut": {"good_sut": {"count": 2}}, "bad_sut": {"count": 1}}