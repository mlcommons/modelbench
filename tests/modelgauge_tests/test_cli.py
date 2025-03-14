import csv
import json
import logging
import re
from pathlib import Path
from unittest.mock import patch

import jsonlines

import pytest
from click.testing import CliRunner, Result

from modelgauge import main
from modelgauge.config import MissingSecretsFromConfig
from modelgauge.command_line import check_secrets
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import SUT, SUTOptions
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS
from tests.modelgauge_tests.fake_annotator import FakeAnnotator
from tests.modelgauge_tests.fake_secrets import FakeRequiredSecret
from tests.modelgauge_tests.fake_test import FakeTest

LOGGER = logging.getLogger(__name__)


def run_cli(*args) -> Result:
    # noinspection PyTypeChecker
    result = CliRunner().invoke(main.modelgauge_cli, args, catch_exceptions=False)
    return result


def test_main():
    result = run_cli()
    assert result.exit_code == 0
    assert re.search(r"Usage: modelgauge \[OPTIONS]", result.stdout)


def test_list():
    result = run_cli("list")

    assert result.exit_code == 0
    assert re.search(r"Plugin Modules:", result.stdout)


def test_list_secrets():
    result = run_cli("list-secrets")

    assert result.exit_code == 0
    assert re.search(r"secrets", result.stdout)


def test_list_tests():
    result = run_cli("list-tests")

    assert result.exit_code == 0
    assert re.search(r"Class: DemoSimpleQATest", result.stdout)


def test_list_suts():
    result = run_cli("list-suts")

    assert result.exit_code == 0
    assert re.search(r"DemoConstantSUT", result.output)


@pytest.mark.parametrize(
    "sut",
    [
        "demo_yes_no",
        "demo_random_words",
        "demo_always_angry",
        "demo_always_sorry",
    ],
)
def test_run_sut_demos(sut):
    result = run_cli("run-sut", "--sut", sut, "--prompt", "Can you say Hello?")
    assert result.exit_code == 0
    assert re.search(r"Native response:", result.output)


def test_run_sut_invalid_uid():
    result = run_cli("run-sut", "--sut", "unknown-uid", "--prompt", "Can you say Hello?")
    assert result.exit_code == 2
    assert re.search(r"Invalid value for '--sut'", result.output)


@patch("modelgauge.suts.demo_01_yes_no_sut.DemoYesNoSUT.translate_text_prompt")
def test_run_sut_with_options(mock_translate_text_prompt):
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        [
            "run-sut",
            "--sut",
            "demo_yes_no",
            "--prompt",
            "Can you say Hello?",
            "--max-tokens",
            "42",
            "--top-p",
            "0",
            "--temp",
            "0.5",
            "--top-k",
            "0",
        ],
        catch_exceptions=False,
    )

    prompt_arg = mock_translate_text_prompt.call_args_list[0][0][0]
    assert prompt_arg.options == SUTOptions(max_tokens=42, temperature=0.5, top_p=0.0, top_k_per_token=0)


@pytest.mark.parametrize("test", ["demo_01", "demo_02", "demo_03"])
def test_run_test_demos(sut_uid, test):
    result = run_cli("run-test", "--test", test, "--sut", sut_uid, "--max-test-items", "1")
    assert result.exit_code == 0
    assert re.search(r"Full TestRecord json written to output", result.output)


def test_run_test_invalid_sut_uid():
    TESTS.register(FakeTest, "fake-test")
    result = run_cli("run-test", "--sut", "unknown-uid", "--test", "fake-test")
    del TESTS._lookup["fake-test"]
    assert result.exit_code == 2
    assert re.search(r"Invalid value for '--sut'", result.output)


def test_run_test_invalid_test_uid(sut_uid):
    result = run_cli("run-test", "--sut", sut_uid, "--test", "unknown_uid")
    assert result.exit_code == 2
    assert re.search(r"Invalid value for '--test'", result.output)


def create_prompts_file(path):
    in_path = (path / "input.csv").absolute()
    with open(in_path, "w") as f:
        f.write("UID,Text,Ignored\np1,Say yes,ignored\np2,Refuse,ignored\n")
    return in_path


def test_run_prompts_normal(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", "--sut", "demo_yes_no", "-o", tmp_path, str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.csv", caplog.text)[0]
    with open(out_path, "r") as f:
        reader = csv.DictReader(f)

        rows = (next(reader), next(reader))
        rows = sorted(rows, key=lambda row: row["UID"])
        expected = (
            {"UID": "p1", "Text": "Say yes", "demo_yes_no": "Yes"},
            {"UID": "p2", "Text": "Refuse", "demo_yes_no": "No"},
        )
        assert rows[0] == expected[0]
        assert rows[1] == expected[1]


@pytest.mark.parametrize("arg_name", ["--sut", "-s"])
def test_run_prompts_invalid_sut(arg_name, tmp_path):
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", arg_name, "unknown-uid", "-o", tmp_path, str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert re.search(r"Invalid value for '-s' / '--sut': Unknown uid: '\['unknown-uid'\]'", result.output)


def test_run_prompts_multiple_invalid_suts(tmp_path):
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", "--sut", "unknown-uid1", "--sut", "unknown-uid2", "-o", tmp_path, str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert re.search(
        r"Invalid value for '-s' / '--sut': Unknown uids: '\['unknown-uid1', 'unknown-uid2'\]'", result.output
    )


def test_run_prompts_invalid_annotator(sut_uid, tmp_path):
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", "--sut", sut_uid, "--annotator", "unknown-uid", "-o", tmp_path, str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert re.search(r"Invalid value for '-a' / '--annotator': Unknown uid: '\['unknown-uid'\]'", result.output)


def test_run_prompts_with_annotators(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        [
            "run-csv-items",
            "--sut",
            "demo_yes_no",
            "--annotator",
            "demo_annotator",
            "--workers",
            "5",
            "-o",
            tmp_path,
            str(in_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.jsonl", caplog.text)[0]
    output = []
    with jsonlines.open(out_path) as reader:
        output.append(reader.read())
        output.append(reader.read())
    assert {
        "UID": "p1",
        "Prompt": "Say yes",
        "SUT": "demo_yes_no",
        "Response": "Yes",
        "Annotations": {"demo_annotator": {"badness": 1.0}},
    } in output
    assert {
        "UID": "p2",
        "Prompt": "Refuse",
        "SUT": "demo_yes_no",
        "Response": "No",
        "Annotations": {"demo_annotator": {"badness": 0.0}},
    } in output


@patch("modelgauge.suts.demo_01_yes_no_sut.DemoYesNoSUT.translate_text_prompt")
@pytest.mark.parametrize("extra_options", [[], ["--annotator", "demo_annotator"]])
def test_run_prompts_with_options(mock_translate_text_prompt, tmp_path, extra_options):
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        [
            "run-csv-items",
            "--sut",
            "demo_yes_no",
            "--max-tokens",
            "42",
            "--top-p",
            "0",
            "--temp",
            "0.5",
            "--top-k",
            "0",
            "-o",
            tmp_path,
            str(in_path),
            *extra_options,
        ],
        catch_exceptions=False,
    )

    prompt_arg = mock_translate_text_prompt.call_args_list[0][0][0]
    assert prompt_arg.options == SUTOptions(max_tokens=42, temperature=0.5, top_p=0.0, top_k_per_token=0)


@modelgauge_sut(capabilities=[])
class NoReqsSUT(SUT):
    pass


def test_run_prompts_bad_sut(tmp_path):
    in_path = create_prompts_file(tmp_path)
    SUTS.register(NoReqsSUT, "noreqs")

    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", "--sut", "noreqs", "-o", tmp_path, str(in_path)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert re.search(r"noreqs does not accept text prompts", str(result.output))


def create_prompt_responses_file(path):
    in_path = (path / "input.csv").absolute()
    with open(in_path, "w") as f:
        f.write("UID,Prompt,SUT,Response\np1,Say yes,demo_yes_no,Yes\np2,Refuse,demo_yes_no,No\n")
    return in_path


def test_run_annotators(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompt_responses_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        [
            "run-csv-items",
            "--annotator",
            "demo_annotator",
            "-o",
            tmp_path,
            str(in_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.jsonl", caplog.text)[0]
    with jsonlines.open(out_path) as reader:
        assert reader.read() == {
            "UID": "p1",
            "Prompt": "Say yes",
            "SUT": "demo_yes_no",
            "Response": "Yes",
            "Annotations": {"demo_annotator": {"badness": 1.0}},
        }
        assert reader.read() == {
            "UID": "p2",
            "Prompt": "Refuse",
            "SUT": "demo_yes_no",
            "Response": "No",
            "Annotations": {"demo_annotator": {"badness": 0.0}},
        }


@pytest.mark.parametrize(
    "option_name,option_val", [("max-tokens", "42"), ("top-p", "0.5"), ("temp", "0.5"), ("top-k", 0)]
)
def test_run_annotators_with_sut_options(tmp_path, option_name, option_val):
    in_path = create_prompt_responses_file(tmp_path)
    runner = CliRunner()
    with pytest.warns(UserWarning, match="Received SUT options"):
        result = runner.invoke(
            main.modelgauge_cli,
            [
                "run-csv-items",
                "--annotator",
                "demo_annotator",
                f"--{option_name}",
                option_val,
                "-o",
                tmp_path,
                str(in_path),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0


def test_check_secrets_checks_annotators_in_test():
    """Make sure that check_secrets recursively checks the annotators' secrets in the test."""

    # Register a test with an annotator that requires a secret.
    class FakeTestWithAnnotator(FakeTest):
        @classmethod
        def get_annotators(cls):
            return ["secret-annotator"]

    class FakeAnnotatorWithSecrets(FakeAnnotator):
        def __init__(self, uid, secret):
            super().__init__(uid)
            self.secret = secret

    ANNOTATORS.register(FakeAnnotatorWithSecrets, "secret-annotator", InjectSecret(FakeRequiredSecret))
    TESTS.register(FakeTestWithAnnotator, "some-test")
    with pytest.raises(MissingSecretsFromConfig):
        check_secrets({}, test_uids=["some-test"])


def test_run_job_sut_only_output_name(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--sut", "demo_yes_no", "--output-dir", tmp_path, str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.csv", caplog.text)[0])

    assert out_path.exists()
    assert out_path.name == "prompt-responses.csv"  # File name
    assert re.match(r"\d{8}-\d{6}-demo_yes_no", out_path.parent.name)  # Subdir name
    assert out_path.parent.parent == tmp_path  # Parent dir


def test_run_job_sut_only_metadata(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--sut", "demo_yes_no", "--output-dir", tmp_path, str(in_path)],
        catch_exceptions=False,
    )
    out_path = Path(re.findall(r"\S+\.csv", caplog.text)[0])
    metadata_path = out_path.parent / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert re.match(r"\d{8}-\d{6}-demo_yes_no", metadata["run_id"])
    assert "started" in metadata["run_info"]
    assert "finished" in metadata["run_info"]
    assert "duration" in metadata["run_info"]
    assert metadata["input"] == {"source": in_path.name, "num_items": 2}
    assert metadata["suts"] == [
        {
            "uid": "demo_yes_no",
            "initialization_record": {
                "args": ["demo_yes_no"],
                "class_name": "DemoYesNoSUT",
                "kwargs": {},
                "module": "modelgauge.suts.demo_01_yes_no_sut",
            },
            "sut_options": {"max_tokens": 100},
        }
    ]
    assert metadata["responses"] == {"count": 2, "by_sut": {"demo_yes_no": {"count": 2}}}


def test_run_job_with_tag_output_name(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--sut", "demo_yes_no", "--output-dir", tmp_path, "--tag", "test", str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.csv", caplog.text)[0])

    assert re.match(r"\d{8}-\d{6}-test-demo_yes_no", out_path.parent.name)  # Subdir name


def test_run_job_sut_and_annotator_output_name(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--sut", "demo_yes_no", "--annotator", "demo_annotator", "--output-dir", tmp_path, str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.jsonl", caplog.text)[0])

    assert out_path.exists()
    assert out_path.name == "prompt-responses-annotated.jsonl"  # File name
    assert re.match(r"\d{8}-\d{6}-demo_yes_no-demo_annotator", out_path.parent.name)  # Subdir name
    assert out_path.parent.parent == tmp_path  # Parent dir


def test_run_job_sut_and_annotator_metadata(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--sut", "demo_yes_no", "--annotator", "demo_annotator", "--output-dir", tmp_path, str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.jsonl", caplog.text)[0])
    metadata_path = out_path.parent / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert re.match(r"\d{8}-\d{6}-demo_yes_no-demo_annotator", metadata["run_id"])
    assert "started" in metadata["run_info"]
    assert "finished" in metadata["run_info"]
    assert "duration" in metadata["run_info"]
    assert metadata["input"] == {"source": in_path.name, "num_items": 2}
    assert metadata["suts"] == [
        {
            "uid": "demo_yes_no",
            "initialization_record": {
                "args": ["demo_yes_no"],
                "class_name": "DemoYesNoSUT",
                "kwargs": {},
                "module": "modelgauge.suts.demo_01_yes_no_sut",
            },
            "sut_options": {"max_tokens": 100},
        }
    ]
    assert metadata["responses"] == {"count": 2, "by_sut": {"demo_yes_no": {"count": 2}}}
    assert metadata["annotators"] == [{"uid": "demo_annotator"}]
    assert metadata["annotations"] == {"count": 2, "by_annotator": {"demo_annotator": {"count": 2}}}


def test_run_job_annotators_only_output_name(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompt_responses_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--annotator", "demo_annotator", "--output-dir", tmp_path, str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.jsonl", caplog.text)[0])

    assert out_path.exists()
    assert out_path.name == "annotations.jsonl"  # File name
    assert re.match(r"\d{8}-\d{6}-demo_annotator", out_path.parent.name)  # Subdir name
    assert out_path.parent.parent == tmp_path  # Parent dir


def test_run_job_annotators_only_metadata(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    in_path = create_prompt_responses_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--annotator", "demo_annotator", "--output-dir", tmp_path, str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.jsonl", caplog.text)[0])
    metadata_path = out_path.parent / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    assert re.match(r"\d{8}-\d{6}-demo_annotator", metadata["run_id"])
    assert "started" in metadata["run_info"]
    assert "finished" in metadata["run_info"]
    assert "duration" in metadata["run_info"]
    assert metadata["input"] == {"source": in_path.name, "num_items": 2}
    assert metadata["annotators"] == [{"uid": "demo_annotator"}]
    assert metadata["annotations"] == {"count": 2, "by_annotator": {"demo_annotator": {"count": 2}}}
