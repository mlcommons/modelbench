import logging
import re
import sys
import types
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner, Result

from modelgauge import main
from modelgauge.annotator_registry import ANNOTATORS
from modelgauge.annotator_set import AnnotatorSet
from modelgauge.command_line import _validate_sut_uid, check_secrets, classify_sut_uids, listify, validate_uid
from modelgauge.config import MissingSecretsFromConfig
from modelgauge.data_schema import (
    DEFAULT_PROMPT_RESPONSE_SCHEMA as PROMPT_RESPONSE_SCHEMA,
    DEFAULT_PROMPT_SCHEMA as PROMPT_SCHEMA,
)
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import SUT, SUTNotFoundException, SUTOptions
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from modelgauge.test_registry import TESTS
from tests.modelgauge_tests.fake_annotator import FakeAnnotator
from tests.modelgauge_tests.fake_params import FakeParams
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
    with pytest.raises(SUTNotFoundException):
        _ = run_cli("run-sut", "--sut", "unknown-uid", "--prompt", "Can you say Hello?")


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

    options_arg = mock_translate_text_prompt.call_args_list[0][0][1]
    assert options_arg == SUTOptions(max_tokens=42, temperature=0.5, top_p=0.0, top_k_per_token=0)


@pytest.mark.parametrize("test", ["demo_01", "demo_02", "demo_03"])
def test_run_test_demos(sut_uid, test):
    result = run_cli("run-test", "--test", test, "--sut", sut_uid, "--max-test-items", "1")
    assert result.exit_code == 0
    assert re.search(r"Full TestRecord json written to output", result.output)


def test_run_test_invalid_sut_uid():
    TESTS.register(FakeTest, "fake-test")
    with pytest.raises(SUTNotFoundException):
        _ = run_cli("run-test", "--sut", "unknown-uid", "--test", "fake-test")
    del TESTS._lookup["fake-test"]


def test_run_test_invalid_test_uid(sut_uid):
    result = run_cli("run-test", "--sut", sut_uid, "--test", "unknown_uid")
    assert result.exit_code == 2
    assert re.search(r"Invalid value for '--test'", result.output)


@pytest.fixture(scope="session")
def prompts_file(tmp_path_factory):
    """Sample file with 2 prompts for testing."""
    file = tmp_path_factory.mktemp("data") / "prompts.csv"
    with open(file, "w") as f:
        f.write(f"{PROMPT_SCHEMA.prompt_uid},{PROMPT_SCHEMA.prompt_text}\n")
        f.write("p1,Say yes,ignored\np2,Refuse,ignored\n")
    return file


@pytest.fixture(scope="session")
def prompt_responses_file(tmp_path_factory):
    """Sample file with 2 prompts + responses from 1 SUT for testing."""
    file = tmp_path_factory.mktemp("data") / "prompt-responses.csv"
    with open(file, "w") as f:
        f.write(
            f"{PROMPT_RESPONSE_SCHEMA.prompt_uid},{PROMPT_RESPONSE_SCHEMA.prompt_text},{PROMPT_RESPONSE_SCHEMA.sut_uid},{PROMPT_RESPONSE_SCHEMA.sut_response}\n"
        )
        f.write("p1,Say yes,demo_yes_no,Yes\np2,Refuse,demo_yes_no,No\n")
    return file


def test_run_prompts_normal(caplog, tmp_path, prompts_file):
    caplog.set_level(logging.INFO)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", "--sut", "demo_yes_no", "-o", tmp_path, str(prompts_file)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.csv", caplog.text)[0]
    assert Path(out_path).exists()


@pytest.mark.parametrize("arg_name", ["--sut", "-s"])
def test_run_prompts_invalid_sut(arg_name, tmp_path, prompts_file):
    runner = CliRunner()
    with pytest.raises(SUTNotFoundException):
        result = runner.invoke(
            main.modelgauge_cli,
            ["run-csv-items", arg_name, "unknown-uid", "-o", tmp_path, str(prompts_file)],
            catch_exceptions=False,
        )


def test_run_prompts_multiple_invalid_suts(tmp_path, prompts_file):
    runner = CliRunner()
    with pytest.raises(SUTNotFoundException):
        result = runner.invoke(
            main.modelgauge_cli,
            ["run-csv-items", "--sut", "unknown-uid1", "--sut", "unknown-uid2", "-o", tmp_path, str(prompts_file)],
            catch_exceptions=False,
        )


def test_run_prompts_invalid_annotator(tmp_path, prompts_file, sut_uid):
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", "--sut", sut_uid, "--annotator", "unknown-uid", "-o", tmp_path, str(prompts_file)],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert re.search(r"Invalid value for '-a' / '--annotator': Unknown uid: '\['unknown-uid'\]'", result.output)


def test_run_prompts_with_annotators(caplog, tmp_path, prompts_file):
    caplog.set_level(logging.INFO)
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
            str(prompts_file),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.csv", caplog.text)[0]
    assert Path(out_path).exists()


@patch("modelgauge.suts.demo_01_yes_no_sut.DemoYesNoSUT.translate_text_prompt")
@pytest.mark.parametrize("extra_options", [[], ["--annotator", "demo_annotator"]])
def test_run_prompts_with_options(mock_translate_text_prompt, tmp_path, prompts_file, extra_options):
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
            str(prompts_file),
            *extra_options,
        ],
        catch_exceptions=False,
    )

    options_arg = mock_translate_text_prompt.call_args_list[0][0][1]
    assert options_arg == SUTOptions(max_tokens=42, temperature=0.5, top_p=0.0, top_k_per_token=0)


@modelgauge_sut(capabilities=[])
class NoReqsSUT(SUT):
    pass


def test_run_prompts_bad_sut(tmp_path, prompts_file):
    SUTS.register(NoReqsSUT, "noreqs")

    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", "--sut", "noreqs", "-o", tmp_path, str(prompts_file)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert re.search(r"noreqs does not accept text prompts", str(result.output))


def test_run_annotators(caplog, tmp_path, prompt_responses_file):
    caplog.set_level(logging.INFO)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        [
            "run-csv-items",
            "--annotator",
            "demo_annotator",
            "-o",
            tmp_path,
            str(prompt_responses_file),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.csv", caplog.text)[0]
    assert Path(out_path).exists()


@pytest.mark.parametrize(
    "option_name,option_val", [("max-tokens", "42"), ("top-p", "0.5"), ("temp", "0.5"), ("top-k", 0)]
)
def test_run_annotators_with_sut_options(tmp_path, prompt_responses_file, option_name, option_val):
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
                str(prompt_responses_file),
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


def test_run_job_sut_only_output_name(caplog, tmp_path, prompts_file):
    caplog.set_level(logging.INFO)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--sut", "demo_yes_no", "--output-dir", tmp_path, str(prompts_file)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.csv", caplog.text)[0])

    assert out_path.exists()
    assert out_path.name == "prompt-responses.csv"  # File name
    assert re.match(r"\d{8}-\d{6}-demo_yes_no", out_path.parent.name)  # Subdir name
    assert out_path.parent.parent == tmp_path  # Parent dir

    metadata_path = out_path.parent / "metadata.json"
    assert metadata_path.exists()


def test_run_job_with_tag_output_name(caplog, tmp_path, prompts_file):
    caplog.set_level(logging.INFO)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--sut", "demo_yes_no", "--output-dir", tmp_path, "--tag", "test", str(prompts_file)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.csv", caplog.text)[0])

    assert re.match(r"\d{8}-\d{6}-test-demo_yes_no", out_path.parent.name)  # Subdir name


def test_run_job_sut_and_annotator_output_name(caplog, tmp_path, prompts_file):
    caplog.set_level(logging.INFO)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        [
            "run-job",
            "--sut",
            "demo_yes_no",
            "--annotator",
            "demo_annotator",
            "--output-dir",
            tmp_path,
            str(prompts_file),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.csv", caplog.text)[0])

    assert out_path.exists()
    assert out_path.name == "prompt-responses-annotated.csv"  # File name
    assert re.match(r"\d{8}-\d{6}-demo_yes_no-demo_annotator", out_path.parent.name)  # Subdir name
    assert out_path.parent.parent == tmp_path  # Parent dir

    metadata_path = out_path.parent / "metadata.json"
    assert metadata_path.exists()


def test_run_job_annotators_only_output_name(caplog, tmp_path, prompt_responses_file):
    caplog.set_level(logging.INFO)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-job", "--annotator", "demo_annotator", "--output-dir", tmp_path, str(prompt_responses_file)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.csv", caplog.text)[0])

    assert out_path.exists()
    assert out_path.name == "annotations.csv"  # File name
    assert re.match(r"\d{8}-\d{6}-demo_annotator", out_path.parent.name)  # Subdir name
    assert out_path.parent.parent == tmp_path  # Parent dir

    metadata_path = out_path.parent / "metadata.json"
    assert metadata_path.exists()


@pytest.mark.parametrize("cmd", ["run-job", "run-csv-items"])
def test_run_ensemble(caplog, tmp_path, prompt_responses_file, cmd):
    caplog.set_level(logging.INFO)

    # Create a dummy module and object
    class FakeEnsemble(AnnotatorSet):
        annotators = ["demo_annotator"]

        def evaluate(self, item):
            return {"ensemble_vote": 1.0}

    dummy_module = types.ModuleType("modelgauge.private_set")
    dummy_annotator_set = FakeEnsemble()
    dummy_module.PRIVATE_ANNOTATOR_SET = dummy_annotator_set
    with patch.dict(sys.modules, {"modelgauge.private_ensemble_annotator_set": dummy_module}):
        runner = CliRunner()
        result = runner.invoke(
            main.modelgauge_cli,
            [
                cmd,
                "--ensemble",
                "--output-dir",
                tmp_path,
                str(prompt_responses_file),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0

    out_path = Path(re.findall(r"\S+\.csv", caplog.text)[0])

    assert out_path.exists()
    assert out_path.name == "annotations.csv"  # File name
    assert re.match(r"\d{8}-\d{6}-ensemble", out_path.parent.name)  # Subdir name
    assert out_path.parent.parent == tmp_path  # Parent dir

    metadata_path = out_path.parent / "metadata.json"
    assert metadata_path.exists()


@pytest.mark.parametrize("cmd", ["run-job", "run-csv-items"])
def test_run_missing_ensemble_raises_error(tmp_path, prompt_responses_file, cmd):
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        [
            cmd,
            "--ensemble",
            "--output-dir",
            tmp_path,
            str(prompt_responses_file),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 2
    assert re.search(r"Invalid value: Ensemble annotators are not available.", result.output)


@patch("modelgauge.command_line.make_dynamic_sut_for", autospec=True)  # prevents huggingface API lookups
def test_classify(patched):
    SUTS.register(SUT, "known", "something")
    classified = classify_sut_uids(["known", "google:gemma:nebius:hfrelay", "not-known"])
    assert classified == {"known": ["known"], "dynamic": ["google:gemma:nebius:hfrelay"], "unknown": ["not-known"]}
    del SUTS._lookup["known"]


@patch(
    "modelgauge.command_line.make_dynamic_sut_for",
    autospec=True,
    return_value=(SUT, "google:gemma:nebius:hfrelay", "google:gemma:nebius:hfrelay"),
)  # prevents huggingface API lookups
def test_validate_sut_uid(patched):
    assert _validate_sut_uid(None, FakeParams(["--sut"]), None) is None
    assert _validate_sut_uid(None, FakeParams(["--sut"]), "") is ""
    with pytest.raises(ValueError):
        _ = _validate_sut_uid(None, FakeParams(["--bad"]), "bogus")

    SUTS.register(SUT, "known", "something")
    assert _validate_sut_uid(None, FakeParams(["--sut"]), "known") == "known"

    SUTS.register(SUT, "known-2", "something-else")
    returned = _validate_sut_uid(None, FakeParams(["--sut"]), ("known", "known-2"))
    assert len(returned) == 2
    assert "known" in returned
    assert "known-2" in returned

    assert (
        _validate_sut_uid(None, FakeParams(["--sut"]), "google:gemma:nebius:hfrelay") == "google:gemma:nebius:hfrelay"
    )
    del SUTS._lookup["known"]
    del SUTS._lookup["known-2"]


def test_validate_uid():
    assert validate_uid(None, None, None) is None
    assert validate_uid(None, None, "") is ""

    with pytest.raises(ValueError):
        _ = validate_uid(None, FakeParams(["--bad"]), "bogus")

    # test single argument
    TESTS.register(FakeTest, "my-fake-test")
    assert (
        validate_uid(
            None,
            FakeParams(
                [
                    "--test",
                ]
            ),
            "my-fake-test",
        )
        == "my-fake-test"
    )

    # test multiple arguments
    # we're not testing multiples for all param types b/c the code is the same
    TESTS.register(FakeTest, "my-fake-test-2")
    assert validate_uid(
        None,
        FakeParams(
            [
                "--test",
            ]
        ),
        ("my-fake-test", "my-fake-test-2"),
    ) == ("my-fake-test", "my-fake-test-2")

    del TESTS._lookup["my-fake-test"]
    del TESTS._lookup["my-fake-test-2"]

    SUTS.register(SUT, "my-fake-sut")
    assert (
        validate_uid(
            None,
            FakeParams(
                [
                    "--sut",
                ]
            ),
            "my-fake-sut",
        )
        == "my-fake-sut"
    )
    del SUTS._lookup["my-fake-sut"]

    ANNOTATORS.register(FakeAnnotator, "my-fake-annotator")
    assert (
        validate_uid(
            None,
            FakeParams(
                [
                    "--annotator",
                ]
            ),
            "my-fake-annotator",
        )
        == "my-fake-annotator"
    )


def test_listify():
    assert listify("string") == [
        "string",
    ]
    assert listify(["a", "b"]) == ["a", "b"]
    assert listify(("a", "b")) == ("a", "b")

    def noop():
        pass

    class Noop:
        pass

    assert listify(noop) == noop
    assert listify(Noop) == Noop
    n = Noop()
    assert listify(n) == n

    assert listify(None) is None
