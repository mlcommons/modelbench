import csv
import jsonlines
import re

import pytest
from click.testing import CliRunner, Result

from modelgauge import main
from modelgauge.sut import SUT
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS


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


@pytest.mark.parametrize("test", ["demo_01", "demo_02", "demo_03", "demo_04"])
def test_run_test_demos(test):
    result = run_cli(
        "run-test", "--test", test, "--sut", "demo_yes_no", "--max-test-items", "1"
    )
    print(result)
    print(result.stdout)
    assert result.exit_code == 0
    assert re.search(r"Full TestRecord json written to output", result.output)


def create_prompts_file(path):
    in_path = (path / "input.csv").absolute()
    with open(in_path, "w") as f:
        f.write("UID,Text,Ignored\np1,Say yes,ignored\np2,Refuse,ignored\n")
    return in_path


def test_run_prompts_normal(tmp_path):
    in_path = create_prompts_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", "--sut", "demo_yes_no", str(in_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.csv", result.stdout)[0]
    with open(tmp_path / out_path, "r") as f:
        reader = csv.DictReader(f)

        row1 = next(reader)
        assert row1["UID"] == "p1"
        assert row1["Text"] == "Say yes"
        assert row1["demo_yes_no"] == "Yes"

        row2 = next(reader)
        assert row2["UID"] == "p2"
        assert row2["Text"] == "Refuse"
        assert row2["demo_yes_no"] == "No"


def test_run_prompts_with_annotators(tmp_path):
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
            str(in_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.jsonl", result.stdout)[0]
    with jsonlines.open(tmp_path / out_path) as reader:
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


@modelgauge_sut(capabilities=[])
class NoReqsSUT(SUT):
    pass


def test_run_prompts_bad_sut(tmp_path):
    in_path = create_prompts_file(tmp_path)
    SUTS.register(NoReqsSUT, "noreqs")

    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        ["run-csv-items", "--sut", "noreqs", str(in_path)],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert re.search(r"noreqs does not accept text prompts", str(result.output))


def create_prompt_responses_file(path):
    in_path = (path / "input.csv").absolute()
    with open(in_path, "w") as f:
        f.write(
            "UID,Prompt,SUT,Response\np1,Say yes,demo_yes_no,Yes\np2,Refuse,demo_yes_no,No\n"
        )
    return in_path


def test_run_annotators(tmp_path):
    in_path = create_prompt_responses_file(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main.modelgauge_cli,
        [
            "run-csv-items",
            "--annotator",
            "demo_annotator",
            str(in_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    out_path = re.findall(r"\S+\.jsonl", result.stdout)[0]
    with jsonlines.open(tmp_path / out_path) as reader:
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
