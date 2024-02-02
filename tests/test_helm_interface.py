from unittest.mock import MagicMock

import pytest

from coffee import helm_interface
from coffee.helm_runner import BbqHelmTest, HelmSut


@pytest.fixture(autouse=True)
def monkeypatch_run_all(monkeypatch):
    mock_obj = MagicMock()
    monkeypatch.setattr(helm_interface.Runner, "run_all", mock_obj)
    return mock_obj


@pytest.fixture(autouse=True)
def monkeypatch_run_one(monkeypatch):
    mock_obj = MagicMock()
    monkeypatch.setattr(helm_interface.Runner, "run_one", mock_obj)
    return mock_obj


@pytest.fixture
def monkeypatch_register_huggingface(monkeypatch):
    mock_obj = MagicMock()
    monkeypatch.setattr(
        helm_interface,
        "register_huggingface_hub_model_from_flag_value",
        mock_obj,
    )
    return mock_obj


@pytest.fixture
def monkeypatch_run_entries_to_run_specs(monkeypatch):
    mock_obj = MagicMock()
    monkeypatch.setattr(helm_interface, "run_entries_to_run_specs", mock_obj)
    return mock_obj


def test_run_executions_registers_huggingface(
    monkeypatch, monkeypatch_register_huggingface, monkeypatch_run_entries_to_run_specs
):
    # have to monkeypatch run_entries_to_runspecs since we can't register due to monkeypatching
    # register_huggingface_hub_model_from_flag_value
    runner = helm_interface.InProcessHelmRunner()

    runner.run([BbqHelmTest()], [HelmSut.FB_OPT_125M, HelmSut.GPT2])
    monkeypatch_register_huggingface.assert_called_once_with("facebook/opt-125m")


@pytest.mark.parametrize(
    "tests, suts, expected",
    [
        ([BbqHelmTest()], [HelmSut.FB_OPT_125M, HelmSut.GPT2], 20),
        ([BbqHelmTest()], [HelmSut.GPT2], 10),
    ],
)
def test_generates_correct_number_runspecs(monkeypatch, monkeypatch_run_entries_to_run_specs, tests, suts, expected):
    runner = helm_interface.InProcessHelmRunner()

    runner.run(tests, suts)
    assert len(monkeypatch_run_entries_to_run_specs.call_args[0][0]) == expected


def test_runs_run_all(monkeypatch, monkeypatch_run_all):
    runner = helm_interface.InProcessHelmRunner()

    runner.run([BbqHelmTest()], [HelmSut.GPT2])
    monkeypatch_run_all.assert_called_once()
