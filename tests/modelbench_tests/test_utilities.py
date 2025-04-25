import ctypes
import sys
from unittest.mock import MagicMock

import pytest
from dataclasses import dataclass
from itertools import groupby
from multiprocessing import Manager, Process

from modelbench.utilities import ConditionalPrometheus, NoOpMetric, ProgressTracker, group_by_key


@dataclass
class SomeClass:
    my_group: int
    value: int


def test_iterables_groupby():
    # This test demonstrates that itertools.groupby requires groups to be sorted.
    group_1_item_1 = SomeClass(my_group=1, value=1)
    group_1_item_2 = SomeClass(my_group=1, value=2)
    group_2_item_1 = SomeClass(my_group=2, value=1)
    group_2_item_2 = SomeClass(my_group=2, value=2)

    items = [
        # Not sorted by group
        group_1_item_1,
        group_2_item_1,
        group_1_item_2,
        group_2_item_2,
    ]
    groups = []
    for key, values in groupby(items, key=lambda c: c.my_group):
        groups.append((key, list(values)))
    # Shows that no grouping was performed.
    assert groups == [
        (1, [group_1_item_1]),
        (2, [group_2_item_1]),
        (1, [group_1_item_2]),
        (2, [group_2_item_2]),
    ]


def test_group_by_key():
    group_1_item_1 = SomeClass(my_group=1, value=1)
    group_1_item_2 = SomeClass(my_group=1, value=2)
    group_2_item_1 = SomeClass(my_group=2, value=1)
    group_2_item_2 = SomeClass(my_group=2, value=2)

    items = [
        # Not sorted by group
        group_1_item_1,
        group_2_item_1,
        group_1_item_2,
        group_2_item_2,
    ]
    groups = []
    for key, values in group_by_key(items, key=lambda c: c.my_group).items():
        groups.append((key, list(values)))
    assert groups == [
        (1, [group_1_item_1, group_1_item_2]),
        (2, [group_2_item_1, group_2_item_2]),
    ]


def test_progress_tracker(capsys):
    progress = ProgressTracker(total=4, print_updates=True)

    progress.increment()
    progress.increment()

    assert progress.complete_count.value == 2
    assert progress.progress == 0.5

    captured = capsys.readouterr()
    assert captured.out == '{"progress": 0.0}\n{"progress": 0.25}\n{"progress": 0.5}\n'


def worker(progress, num_updates):
    for _ in range(num_updates):
        progress.increment()


def test_progress_tracker_concurrency(capfd):
    with Manager() as manager:
        shared_count = manager.Value(ctypes.c_double, 0.0)
        lock = manager.Lock()
        progress = ProgressTracker(total=4, print_updates=True, shared_count=shared_count, lock=lock)
        processes = [
            Process(target=worker, args=(progress, 1)),
            Process(target=worker, args=(progress, 1)),
            Process(target=worker, args=(progress, 2)),
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        assert progress.complete_count.value == 4
        assert progress.progress == 1.0

        # Machine-readable progress updates are in correct order.
        captured = capfd.readouterr()
        assert (
            captured.out
            == '{"progress": 0.0}\n{"progress": 0.25}\n{"progress": 0.5}\n{"progress": 0.75}\n{"progress": 1.0}\n'
        )


def test_progress_tracker_invalid_total():
    with pytest.raises(AssertionError) as err_info:
        progress = ProgressTracker(total=0)

        assert str(err_info.value) == "Total must be a positive number, got 0."


class TestConditionalPrometheus:
    @pytest.fixture
    def mock_prometheus_client(self, monkeypatch):
        mock_module = MagicMock()
        mock_module.Counter = MagicMock()
        mock_module.Gauge = MagicMock()
        mock_module.Histogram = MagicMock()
        mock_module.Summary = MagicMock()
        mock_module.REGISTRY = MagicMock()
        mock_module.push_to_gateway = MagicMock()

        monkeypatch.setitem(sys.modules, "prometheus_client", mock_module)
        return mock_module

    @pytest.fixture
    def prometheus_env(self, monkeypatch):
        monkeypatch.setenv("PUSHGATEWAY_IP", "localhost")
        monkeypatch.setenv("PUSHGATEWAY_PORT", "9091")
        monkeypatch.setenv("MODELRUNNER_CONTAINER_NAME", "test-container")

    def test_uses_env_vars(self, prometheus_env, mock_prometheus_client):
        prometheus = ConditionalPrometheus(enabled=True)
        assert prometheus.enabled is True
        assert prometheus.pushgateway_ip == "localhost"
        assert prometheus.pushgateway_port == "9091"
        assert prometheus.job_name == "test-container"
        assert len(prometheus._metric_types) == 4

    def test_not_enabled_without_env_vars(self, mock_prometheus_client):
        prometheus = ConditionalPrometheus(enabled=True)
        assert prometheus.enabled is False

    def test_import_errors_disable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "prometheus_client", None)
        prometheus = ConditionalPrometheus(enabled=True)
        assert prometheus.enabled is False

    @pytest.mark.parametrize("metric", ["counter", "gauge", "histogram", "summary"])
    def test_disabled_uses_noop(self, metric):
        prometheus = ConditionalPrometheus(enabled=False)
        metric = getattr(prometheus, metric)(f"test_{metric}", f"Test {metric}")
        assert isinstance(metric, NoOpMetric)
        assert len(prometheus._metrics) == 0

    @pytest.mark.parametrize(
        "metric",
        [
            "counter",
            "gauge",
            "histogram",
            "summary",
        ],
    )
    def test_create_metric_enabled(self, prometheus_env, mock_prometheus_client, metric):
        prometheus = ConditionalPrometheus(enabled=True)
        mock_metric_class = getattr(mock_prometheus_client, metric.capitalize())
        metric1 = getattr(prometheus, metric)("test_metric", "Test metric")
        mock_metric_class.assert_called_once_with("test_metric", "Test metric")
        assert f"{metric}_test_metric" in prometheus._metrics
        metric2 = getattr(prometheus, metric)("test_metric", "Test metric")
        assert metric1 is metric2
        assert mock_metric_class.call_count == 1

    def test_push_metrics(self, prometheus_env, mock_prometheus_client):
        prometheus = ConditionalPrometheus(enabled=True)
        prometheus.push_metrics()

        mock_prometheus_client.push_to_gateway.assert_called_once_with(
            "localhost:9091", job="test-container", registry=mock_prometheus_client.REGISTRY
        )
