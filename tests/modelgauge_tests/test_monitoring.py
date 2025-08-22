import sys
from unittest.mock import MagicMock

import pytest

from modelgauge.monitoring import ConditionalPrometheus, NoOpMetric


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
        assert len(prometheus._metric_types) == 6

    def test_not_enabled_without_env_vars(self, mock_prometheus_client):
        prometheus = ConditionalPrometheus(enabled=True)
        assert prometheus.enabled is False

    def test_import_errors_disable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "prometheus_client", None)
        prometheus = ConditionalPrometheus(enabled=True)
        assert prometheus.enabled is False

    @pytest.mark.parametrize("metric", ["counter", "gauge", "histogram", "summary", "info", "enum"])
    def test_disabled_uses_noop(self, metric):
        prometheus = ConditionalPrometheus(enabled=False)
        metric = getattr(prometheus, metric)(f"test_{metric}", f"Test {metric}")
        assert isinstance(metric, NoOpMetric)
        assert len(prometheus._metrics) == 0

    @pytest.mark.parametrize(
        "metric",
        ["counter", "gauge", "histogram", "summary", "info", "enum"],
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
