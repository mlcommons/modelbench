import os
import socket
import time


class NoOpMetric:
    def inc(self, *args, **kwargs):
        return self

    def dec(self, *args, **kwargs):
        return self

    def set(self, *args, **kwargs):
        return self

    def observe(self, *args, **kwargs):
        return self

    def time(self):
        class Timer:
            def __init__(self):
                self._start = time.perf_counter()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._duration = time.perf_counter() - self._start

        return Timer()

    def labels(self, *args, **kwargs):
        return self

    def state(self, *args, **kwargs):
        return self

    def info(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


class ConditionalPrometheus:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._metrics = {}
        self._metric_types = {k: NoOpMetric for k in ["counter", "gauge", "histogram", "summary", "info", "enum"]}

        self.pushgateway_ip = os.environ.get("PUSHGATEWAY_IP")
        self.pushgateway_port = os.environ.get("PUSHGATEWAY_PORT")
        self.job_name = os.environ.get("MODELRUNNER_CONTAINER_NAME", socket.gethostname())

        if not (self.pushgateway_ip and self.pushgateway_port):
            self.enabled = False

        if self.enabled:
            try:
                from prometheus_client import Counter, Gauge, Histogram, Summary, Info, Enum, REGISTRY, push_to_gateway

                self._metric_types = {
                    "counter": Counter,
                    "gauge": Gauge,
                    "histogram": Histogram,
                    "summary": Summary,
                    "info": Info,
                    "enum": Enum,
                }
                self._registry = REGISTRY
                self._push_to_gateway = push_to_gateway
            except ImportError:
                self.enabled = False

    def __getattr__(self, name):
        if name in self._metric_types:

            def metric_method(*args, **kwargs):
                if not self.enabled:
                    return NoOpMetric()

                metric_name = args[0] if args else kwargs.get("name")
                metric_key = f"{name}_{metric_name}"

                if metric_key not in self._metrics:
                    self._metrics[metric_key] = self._metric_types[name](*args, **kwargs)

                return self._metrics[metric_key]

            return metric_method

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def push_metrics(self):
        if not self.enabled:
            return None

        try:
            pushgateway_url = f"{self.pushgateway_ip}:{self.pushgateway_port}"
            self._push_to_gateway(pushgateway_url, job=self.job_name, registry=self._registry)
        except Exception as e:
            return e


PROMETHEUS = ConditionalPrometheus()
