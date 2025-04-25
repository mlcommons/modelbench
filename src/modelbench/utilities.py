import ctypes
import json
import os
import socket
from collections import defaultdict
from multiprocessing import Value
from typing import Callable, Mapping, Sequence


def group_by_key(items: Sequence, key=Callable) -> Mapping:
    """Group `items` by the value returned by `key(item)`."""
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return groups


class ProgressTracker:
    def __init__(self, total, print_updates=False, shared_count=None, lock=None):
        assert total > 0, f"Total must be a positive number, got {total}."
        self.complete_count = shared_count or Value(ctypes.c_double, 0.0)
        self.total = total
        self.print_updates = print_updates
        self.lock = lock
        if self.print_updates:
            # Print initial progress json.
            self.print_progress_json()

    @property
    def progress(self):
        return self.complete_count.value / self.total

    def print_progress_json(self):
        # using json.dumps instead of just print so the output is consumable as json
        print(json.dumps({"progress": self.progress}))

    def increment(self):
        if self.lock:
            with self.lock:
                self._increment()
        else:
            self._increment()

    def _increment(self):
        self.complete_count.value += 1.0
        if self.print_updates:
            self.print_progress_json()


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
        return self

    def labels(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


class ConditionalPrometheus:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._metrics = {}
        self._metric_types = {k: NoOpMetric for k in ["counter", "gauge", "histogram", "summary"]}

        self.pushgateway_ip = os.environ.get("PUSHGATEWAY_IP")
        self.pushgateway_port = os.environ.get("PUSHGATEWAY_PORT")
        self.job_name = os.environ.get("MODELRUNNER_CONTAINER_NAME", socket.gethostname())

        if not (self.pushgateway_ip and self.pushgateway_port):
            self.enabled = False

        if self.enabled:
            try:
                from prometheus_client import Counter, Gauge, Histogram, Summary, REGISTRY, push_to_gateway

                self._metric_types = {"counter": Counter, "gauge": Gauge, "histogram": Histogram, "summary": Summary}
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
