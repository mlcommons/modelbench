import collections.abc
from abc import ABC, abstractmethod

import diskcache

from modelgauge.monitoring import PROMETHEUS

CACHE_GETS = PROMETHEUS.counter("mm_cache_gets", "Cache gets", ["name"])
CACHE_PUTS = PROMETHEUS.counter("mm_cache_puts", "Cache puts", ["name"])
CACHE_HITS = PROMETHEUS.counter("mm_cache_hits", "Cache hits", ["name"])
CACHE_SIZE = PROMETHEUS.gauge("mm_cache_size", "Cache size", ["name"])

MAX_CACHE_SIZE = 20 * 2**30  # 3 GB


class MBCache(ABC, collections.abc.Mapping):
    @abstractmethod
    def __setitem__(self, __key, __value):
        pass

    def __enter__(self):
        return self

    def __exit__(self, __type, __value, __traceback):
        pass


class NullCache(MBCache):
    """Doesn't save anything"""

    def __setitem__(self, __key, __value):
        pass

    def __getitem__(self, key, /):
        raise KeyError()

    def __len__(self):
        return 0

    def __iter__(self):
        pass


class InMemoryCache(MBCache):
    """Holds stuff in memory only"""

    def __init__(self):
        super().__init__()
        self.contents = dict()

    def __setitem__(self, __key, __value):
        self.contents.__setitem__(__key, __value)

    def __getitem__(self, key, /):
        return self.contents.__getitem__(key)

    def __len__(self):
        return self.contents.__len__()

    def __iter__(self):
        return self.contents.__iter__()


class DiskCache(MBCache):
    """
    Holds stuff in memory only. The docs recommend using
    it as a context manager in a threaded context:

    "Each thread that accesses a cache should also call close
    on the cache. Cache objects can be used in a with statement
    to safeguard calling close."

    """

    def __init__(self, cache_path):
        super().__init__()
        self.cache_path = cache_path
        self.cache_name = str(cache_path).split("/")[-1]
        if self.cache_name.endswith("_cache"):
            self.cache_name = self.cache_name[: -len("_cache")]
        self.raw_cache = diskcache.Cache(cache_path, size_limit=MAX_CACHE_SIZE)
        self.contents = self.raw_cache

    def __enter__(self):
        self.contents = self.raw_cache.__enter__()
        return self.contents

    def __exit__(self, __type, __value, __traceback):
        self.raw_cache.__exit__(__type, __value, __traceback)
        self.contents = self.raw_cache

    def __setitem__(self, __key, __value):
        CACHE_PUTS.labels(self.cache_name).inc()
        self.contents.__setitem__(__key, __value)
        CACHE_SIZE.labels(self.cache_name).set(self.__len__())

    def __getitem__(self, key, /):
        CACHE_GETS.labels(self.cache_name).inc()
        result = self.contents.__getitem__(key)
        if result:
            CACHE_HITS.labels(self.cache_name).inc()
        CACHE_SIZE.labels(self.cache_name).set(self.__len__())
        return result

    def __len__(self):
        return self.contents.__len__()

    def __iter__(self):
        return self.contents.__iter__()

    def __str__(self):
        return self.__class__.__name__ + f"({self.cache_path})"
