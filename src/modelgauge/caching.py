import hashlib
import json
import os
from abc import ABC, abstractmethod
from modelgauge.general import normalize_filename
from modelgauge.typed_data import Typeable, TypedData, is_typeable
from pydantic import BaseModel
from sqlitedict import SqliteDict  # type: ignore


class Cache(ABC):
    """Interface for caching."""

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, *exc_info):
        pass

    @abstractmethod
    def get_or_call(self, request, callable):
        pass

    @abstractmethod
    def get_cached_response(self, request):
        pass

    @abstractmethod
    def update_cache(self, request, response):
        pass


class CacheEntry(BaseModel):
    """Wrapper around the data we write to the cache."""

    payload: TypedData


class SqlDictCache(Cache):
    """Cache the response from a method using the request as the key.

    Will create a `file_identifier`_cache.sqlite file in `data_dir` to persist
    the cache.
    """

    _CACHE_SCHEMA_VERSION = "v1"
    """Version is encoded in the table name to identify the schema."""

    def __init__(self, data_dir, file_identifier):
        os.makedirs(data_dir, exist_ok=True)
        fname = normalize_filename(f"{file_identifier}_cache.sqlite")
        path = os.path.join(data_dir, fname)
        self.cached_responses = SqliteDict(
            path,
            tablename=self._CACHE_SCHEMA_VERSION,
            encode=json.dumps,
            decode=json.loads,
        )
        tables = SqliteDict.get_tablenames(path)
        assert tables == [self._CACHE_SCHEMA_VERSION], (
            f"Expected only table to be {self._CACHE_SCHEMA_VERSION}, "
            f"but found {tables} in {path}."
        )

    def __enter__(self):
        self.cached_responses.__enter__()
        return self

    def __exit__(self, *exc_info):
        self.cached_responses.close()

    def get_or_call(self, request, callable):
        """Return the cached value, otherwise cache calling `callable`"""
        response = self.get_cached_response(request)
        if response is not None:
            return response
        response = callable(request)
        self.update_cache(request, response)
        return response

    def get_cached_response(self, request):
        """Return the cached value, or None if `request` is not in the cache."""
        if not self._can_encode(request):
            return None
        cache_key = self._hash_request(request)
        encoded_response = self.cached_responses.get(cache_key)
        if encoded_response:
            return self._decode_response(encoded_response)
        else:
            return None

    def update_cache(self, request, response: Typeable):
        """Save `response` in the cache, keyed by `request`."""
        if not self._can_encode(request) or not self._can_encode(response):
            return
        cache_key = self._hash_request(request)
        encoded_response = self._encode_response(response)
        self.cached_responses[cache_key] = encoded_response
        self.cached_responses.commit()

    def _can_encode(self, obj) -> bool:
        # Encoding currently requires Pydanic objects.
        return is_typeable(obj)

    def _encode_response(self, response: Typeable) -> str:
        return CacheEntry(payload=TypedData.from_instance(response)).model_dump_json()

    def _decode_response(self, encoded_response: str):
        return CacheEntry.model_validate_json(encoded_response).payload.to_instance()

    def _hash_request(self, request) -> str:
        return hashlib.sha256(
            TypedData.from_instance(request).model_dump_json().encode()
        ).hexdigest()


class NoCache(Cache):
    """Implements the caching interface, but never actually caches."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def get_or_call(self, request, callable):
        return callable(request)

    def get_cached_response(self, request):
        return None

    def update_cache(self, request, response):
        pass
