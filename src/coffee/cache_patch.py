import hashlib
from modelgauge.caching import SqlDictCache, TypedData, CacheEntry


def patched_get_or_call(self: SqlDictCache, request, callable):
    response = self.get_cached_response(request)
    if response is not None:
        return response
    # Check if we can find the response using the old hash
    response = patched_get_cached_response(self, request)
    if response is not None:
        # Store it back using the new format
        patched_update_cache(self, request, response)
        return response
    # Return to normal behavior
    response = callable(request)
    self.update_cache(request, response)
    return response


def patched_get_cached_response(self: SqlDictCache, request):
    if not self._can_encode(request):
        return None
    cache_key = patched_hash_request(request)
    encoded_response = self.cached_responses.get(cache_key)
    if encoded_response:
        return patched_decode_response(encoded_response)
    else:
        return None


def patched_update_cache(self: SqlDictCache, request: SqlDictCache, response):
    """Save `response` in the cache, keyed by `request`."""
    if not self._can_encode(request) or not self._can_encode(response):
        return
    # Hash using the unmodified version of the request
    cache_key = self._hash_request(request)
    # Convert the response to the new version
    encoded_response = patched_encode_response(response)
    self.cached_responses[cache_key] = encoded_response
    self.cached_responses.commit()


def patched_hash_request(request) -> str:
    typed_data = TypedData.from_instance(request)
    typed_data.module = typed_data.module.replace("modelgauge", "newhelm")
    return hashlib.sha256(typed_data.model_dump_json().encode()).hexdigest()


def patched_decode_response(encoded_response: str):
    entry = CacheEntry.model_validate_json(encoded_response)
    entry.payload.module = entry.payload.module.replace("newhelm", "modelgauge")
    return entry.payload.to_instance()


def patched_encode_response(response) -> str:
    typed_data = TypedData.from_instance(response)
    typed_data.module = typed_data.module.replace("newhelm", "modelgauge")
    return CacheEntry(payload=TypedData.from_instance(response)).model_dump_json()


def apply_patch():
    SqlDictCache.get_or_call = patched_get_or_call
