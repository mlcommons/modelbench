import hashlib
from modelgauge.caching import SqlDictCache, TypedData, CacheEntry


def patched_get_or_call(self: SqlDictCache, request, callable):
    # Check if we can find the response using the old hash
    old_response = patched_get_cached_response(self, request)
    if old_response is not None:
        # Store it back using the new format
        self.update_cache(request, old_response)
        # Clear out the old version
        del self.cached_responses[patched_hash_request(request)]
        self.cached_responses.commit()
        return old_response

    # Return to normal behavior
    response = self.get_cached_response(request)
    if response is not None:
        return response
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


def patched_hash_request(request) -> str:
    typed_data = TypedData.from_instance(request)
    typed_data.module = typed_data.module.replace("modelgauge", "newhelm")
    return hashlib.sha256(typed_data.model_dump_json().encode()).hexdigest()


def patched_decode_response(encoded_response: str):
    entry = CacheEntry.model_validate_json(encoded_response)
    entry.payload.module = entry.payload.module.replace("newhelm", "modelgauge")
    return entry.payload.to_instance()


def apply_patch():
    SqlDictCache.get_or_call = patched_get_or_call
