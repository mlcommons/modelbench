import os
from sqlitedict import SqliteDict  # type: ignore

from newhelm.typed_data import TypedData


class SUTResponseCache:
    """When caching is enabled, use previously cached SUT responses.

    When used, the local directory structure will look like this:
    data_dir/
        test_1/
            cached_responses/
                sut_1.sqlite
                sut_2.sqlite
        ...
      ...
    """

    def __init__(self, data_dir, sut_name):
        self.data_dir = data_dir
        self.fname = f"{sut_name}.sqlite"
        self.cached_responses = self._load_cached_responses()

    def __enter__(self):
        self.cached_responses.__enter__()
        return self

    def __exit__(self, *exc_info):
        self.cached_responses.close()

    def get_cached_response(self, request):
        encoded_request = self._encode_request(request)
        encoded_response = self.cached_responses.get(encoded_request)
        if encoded_response:
            return self._decode_response(encoded_response)
        else:
            return None

    def update_cache(self, request, response):
        encoded_request = self._encode_request(request)
        encoded_response = self._encode_response(response)
        self.cached_responses[encoded_request] = encoded_response
        self.cached_responses.commit()

    def _load_cached_responses(self):
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, self.fname)
        return SqliteDict(path)

    def _encode_response(self, response) -> TypedData:
        return TypedData.from_instance(response)

    def _decode_response(self, encoded_response: TypedData):
        return encoded_response.to_instance()

    def _encode_request(self, request) -> str:
        return TypedData.from_instance(request).model_dump_json()

    def _decode_request(self, request_json: str):
        return TypedData.model_validate_json(request_json).to_instance()
