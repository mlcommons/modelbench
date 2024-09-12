import os
import pytest
from modelgauge.caching import SqlDictCache
from pydantic import BaseModel
from tests.utilities import parent_directory
from sqlitedict import SqliteDict  # type: ignore


class SimpleClass(BaseModel):
    value: str


class ParentClass(BaseModel):
    parent_value: str


class ChildClass1(ParentClass):
    child_value: str


class ChildClass2(ParentClass):
    pass


def test_simple_request_serialization(tmpdir):
    with SqlDictCache(tmpdir, "sut_name") as cache:
        simple_request1 = SimpleClass(value="simple request 1")
        assert cache.get_cached_response(simple_request1) is None

        response = SimpleClass(value="simple response")
        cache.update_cache(simple_request1, response)

        simple_request2 = SimpleClass(value="simple request 2")
        assert cache.get_cached_response(simple_request2) is None


def test_simple_round_trip(tmpdir):
    with SqlDictCache(tmpdir, "sut_name") as cache:
        request = SimpleClass(value="simple request")
        assert cache.get_cached_response(request) is None

        response = SimpleClass(value="simple response")
        cache.update_cache(request, response)
        returned_response = cache.get_cached_response(request)
        assert returned_response == response


def test_simple_round_trip_dicts(tmpdir):
    with SqlDictCache(tmpdir, "sut_name") as cache:
        request = {"some-key": "some-value"}
        assert cache.get_cached_response(request) is None

        response = {"value": "some-response"}
        cache.update_cache(request, response)
        returned_response = cache.get_cached_response(request)
        assert returned_response == response


def test_request_cannot_cache(tmpdir):
    with SqlDictCache(tmpdir, "sut_name") as cache:
        request = 14
        response = SimpleClass(value="simple response")
        cache.update_cache(request, response)
        # Not stored, but also no error.
        assert cache.get_cached_response(request) is None


def test_response_cannot_cache(tmpdir):
    with SqlDictCache(tmpdir, "sut_name") as cache:
        request = SimpleClass(value="simple request")
        response = 14
        cache.update_cache(request, response)
        # Not stored, but also no error.
        assert cache.get_cached_response(request) is None


def test_polymorphic_request(tmpdir):
    with SqlDictCache(tmpdir, "sut_name") as cache:
        parent_request = ParentClass(parent_value="parent")
        parent_response = SimpleClass(value="parent response")
        cache.update_cache(parent_request, parent_response)

        child1_request = ChildClass1(parent_value="parent 1", child_value="child 1")
        assert cache.get_cached_response(child1_request) is None
        child1_response = SimpleClass(value="child 1 response")
        cache.update_cache(child1_request, child1_response)

        child2_request = ChildClass2(parent_value="parent")
        assert cache.get_cached_response(child2_request) is None
        child2_response = SimpleClass(value="child 2 response")
        cache.update_cache(child2_request, child2_response)

        assert cache.get_cached_response(parent_request) == parent_response
        assert cache.get_cached_response(child1_request) == child1_response
        assert cache.get_cached_response(child1_request) != child2_response
        assert cache.get_cached_response(child2_request) == child2_response
        assert cache.get_cached_response(child2_request) != parent_response


def test_cache_update(tmpdir):
    with SqlDictCache(tmpdir, "sut_name") as cache:
        request = SimpleClass(value="val")
        cache.update_cache(request, SimpleClass(value="response 1"))
        new_response = SimpleClass(value="response 2")
        cache.update_cache(request, new_response)
        assert cache.get_cached_response(request) == new_response


def test_polymorphic_response(tmpdir):
    with SqlDictCache(tmpdir, "sut_name") as cache:
        parent_request = SimpleClass(value="parent request")
        parent_response = ParentClass(parent_value="parent")
        cache.update_cache(parent_request, parent_response)

        child1_request = SimpleClass(value="child 1 request")
        child1_response = ChildClass1(parent_value="parent", child_value="child")
        cache.update_cache(child1_request, child1_response)

        child2_request = SimpleClass(value="child 2 request")
        child2_response = ChildClass2(parent_value="parent")  # Same value as parent
        cache.update_cache(child2_request, child2_response)

        assert cache.get_cached_response(parent_request) == parent_response
        assert cache.get_cached_response(child1_request) == child1_response
        assert cache.get_cached_response(child1_request) != child2_response
        assert cache.get_cached_response(child2_request) == child2_response
        assert cache.get_cached_response(child2_request) != parent_response


def test_slashes_in_file_identifier(tmpdir):
    with SqlDictCache(tmpdir, "sut/name") as cache:
        request = SimpleClass(value="val")
        response = SimpleClass(value="response")
        cache.update_cache(request, response)
        assert cache.get_cached_response(request) == response


def test_non_existing_directory(tmpdir):
    """Tests that the directory given to SUTResponseCache is created if it does not already exist."""
    cache_dir = os.path.join(tmpdir, "data", "new_dir")
    assert not os.path.exists(cache_dir)
    request = SimpleClass(value="request")
    response = SimpleClass(value="response")
    # Create new cache
    with SqlDictCache(cache_dir, "sample") as cache:
        assert len(cache.cached_responses) == 0
        cache.update_cache(request, response)
    # Confirm the cache persists.
    with SqlDictCache(cache_dir, "sample") as cache:
        assert len(cache.cached_responses) == 1
        assert cache.get_cached_response(request) == response


def test_fails_on_unexpected_table(tmpdir):
    cache_location = os.path.join(tmpdir, "sample_cache.sqlite")
    SqliteDict(cache_location, tablename="some_table")
    with pytest.raises(AssertionError) as err_info:
        SqlDictCache(tmpdir, "sample")
    assert "Expected only table to be v1, but found ['some_table', 'v1']" in str(
        err_info.value
    )
    assert "sample_cache.sqlite" in str(err_info.value)


@pytest.mark.skip(reason="Comment out this skip to rebuild the cache file.")
def test_rewrite_sample_cache(parent_directory):
    cache_dir = str(parent_directory.joinpath("data"))
    os.remove(os.path.join(cache_dir, "sample_cache.sqlite"))
    with SqlDictCache(cache_dir, "sample") as cache:
        cache.update_cache(
            SimpleClass(value="request 1"), ParentClass(parent_value="response 1")
        )
        cache.update_cache(
            SimpleClass(value="request 2"),
            ChildClass1(parent_value="response 2", child_value="child val"),
        )


def test_format_stability(parent_directory):
    """Reads from existing sample_cache.sqlite and checks deserialization."""
    cache_dir = str(parent_directory.joinpath("data"))
    with SqlDictCache(cache_dir, "sample") as cache:
        assert len(cache.cached_responses) == 2
        response_1 = cache.get_cached_response(SimpleClass(value="request 1"))
        assert isinstance(response_1, ParentClass)
        assert response_1.parent_value == "response 1"
        response_2 = cache.get_cached_response(SimpleClass(value="request 2"))
        assert isinstance(response_2, ChildClass1)
        assert response_2.parent_value == "response 2"
        assert response_2.child_value == "child val"


class CallCounter:
    def __init__(self, response):
        self.response = response
        self.counter = 0

    def some_call(self, request):
        self.counter += 1
        return self.response


def test_get_or_call(tmpdir):
    request = SimpleClass(value="simple request")
    response = SimpleClass(value="simple response")
    mock_evaluate = CallCounter(response)
    with SqlDictCache(tmpdir, "sut_name") as cache:
        assert cache.get_or_call(request, mock_evaluate.some_call) == response
        assert mock_evaluate.counter == 1

        # Call again, this time it shouldn't call `some_call`
        assert cache.get_or_call(request, mock_evaluate.some_call) == response
        assert mock_evaluate.counter == 1


def test_unencodable_request(tmpdir):
    request = "some-request"
    response = SimpleClass(value="some-response")
    mock_evaluate = CallCounter(response)
    with SqlDictCache(tmpdir, "sut_name") as cache:
        assert cache.get_or_call(request, mock_evaluate.some_call) == response
        assert mock_evaluate.counter == 1

        # We should not get a cache hit because we can't cache the request
        assert cache.get_or_call(request, mock_evaluate.some_call) == response
        assert mock_evaluate.counter == 2


def test_unencodable_response(tmpdir):
    request = SimpleClass(value="some-request")
    response = "some-response"
    mock_evaluate = CallCounter(response)
    with SqlDictCache(tmpdir, "sut_name") as cache:
        assert cache.get_or_call(request, mock_evaluate.some_call) == response
        assert mock_evaluate.counter == 1

        # We should not get a cache hit because we can't cache the response
        assert cache.get_or_call(request, mock_evaluate.some_call) == response
        assert mock_evaluate.counter == 2
