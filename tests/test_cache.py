import os
from pydantic import BaseModel

from newhelm.cache_helper import SUTResponseCache
from tests.utilities import parent_directory


class SimpleClass(BaseModel):
    value: str


class ParentClass(BaseModel):
    parent_value: str


class ChildClass1(ParentClass):
    child_value: str


class ChildClass2(ParentClass):
    pass


def test_simple_request_serialization(tmpdir):
    with SUTResponseCache(tmpdir, "sut_name") as cache:
        simple_request1 = SimpleClass(value="simple request 1")
        assert cache.get_cached_response(simple_request1) is None

        response = SimpleClass(value="simple response")
        cache.update_cache(simple_request1, response)

        simple_request2 = SimpleClass(value="simple request 2")
        assert cache.get_cached_response(simple_request2) is None


def test_simple_round_trip(tmpdir):
    with SUTResponseCache(tmpdir, "sut_name") as cache:
        request = SimpleClass(value="simple request")
        assert cache.get_cached_response(request) is None

        response = SimpleClass(value="simple response")
        cache.update_cache(request, response)
        returned_response = cache.get_cached_response(request)
        assert returned_response == response


def test_polymorphic_request(tmpdir):
    with SUTResponseCache(tmpdir, "sut_name") as cache:
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
    with SUTResponseCache(tmpdir, "sut_name") as cache:
        request = SimpleClass(value="val")
        cache.update_cache(request, SimpleClass(value="response 1"))
        new_response = SimpleClass(value="response 2")
        cache.update_cache(request, new_response)
        assert cache.get_cached_response(request) == new_response


def test_polymorphic_response(tmpdir):
    with SUTResponseCache(tmpdir, "sut_name") as cache:
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


def test_non_exisiting_directory(parent_directory):
    """Tests that the directory given to SUTResponseCache is created if it does not already exist."""
    cache_dir = str(parent_directory.joinpath("data", "new_dir"))
    assert not os.path.exists(cache_dir)
    request = SimpleClass(value="request")
    response = SimpleClass(value="response")
    # Create new cache
    with SUTResponseCache(cache_dir, "sample_cache") as cache:
        assert len(cache.cached_responses) == 0
        cache.update_cache(request, response)
    # Confirm the cache persists.
    with SUTResponseCache(cache_dir, "sample_cache") as cache:
        assert len(cache.cached_responses) == 1
        assert cache.get_cached_response(request) == response
    # Delete newly-created cache file and directory
    os.remove(os.path.join(cache_dir, "sample_cache.sqlite"))
    os.rmdir(cache_dir)


def test_format_stability(parent_directory):
    """Reads from existing sample_cache.sqlite and checks deserialization."""
    cache_dir = str(parent_directory.joinpath("data"))
    with SUTResponseCache(cache_dir, "sample_cache") as cache:
        assert len(cache.cached_responses) == 2
        response_1 = cache.get_cached_response(SimpleClass(value="request 1"))
        assert isinstance(response_1, ParentClass)
        assert response_1.parent_value == "response 1"
        response_2 = cache.get_cached_response(SimpleClass(value="request 2"))
        assert isinstance(response_2, ChildClass1)
        assert response_2.parent_value == "response 2"
        assert response_2.child_value == "child val"
