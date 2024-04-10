import pytest
from modelgauge.annotators.perspective_api import (
    ATTRIBUTE_THREAT,
    ATTRIBUTE_TOXICITY,
    PerspectiveAPIAnnotation,
    PerspectiveAPIAnnotator,
    PerspectiveDeveloperKey,
)
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from typing import Dict, List
from unittest.mock import patch


def _annotate_test_item(annotator, completion_text):
    prompt = PromptWithContext(prompt=TextPrompt(text="The prompt"), source_id=None)
    completion = SUTCompletion(text=completion_text)
    # TODO: Remove this and test on actual interface
    request = annotator.translate_request(prompt, completion)
    response = annotator.annotate(request)
    return annotator.translate_response(request, response)


def _batch_annotate_test_item(annotator, completion_texts: List[str]):
    # TODO: Remove this and test on actual interface
    requests = []
    for completion_text in completion_texts:
        prompt = PromptWithContext(prompt=TextPrompt(text="The prompt"), source_id=None)
        completion = SUTCompletion(text=completion_text)
        requests.append(annotator.translate_request(prompt, completion))
    batch_responses = annotator._batch_annotate(requests)
    assert len(requests) == len(batch_responses)
    responses = []
    for request, response in zip(requests, batch_responses):
        responses.append(annotator.translate_response(request, response))
    return responses


def _make_response(scores: Dict[str, float]) -> Dict:
    attribute_scores = {}
    for attribute, score in scores.items():
        attribute_scores[attribute] = {"summaryScore": {"value": score}}
    return {"attributeScores": attribute_scores}


class MockError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class FakeDiscoveryResource:
    """A fake version of discovery.Resource so we can do hermetic tests."""

    def __init__(self, batch_responses: List[List[Dict | MockError]]):
        self.batch_responses = batch_responses
        # These are set during the call.
        self.requests: List[List[Dict]] = [[] for _ in range(len(batch_responses))]
        self.batches_made = 0

    class FakeComments:
        def analyze(self, **kwargs):
            return kwargs

    def comments(self):
        return FakeDiscoveryResource.FakeComments()

    class FakeBatchRequest:
        def __init__(self, requests, responses):
            # Requests added here mutate the field in FakeDiscoveryResource.
            self.requests = requests
            self.responses = responses

        def add(self, **kwargs):
            self.requests.append(kwargs)

        def execute(self):
            assert len(self.requests) == len(
                self.responses
            ), "Didn't set up the right number of responses."
            for i, request in enumerate(self.requests):
                response = self.responses[i]
                error = None
                if isinstance(response, MockError):
                    error = response
                    response = {}
                request["callback"](request["request_id"], response, error)

            return None

    def new_batch_http_request(self):
        assert self.batches_made < len(
            self.batch_responses
        ), "Didn't set up the right number of batches"
        requests = self.requests[self.batches_made]
        responses = self.batch_responses[self.batches_made]
        self.batches_made += 1
        return FakeDiscoveryResource.FakeBatchRequest(requests, responses)


def test_perspective_api_single_prompt():
    responses = [_make_response({ATTRIBUTE_TOXICITY: 0.5})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = _annotate_test_item(annotator, "the text")

    assert result == PerspectiveAPIAnnotation(scores={"TOXICITY": 0.5})
    requests_made = [request["request"] for request in fake_client.requests[0]]
    assert requests_made == [
        {
            "body": {
                "comment": {"text": "the text"},
                "requestedAttributes": {"TOXICITY": {}},
            }
        }
    ]


def test_perspective_api_multiple_prompts():
    responses = [
        _make_response({ATTRIBUTE_TOXICITY: 0.1}),
        _make_response({ATTRIBUTE_TOXICITY: 0.2}),
    ]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = _batch_annotate_test_item(annotator, ["first", "second"])

    assert result == [
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.1}),
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.2}),
    ]
    requests_made = [request["request"] for request in fake_client.requests[0]]
    assert requests_made == [
        {
            "body": {
                "comment": {"text": "first"},
                "requestedAttributes": {"TOXICITY": {}},
            }
        },
        {
            "body": {
                "comment": {"text": "second"},
                "requestedAttributes": {"TOXICITY": {}},
            }
        },
    ]


def test_perspective_api_multiple_completions():
    responses = [
        _make_response({ATTRIBUTE_TOXICITY: 0.1}),
        _make_response({ATTRIBUTE_TOXICITY: 0.2}),
    ]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = _batch_annotate_test_item(annotator, ["first", "second"])

    assert result == [
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.1}),
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.2}),
    ]
    requests_made = [request["request"] for request in fake_client.requests[0]]
    assert requests_made == [
        {
            "body": {
                "comment": {"text": "first"},
                "requestedAttributes": {"TOXICITY": {}},
            }
        },
        {
            "body": {
                "comment": {"text": "second"},
                "requestedAttributes": {"TOXICITY": {}},
            }
        },
    ]


def test_perspective_api_multiple_prompts_with_multiple_completions():
    responses = [
        _make_response({ATTRIBUTE_TOXICITY: 0.1}),
        _make_response({ATTRIBUTE_TOXICITY: 0.2}),
        _make_response({ATTRIBUTE_TOXICITY: 0.3}),
        _make_response({ATTRIBUTE_TOXICITY: 0.4}),
    ]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = _batch_annotate_test_item(
        annotator, ["first", "second", "third", "fourth"]
    )

    assert result == [
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.1}),
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.2}),
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.3}),
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.4}),
    ]


def test_perspective_api_multiple_attributes():
    responses = [_make_response({ATTRIBUTE_TOXICITY: 0.1, ATTRIBUTE_THREAT: 0.2})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY, ATTRIBUTE_THREAT], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = _annotate_test_item(annotator, "the text")

    assert result == PerspectiveAPIAnnotation(scores={"TOXICITY": 0.1, "THREAT": 0.2})

    requests_made = [request["request"] for request in fake_client.requests[0]]
    assert requests_made == [
        {
            "body": {
                "comment": {"text": "the text"},
                "requestedAttributes": {"TOXICITY": {}, "THREAT": {}},
            }
        }
    ]


def test_perspective_api_single_prompt_fatal_error():
    responses = [MockError(403)]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    with pytest.raises(MockError) as err_info:
        _annotate_test_item(annotator, "the text")

    err_text = str(err_info.value)
    assert err_text == ("403")


@patch("time.sleep")
def test_perspective_api_single_prompt_retriable_error(sleep):
    batch_one = [MockError(503)]
    batch_two = [_make_response({ATTRIBUTE_TOXICITY: 0.5})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([batch_one, batch_two])
    annotator.client = fake_client

    result = _annotate_test_item(annotator, "the text")

    assert result == PerspectiveAPIAnnotation(scores={"TOXICITY": 0.5})
    requests_made = [request["request"] for request in fake_client.requests[1]]
    assert requests_made == [
        {
            "body": {
                "comment": {"text": "the text"},
                "requestedAttributes": {"TOXICITY": {}},
            }
        }
    ]
    assert sleep.call_count == 1


@patch("time.sleep")
def test_perspective_api_multiple_completions_retriable_error(sleep):
    batch_one = [_make_response({ATTRIBUTE_TOXICITY: 0.1}), MockError(503)]
    batch_two = [_make_response({ATTRIBUTE_TOXICITY: 0.2})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([batch_one, batch_two])
    annotator.client = fake_client

    result = _batch_annotate_test_item(annotator, ["works", "fails once"])

    assert result == [
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.1}),
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.2}),
    ]
    batch_one_requests_made = [
        request["request"] for request in fake_client.requests[0]
    ]
    assert batch_one_requests_made == [
        {
            "body": {
                "comment": {"text": "works"},
                "requestedAttributes": {"TOXICITY": {}},
            }
        },
        {
            "body": {
                "comment": {"text": "fails once"},
                "requestedAttributes": {"TOXICITY": {}},
            }
        },
    ]
    batch_two_requests_made = [
        request["request"] for request in fake_client.requests[1]
    ]
    assert batch_two_requests_made == [
        {
            "body": {
                "comment": {"text": "fails once"},
                "requestedAttributes": {"TOXICITY": {}},
            }
        },
    ]
    assert sleep.call_count == 1


def test_perspective_api_empty_comment_text_string():
    responses = [_make_response({ATTRIBUTE_TOXICITY: 0.1, ATTRIBUTE_THREAT: 0.2})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY, ATTRIBUTE_THREAT], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = _batch_annotate_test_item(annotator, ["", "the text"])

    assert result == [
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.0, "THREAT": 0.0}),
        PerspectiveAPIAnnotation(scores={"TOXICITY": 0.1, "THREAT": 0.2}),
    ]

    requests_made = [request["request"] for request in fake_client.requests[0]]
    assert len(requests_made) == 1
    assert requests_made[0] == {
        "body": {
            "comment": {"text": "the text"},
            "requestedAttributes": {"TOXICITY": {}, "THREAT": {}},
        }
    }


@patch("time.sleep")
def test_perspective_api_no_retries_retriable_error(sleep):
    responses = [MockError(503)]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value"), num_retries=0
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    with pytest.raises(MockError) as err_info:
        _annotate_test_item(annotator, "the text")

    err_text = str(err_info.value)
    assert err_text == ("503")
    assert sleep.call_count == 0


@patch("time.sleep")
def test_perspective_api_continuous_retriable_error(sleep):
    batches = [[MockError(503)], [MockError(503)]]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value"), num_retries=1
    )
    fake_client = FakeDiscoveryResource(batches)
    annotator.client = fake_client

    with pytest.raises(MockError) as err_info:
        _annotate_test_item(annotator, "the text")

    err_text = str(err_info.value)
    assert err_text == ("503")
    assert sleep.call_count == 1


@patch("time.sleep")
def test_perspective_api_many_retries_succeed(sleep):
    batches = [
        [MockError(503)],
        [MockError(503)],
        [MockError(503)],
        [MockError(503)],
        [_make_response({ATTRIBUTE_TOXICITY: 0.1})],
    ]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value"), num_retries=4
    )
    fake_client = FakeDiscoveryResource(batches)
    annotator.client = fake_client

    result = _annotate_test_item(annotator, "the text")

    assert result == PerspectiveAPIAnnotation(scores={"TOXICITY": 0.1})

    assert sleep.call_count == 4
    # Ensure each call sleeps longer than the previous one:
    for i in range(sleep.call_count - 1):
        assert sleep.call_args_list[i] < sleep.call_args_list[i + 1]
