import inspect
from typing import Dict, List
from unittest.mock import patch

import pytest
from newhelm.prompt import TextPrompt
from newhelm.annotators.perspective_api import (
    ATTRIBUTE_THREAT,
    ATTRIBUTE_TOXICITY,
    PerspectiveAPIAnnotator,
    PerspectiveAPIAnnotation,
    PerspectiveDeveloperKey,
)
from newhelm.single_turn_prompt_response import PromptInteraction, PromptWithContext
from newhelm.sut import SUTCompletion, SUTResponse

Interaction = PerspectiveAPIAnnotation.Interaction
Completion = PerspectiveAPIAnnotation.Completion


def _make_interaction(completions: List[str]) -> PromptInteraction:
    sut_completions = [SUTCompletion(text=text) for text in completions]
    return PromptInteraction(
        prompt=PromptWithContext(prompt=TextPrompt(text="The prompt")),
        response=SUTResponse(completions=sut_completions),
    )


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
    interactions = [_make_interaction(["the text"])]
    responses = [_make_response({ATTRIBUTE_TOXICITY: 0.5})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        interaction=[Interaction(completions=[Completion(scores={"TOXICITY": 0.5})])]
    )
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
    interactions = [_make_interaction(["first"]), _make_interaction(["second"])]
    responses = [
        _make_response({ATTRIBUTE_TOXICITY: 0.1}),
        _make_response({ATTRIBUTE_TOXICITY: 0.2}),
    ]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        interaction=[
            Interaction(completions=[Completion(scores={"TOXICITY": 0.1})]),
            Interaction(completions=[Completion(scores={"TOXICITY": 0.2})]),
        ]
    )
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
    interactions = [_make_interaction(["first", "second"])]
    responses = [
        _make_response({ATTRIBUTE_TOXICITY: 0.1}),
        _make_response({ATTRIBUTE_TOXICITY: 0.2}),
    ]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        interaction=[
            Interaction(
                completions=[
                    Completion(scores={"TOXICITY": 0.1}),
                    Completion(scores={"TOXICITY": 0.2}),
                ]
            )
        ]
    )
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
    interactions = [
        _make_interaction(["first", "second"]),
        _make_interaction(["third", "fourth"]),
    ]
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

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        interaction=[
            Interaction(
                completions=[
                    Completion(scores={"TOXICITY": 0.1}),
                    Completion(scores={"TOXICITY": 0.2}),
                ]
            ),
            Interaction(
                completions=[
                    Completion(scores={"TOXICITY": 0.3}),
                    Completion(scores={"TOXICITY": 0.4}),
                ]
            ),
        ]
    )


def test_perspective_api_multiple_attributes():
    interactions = [_make_interaction(["the text"])]
    responses = [_make_response({ATTRIBUTE_TOXICITY: 0.1, ATTRIBUTE_THREAT: 0.2})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY, ATTRIBUTE_THREAT], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        interaction=[
            Interaction(
                completions=[Completion(scores={"TOXICITY": 0.1, "THREAT": 0.2})]
            )
        ]
    )
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
    interactions = [_make_interaction(["the text"])]
    responses = [MockError(403)]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    with pytest.raises(MockError) as err_info:
        annotator.annotate_test_item(interactions)

    err_text = str(err_info.value)
    assert err_text == ("403")


@patch("time.sleep")
def test_perspective_api_single_prompt_retriable_error(sleep):
    interactions = [_make_interaction(["the text"])]
    batch_one = [MockError(503)]
    batch_two = [_make_response({ATTRIBUTE_TOXICITY: 0.5})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([batch_one, batch_two])
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        interaction=[Interaction(completions=[Completion(scores={"TOXICITY": 0.5})])]
    )
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
    interactions = [_make_interaction(["works", "fails once"])]
    batch_one = [_make_response({ATTRIBUTE_TOXICITY: 0.1}), MockError(503)]
    batch_two = [_make_response({ATTRIBUTE_TOXICITY: 0.2})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([batch_one, batch_two])
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        interaction=[
            Interaction(
                completions=[
                    Completion(scores={"TOXICITY": 0.1}),
                    Completion(scores={"TOXICITY": 0.2}),
                ]
            )
        ]
    )
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
    interactions = [_make_interaction([""]), _make_interaction(["the text"])]
    responses = [_make_response({ATTRIBUTE_TOXICITY: 0.1, ATTRIBUTE_THREAT: 0.2})]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY, ATTRIBUTE_THREAT], PerspectiveDeveloperKey("some-value")
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        interaction=[
            Interaction(
                completions=[Completion(scores={"TOXICITY": 0.0, "THREAT": 0.0})]
            ),
            Interaction(
                completions=[Completion(scores={"TOXICITY": 0.1, "THREAT": 0.2})]
            ),
        ]
    )

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
    interactions = [_make_interaction(["the text"])]
    responses = [MockError(503)]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value"), num_retries=0
    )
    fake_client = FakeDiscoveryResource([responses])
    annotator.client = fake_client

    with pytest.raises(MockError) as err_info:
        annotator.annotate_test_item(interactions)

    err_text = str(err_info.value)
    assert err_text == ("503")
    assert sleep.call_count == 0


@patch("time.sleep")
def test_perspective_api_continuous_retriable_error(sleep):
    interactions = [_make_interaction(["the text"])]
    batches = [[MockError(503)], [MockError(503)]]
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY], PerspectiveDeveloperKey("some-value"), num_retries=1
    )
    fake_client = FakeDiscoveryResource(batches)
    annotator.client = fake_client

    with pytest.raises(MockError) as err_info:
        annotator.annotate_test_item(interactions)

    err_text = str(err_info.value)
    assert err_text == ("503")
    assert sleep.call_count == 1


@patch("time.sleep")
def test_perspective_api_many_retries_succeed(sleep):
    interactions = [_make_interaction(["the text"])]
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

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        interaction=[
            Interaction(
                completions=[
                    Completion(scores={"TOXICITY": 0.1}),
                ]
            )
        ]
    )
    assert sleep.call_count == 4
    # Ensure each call sleeps longer than the previous one:
    for i in range(sleep.call_count - 1):
        assert sleep.call_args_list[i] < sleep.call_args_list[i + 1]
