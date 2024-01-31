from typing import Dict, List
from newhelm.placeholders import Prompt
from newhelm.annotators.perspective_api import (
    ATTRIBUTE_THREAT,
    ATTRIBUTE_TOXICITY,
    PerspectiveAPIAnnotator,
    PerspectiveAPIAnnotation,
)
from newhelm.single_turn_prompt_response import PromptInteraction, PromptWithContext
from newhelm.sut import SUTCompletion, SUTResponse

Interaction = PerspectiveAPIAnnotation.Interaction
Completion = PerspectiveAPIAnnotation.Completion


def _make_interaction(completions: List[str]) -> PromptInteraction:
    sut_completions = [SUTCompletion(text) for text in completions]
    return PromptInteraction(
        PromptWithContext(Prompt("The prompt")),
        SUTResponse(sut_completions),
    )


def _make_response(scores: Dict[str, float]) -> Dict:
    attribute_scores = {}
    for attribute, score in scores.items():
        attribute_scores[attribute] = {"summaryScore": {"value": score}}
    return {"attributeScores": attribute_scores}


class FakeDiscoveryResource:
    """A fake version of discovery.Resource so we can do hermetic tests."""

    def __init__(self, responses: List[Dict]):
        self.responses = responses
        # These are set during the call.
        self.requests: List[Dict] = []

    class FakeComments:
        def analyze(self, **kwargs):
            return kwargs

    def comments(self):
        return FakeDiscoveryResource.FakeComments()

    class FakeBatchRequest:
        def __init__(self, outer_fake):
            self.outer_fake = outer_fake

        def add(self, **kwargs):
            self.outer_fake.requests.append(kwargs)

        def execute(self):
            assert len(self.outer_fake.requests) == len(
                self.outer_fake.responses
            ), "Didn't set up the right number of responses."
            for i, request in enumerate(self.outer_fake.requests):
                request["callback"](str(i), self.outer_fake.responses[i], None)
            return None

    def new_batch_http_request(self):
        return FakeDiscoveryResource.FakeBatchRequest(self)


def test_perspective_api_single_prompt():
    interactions = [_make_interaction(["the text"])]
    responses = [_make_response({ATTRIBUTE_TOXICITY: 0.5})]
    annotator = PerspectiveAPIAnnotator([ATTRIBUTE_TOXICITY])
    fake_client = FakeDiscoveryResource(responses)
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        [Interaction([Completion({"TOXICITY": 0.5})])]
    )
    requests_made = [request["request"] for request in fake_client.requests]
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
    annotator = PerspectiveAPIAnnotator([ATTRIBUTE_TOXICITY])
    fake_client = FakeDiscoveryResource(responses)
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        [
            Interaction([Completion({"TOXICITY": 0.1})]),
            Interaction([Completion({"TOXICITY": 0.2})]),
        ]
    )
    requests_made = [request["request"] for request in fake_client.requests]
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
    annotator = PerspectiveAPIAnnotator([ATTRIBUTE_TOXICITY])
    fake_client = FakeDiscoveryResource(responses)
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        [Interaction([Completion({"TOXICITY": 0.1}), Completion({"TOXICITY": 0.2})])]
    )
    requests_made = [request["request"] for request in fake_client.requests]
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
    annotator = PerspectiveAPIAnnotator([ATTRIBUTE_TOXICITY])
    fake_client = FakeDiscoveryResource(responses)
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        [
            Interaction([Completion({"TOXICITY": 0.1}), Completion({"TOXICITY": 0.2})]),
            Interaction([Completion({"TOXICITY": 0.3}), Completion({"TOXICITY": 0.4})]),
        ]
    )


def test_perspective_api_multiple_attributes():
    interactions = [_make_interaction(["the text"])]
    responses = [_make_response({ATTRIBUTE_TOXICITY: 0.1, ATTRIBUTE_THREAT: 0.2})]
    annotator = PerspectiveAPIAnnotator([ATTRIBUTE_TOXICITY, ATTRIBUTE_THREAT])
    fake_client = FakeDiscoveryResource(responses)
    annotator.client = fake_client

    result = annotator.annotate_test_item(interactions)

    assert result == PerspectiveAPIAnnotation(
        [Interaction([Completion({"TOXICITY": 0.1, "THREAT": 0.2})])]
    )
    requests_made = [request["request"] for request in fake_client.requests]
    assert requests_made == [
        {
            "body": {
                "comment": {"text": "the text"},
                "requestedAttributes": {"TOXICITY": {}, "THREAT": {}},
            }
        }
    ]
