from dataclasses import dataclass
import threading
from typing import Dict, List, Mapping, Optional, Sequence
from googleapiclient import discovery  # type: ignore[import-untyped]
from googleapiclient.http import BatchHttpRequest  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from newhelm.annotation import Annotation

from newhelm.base_annotator import BaseAnnotator
from newhelm.secrets_registry import SECRETS
from newhelm.single_turn_prompt_response import PromptInteraction, PromptWithContext
from newhelm.sut import SUTCompletion

# Constants for requesting specific kinds of scoring from Perspective API.
# https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages
ATTRIBUTE_TOXICITY = "TOXICITY"
ATTRIBUTE_SEVERE_TOXICITY = "SEVERE_TOXICITY"
ATTRIBUTE_IDENTITY_ATTACK = "IDENTITY_ATTACK"
ATTRIBUTE_INSULT = "INSULT"
ATTRIBUTE_PROFANITY = "PROFANITY"
ATTRIBUTE_THREAT = "THREAT"
ATTRIBUTE_SEXUALLY_EXPLICIT = "SEXUALLY_EXPLICIT"
ATTRIBUTE_FLIRTATION = "FLIRTATION"


@dataclass(frozen=True)
class PerspectiveAPIAnnotation(Annotation):
    @dataclass(frozen=True)
    class Completion:
        scores: Mapping[str, float]
        """Mapping from ATTRIBUTE to score."""

    @dataclass(frozen=True)
    class Interaction:
        completions: Sequence["PerspectiveAPIAnnotation.Completion"]
        """Each element corresponds to a completion to a single Prompt."""

    interaction: Sequence[Interaction]
    """Each element in the list corresponds to a Prompt in the TestItem."""


SECRETS.register(
    "perspective_api",
    "api_key",
    "See https://crfm-helm.readthedocs.io/en/latest/benchmark/#perspective-api",
)


class PerspectiveAPIAnnotator(BaseAnnotator[PerspectiveAPIAnnotation]):
    """Collect information about a SUT's response using PerspectiveAPI 'analyze comment' interface."""

    def __init__(
        self, desired_attributes: List[str], languages: Optional[List[str]] = None
    ):
        assert desired_attributes, "Must specify at least one desired attribute"
        assert len(desired_attributes) == len(
            set(desired_attributes)
        ), "All desired attributes should be unique."
        self.attributes = desired_attributes
        self.languages = languages
        self.client: Optional[discovery.Resource] = None
        # httplib2 is not thread-safe. Acquire this lock when sending requests to PerspectiveAPI
        self._client_lock: threading.Lock = threading.Lock()

    def _load_client(self) -> discovery.Resource:
        return discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=SECRETS.get_required("perspective_api", "api_key"),
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def annotate_test_item(
        self, interactions: List[PromptInteraction]
    ) -> PerspectiveAPIAnnotation:
        with self._client_lock:
            # Handle lazy init.
            if self.client is None:
                self.client = self._load_client()

        requests = []
        for interaction in interactions:
            for completion in interaction.response.completions:
                requests.append(
                    self.client.comments().analyze(
                        body=self._make_analyze_comment_request(completion.text)
                    )
                )
        with self._client_lock:
            responses = _batch_execute_requests(self.client, requests)

        index = 0
        interaction_scores = []
        for interaction in interactions:
            completion_scores = []
            for _ in interaction.response.completions:
                completion_scores.append(
                    PerspectiveAPIAnnotation.Completion(
                        self._process_analyze_comment_response(responses[index])
                    )
                )
                index += 1
            interaction_scores.append(
                PerspectiveAPIAnnotation.Interaction(completion_scores)
            )
        return PerspectiveAPIAnnotation(interaction_scores)

    def _make_analyze_comment_request(self, completion: str):
        # https://developers.perspectiveapi.com/s/about-the-api-methods
        request = {
            # TODO: Consider what to do if text is too long.
            "comment": {"text": completion},
            # TODO: Consider setting the "context" field based on the prompt's text.
            "requestedAttributes": {attribute: {} for attribute in self.attributes},
            # HELM sets spanAnnotations=True
        }
        if self.languages is not None:
            request["languages"] = self.languages
        return request

    def _process_analyze_comment_response(self, response: Dict) -> Dict[str, float]:
        # https://developers.perspectiveapi.com/s/about-the-api-methods
        scores = response["attributeScores"]
        flattened: Dict[str, float] = {
            # This differs from HELM by using summaryScore.
            attribute: scores[attribute]["summaryScore"]["value"]
            for attribute in self.attributes
        }
        return flattened


def _batch_execute_requests(client: discovery.Resource, requests: List) -> List:
    """Wrapper around Google's batch API.

    https://googleapis.github.io/google-api-python-client/docs/batch.html
    """

    batch_request: BatchHttpRequest = client.new_batch_http_request()
    responses: List[Dict] = [{}] * len(requests)

    def _callback(request_id: str, response: Dict, error: HttpError):
        if error:
            raise error
        responses[int(request_id)] = response

    for i, request in enumerate(requests):
        batch_request.add(
            request=request,
            request_id=str(i),
            callback=_callback,
        )
    # Fills up responses using the callback
    batch_request.execute()
    return responses


if __name__ == "__main__":
    import sys
    from newhelm.placeholders import Prompt
    from newhelm.single_turn_prompt_response import PromptWithContext
    from newhelm.general import get_or_create_json_file
    from newhelm.sut import SUTResponse

    sut_text = sys.argv[1]
    print("Sending:", sut_text)
    SECRETS.set_values(get_or_create_json_file("secrets", "default.json"))
    annotator = PerspectiveAPIAnnotator([ATTRIBUTE_TOXICITY, ATTRIBUTE_THREAT])
    interactions: List[PromptInteraction] = [
        PromptInteraction(
            PromptWithContext(Prompt("The prompt")),
            SUTResponse([SUTCompletion(sut_text)]),
        )
    ]
    response = annotator.annotate_test_item(interactions)
    print(response)
