import logging
import random
import threading
import time
from googleapiclient import discovery  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from googleapiclient.http import BatchHttpRequest  # type: ignore[import-untyped]
from modelgauge.annotator import CompletionAnnotator
from modelgauge.secret_values import RequiredSecret, SecretDescription
from modelgauge.single_turn_prompt_response import PromptWithContext
from modelgauge.sut import SUTCompletion
from pydantic import BaseModel  # type: ignore[import-untyped]
from typing import Dict, List, Mapping, Optional, Sequence

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


class PerspectiveAPIAnnotation(BaseModel):
    scores: Mapping[str, float]
    """Mapping from ATTRIBUTE to score."""


class PerspectiveDeveloperKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="perspective_api",
            key="api_key",
            instructions="First request access https://developers.perspectiveapi.com/s/docs-get-started?language=en_US"
            " and then you can generate a key with https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US",
        )


class AnalyzeCommentRequest(BaseModel):
    # https://developers.perspectiveapi.com/s/about-the-api-methods
    class Comment(BaseModel):
        text: str
        type: Optional[str] = None

    comment: Comment

    class RequestedAttribute(BaseModel):
        scoreType: Optional[str] = None
        scoreThreshold: Optional[float] = None

    requestedAttributes: Mapping[str, RequestedAttribute]
    languages: Optional[Sequence[str]]

    class Context(BaseModel):
        entries: Optional[Sequence["AnalyzeCommentRequest.ContextEntry"]] = None

    class ContextEntry(BaseModel):
        text: Optional[str] = None
        type: Optional[str] = None

    context: Optional[Context] = None
    spanAnnotations: Optional[bool] = None
    doNotStore: Optional[bool] = None
    clientToken: Optional[str] = None
    sessionId: Optional[str] = None
    communityId: Optional[str] = None


class AnalyzeCommentResponse(BaseModel):
    # https://developers.perspectiveapi.com/s/about-the-api-methods

    class AttributeScore(BaseModel):
        summaryScore: "AnalyzeCommentResponse.Score"
        spanScores: Optional[Sequence["AnalyzeCommentResponse.SpanScore"]] = None

    class Score(BaseModel):
        value: float
        type: Optional[str] = None

    class SpanScore(BaseModel):
        begin: int
        end: int
        score: "AnalyzeCommentResponse.Score"

    attributeScores: Mapping[str, AttributeScore]
    languages: Optional[Sequence[str]] = None
    clientToken: Optional[str] = None


class PerspectiveAPIAnnotator(CompletionAnnotator[PerspectiveAPIAnnotation]):
    """Collect information about a SUT's response using PerspectiveAPI 'analyze comment' interface."""

    def __init__(
        self,
        desired_attributes: List[str],
        developer_key: PerspectiveDeveloperKey,
        languages: Optional[List[str]] = None,
        num_retries: int = 6,
    ):
        assert desired_attributes, "Must specify at least one desired attribute"
        assert len(desired_attributes) == len(
            set(desired_attributes)
        ), "All desired attributes should be unique."
        self.attributes = desired_attributes
        self.developer_key = developer_key.value
        self.languages = languages
        self.num_retries = num_retries
        self.rng = random.Random()  # Used for exponential backoff
        self.client: Optional[discovery.Resource] = None
        # httplib2 is not thread-safe. Acquire this lock when sending requests to PerspectiveAPI
        self._client_lock: threading.Lock = threading.Lock()

    def _load_client(self) -> discovery.Resource:
        return discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.developer_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def translate_request(
        self, prompt: PromptWithContext, completion: SUTCompletion
    ) -> AnalyzeCommentRequest:
        return self._make_analyze_comment_request(completion.text)

    def annotate(
        self, annotation_request: AnalyzeCommentRequest
    ) -> AnalyzeCommentResponse:
        """Returns an annotation for a single TestItem's interactions."""
        return self._batch_annotate([annotation_request])[0]

    def _batch_annotate(
        self, annotation_requests: Sequence[AnalyzeCommentRequest]
    ) -> Sequence[AnalyzeCommentResponse]:
        # TODO #313: Expose this ability to batch requests.
        with self._client_lock:
            # Handle lazy init.
            if self.client is None:
                self.client = self._load_client()

        requests = []
        empty_completion_indexes = set()
        request_index = 0
        for annotation_request in annotation_requests:
            if annotation_request.comment.text == "":
                empty_completion_indexes.add(request_index)
            else:
                body = annotation_request.model_dump(exclude_none=True)
                requests.append(self.client.comments().analyze(body=body))
            request_index += 1

        with self._client_lock:
            responses = _batch_execute_requests(
                self.client, requests, self.num_retries, self.rng
            )
        index = 0
        response_index = 0
        results = []
        for index in range(len(annotation_requests)):
            if index in empty_completion_indexes:
                scores = {
                    attribute: AnalyzeCommentResponse.AttributeScore(
                        summaryScore=AnalyzeCommentResponse.Score(value=0.0)
                    )
                    for attribute in annotation_request.requestedAttributes.keys()
                }
                results.append(AnalyzeCommentResponse(attributeScores=scores))
            else:
                results.append(
                    AnalyzeCommentResponse.model_validate(responses[response_index])
                )
                response_index += 1
        return results

    def translate_response(
        self, request: AnalyzeCommentRequest, response: AnalyzeCommentResponse
    ) -> PerspectiveAPIAnnotation:
        return PerspectiveAPIAnnotation(
            scores=self._process_analyze_comment_response(response)
        )

    def _make_analyze_comment_request(self, completion: str):
        requested_attributes = {
            attribute: AnalyzeCommentRequest.RequestedAttribute()
            for attribute in self.attributes
        }
        return AnalyzeCommentRequest(
            # TODO: Consider what to do if text is too long.
            comment=AnalyzeCommentRequest.Comment(text=completion),
            # TODO: Consider setting the "context" field based on the prompt's text.
            requestedAttributes=requested_attributes,
            languages=self.languages,
            # HELM sets spanAnnotations=True
        )

    def _process_analyze_comment_response(
        self, response: AnalyzeCommentResponse
    ) -> Dict[str, float]:
        flattened: Dict[str, float] = {
            # This differs from HELM by using summaryScore.
            attribute: response.attributeScores[attribute].summaryScore.value
            for attribute in self.attributes
        }
        return flattened


def _batch_execute_requests(
    client: discovery.Resource, requests: List, num_retries: int, rng: random.Random
) -> List:
    """Wrapper around Google's batch API.

    This can give significant speedup. For example for PerspectiveAPI, batching
    25 requests is about 15x faster than doing each as separate calls.
    https://googleapis.github.io/google-api-python-client/docs/batch.html
    """

    if not requests:
        return []

    errors = [None] * len(requests)
    responses: List[Dict] = [{}] * len(requests)

    def _callback(request_id: str, response: Dict, error: HttpError):
        index = int(request_id)
        if error:
            errors[index] = error
        else:
            # Clear any past errors
            errors[index] = None
        responses[index] = response

    # Keep track of what requests have not yet successfully gotten a response
    needs_call = list(range(len(requests)))
    retriable_errors: List[HttpError] = []
    for retry_count in range(num_retries + 1):
        if retry_count > 0:
            # Perform exponential backoff
            sleep_amount = rng.uniform(1, 2) * 2**retry_count
            logging.info("Performing exponential backoff. Sleeping:", sleep_amount)
            time.sleep(sleep_amount)

        # Build up a batch
        batch_request: BatchHttpRequest = client.new_batch_http_request()
        for i in needs_call:
            batch_request.add(
                request=requests[i],
                request_id=str(i),
                callback=_callback,
            )
        # Fills up responses using the callback
        batch_request.execute()

        # Figure out which requests need to be tried again.
        next_round_needs_call: List[int] = []
        fatal_errors: List[HttpError] = []
        retriable_errors = []
        for i in needs_call:
            error = errors[i]
            if error is not None:
                if _is_retriable(error):
                    next_round_needs_call.append(i)
                    retriable_errors.append(error)
                else:
                    fatal_errors.append(error)
        if fatal_errors:
            # Just use the first one as an example.
            raise fatal_errors[0]
        if not next_round_needs_call:
            break
        needs_call = next_round_needs_call
    if retriable_errors:
        # We exhausted our retries, so raise the first as an example.
        raise retriable_errors[0]
    return responses


def _is_retriable(error: HttpError) -> bool:
    """Check if this error can be retried."""
    # Retry any 5XX status.
    if 500 <= error.status_code < 600:
        return True
    # 429 is "Too Many Requests" and for PerspectiveAPI means "RATE_LIMIT_EXCEEDED"
    if error.status_code == 429:
        return True
    return False


if __name__ == "__main__":
    import sys
    from modelgauge.config import load_secrets_from_config
    from modelgauge.prompt import TextPrompt
    from modelgauge.single_turn_prompt_response import PromptWithContext

    sut_text = sys.argv[1]
    print("Sending:", sut_text)
    secrets = load_secrets_from_config()
    annotator = PerspectiveAPIAnnotator(
        [ATTRIBUTE_TOXICITY, ATTRIBUTE_THREAT], PerspectiveDeveloperKey.make(secrets)
    )
    prompt = PromptWithContext(prompt=TextPrompt(text="not used"), source_id=None)
    completion = SUTCompletion(text=sut_text)
    request = annotator.translate_request(prompt, completion)
    print("Request:", request)
    response = annotator.annotate(request)
    print("Response:", response)
    annotation = annotator.translate_response(request, response)

    print("Annotation:", annotation)
