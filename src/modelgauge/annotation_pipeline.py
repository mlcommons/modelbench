import time

from airrlogger.log_config import get_logger
from pydantic import BaseModel

from modelgauge.annotator import Annotator
from modelgauge.dataset import AnnotationDataset, PromptResponseDataset
from modelgauge.pipeline import CachingPipe, Pipe, Sink, Source
from modelgauge.single_turn_prompt_response import AnnotatedSUTInteraction, SUTInteraction

logger = get_logger(__name__)

MAX_ANNOTATION_ATTEMPTS = 3
FAILURE_ANNOTATION = "failure"


class AnnotatorSource(Source):
    def __init__(self, input: PromptResponseDataset):
        super().__init__()
        self.input = input

    def new_item_iterable(self):
        return self.input


class AnnotatorAssigner(Pipe):
    def __init__(self, annotators: dict[str, Annotator]):
        super().__init__()
        self.annotators = annotators

    def handle_item(self, item: SUTInteraction):
        for annotator_uid in self.annotators:
            self.downstream_put((item, annotator_uid))


class AnnotatorWorkers(CachingPipe):
    def __init__(self, annotators: dict[str, Annotator], workers=None, cache_path=None):
        self.sleep_time = 10
        if workers is None:
            workers = 8
        super().__init__(thread_count=workers, cache_path=cache_path)
        self.annotators = annotators
        self.annotation_counts = {uid: 0 for uid in annotators}

    def key(self, item):
        sut_interaction, annotator_uid = item
        annotator = self.annotators[annotator_uid]
        request = annotator.translate_request(sut_interaction.prompt, sut_interaction.response)
        if isinstance(request, BaseModel):
            request = request.model_dump_json()
        return (sut_interaction.prompt.source_id, request, annotator_uid)

    def handle_uncached_item(self, item):
        sut_interaction, annotator_uid = item
        annotator = self.annotators[annotator_uid]
        request = annotator.translate_request(sut_interaction.prompt, sut_interaction.response)
        for attempt in range(1, MAX_ANNOTATION_ATTEMPTS + 1):
            try:
                response = annotator.annotate(request)
                result = annotator.translate_response(request, response)
                self.annotation_counts[annotator_uid] += 1
                return AnnotatedSUTInteraction(
                    annotator_uid=annotator_uid, annotation=result, sut_interaction=sut_interaction
                )
            except Exception as e:
                if attempt < MAX_ANNOTATION_ATTEMPTS:
                    logger.warning(
                        f"Exception calling annotator {annotator_uid} on attempt {attempt}/{MAX_ANNOTATION_ATTEMPTS}: {e}\nRetrying.....",
                        exc_info=True,
                    )
                    time.sleep(self.sleep_time)
                else:
                    logger.error(
                        f"Exception calling annotator {annotator_uid} on attempt {attempt}/{MAX_ANNOTATION_ATTEMPTS}: {e}. "
                        f"Recording annotation as {FAILURE_ANNOTATION!r}.",
                        exc_info=True,
                    )
        self.annotation_counts[annotator_uid] += 1
        return AnnotatedSUTInteraction(
            annotator_uid=annotator_uid, annotation=FAILURE_ANNOTATION, sut_interaction=sut_interaction
        )


class AnnotatorSink(Sink):
    def __init__(self, writer: AnnotationDataset):
        super().__init__()
        self.writer = writer

    def run(self):
        with self.writer:
            super().run()

    def handle_item(self, item: AnnotatedSUTInteraction):
        self.writer.write(item)
