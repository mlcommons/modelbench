from abc import ABC, abstractmethod

from modelgauge.annotation_pipeline import (
    AnnotatorAssigner,
    AnnotatorSink,
    AnnotatorSource,
    AnnotatorWorkers,
    CsvAnnotatorInput,
    JsonlAnnotatorOutput,
)
from modelgauge.pipeline import Pipeline
from modelgauge.prompt_pipeline import (
    PromptSource,
    PromptSutAssigner,
    PromptSutWorkers,
    PromptSink,
    CsvPromptInput,
    CsvPromptOutput,
)


class PipelineRunner(ABC):
    def __init__(self, num_workers, input_path, output_path, cache_dir):
        self.num_workers = num_workers
        self.input_path = input_path
        self.output_path = output_path
        self.cache_dir = cache_dir
        self.pipeline_segments = []

        self._initialize_segments()

    @property
    def num_input_items(self):
        """Number of items in the input file.

        Corresponds to the number of prompts when running SUTs or the number of SUT interactions when only running annotators.
        """
        return len(self.pipeline_segments[0].input)

    @property
    @abstractmethod
    def num_total_items(self):
        """Total number of items to process."""
        pass

    def run(self, progress_callback, debug):
        pipeline = Pipeline(
            *self.pipeline_segments,
            progress_callback=progress_callback,
            debug=debug,
        )
        pipeline.run()

    @abstractmethod
    def _initialize_segments(self):
        pass

    def _add_prompt_segments(self, suts, include_sink=True):
        input = CsvPromptInput(self.input_path)
        self.pipeline_segments.append(PromptSource(input))
        self.pipeline_segments.append(PromptSutAssigner(suts))
        self.pipeline_segments.append(
            PromptSutWorkers(suts, self.num_workers, cache_path=self.cache_dir)
        )
        if include_sink:
            output = CsvPromptOutput(self.output_path, suts)
            self.pipeline_segments.append(PromptSink(suts, output))

    def _add_annotator_segments(self, annotators, include_source=True):
        if include_source:
            input = CsvAnnotatorInput(self.input_path)
            self.pipeline_segments.append(AnnotatorSource(input))
        self.pipeline_segments.append(AnnotatorAssigner(annotators))
        self.pipeline_segments.append(AnnotatorWorkers(annotators, self.num_workers))
        output = JsonlAnnotatorOutput(self.output_path)
        self.pipeline_segments.append(AnnotatorSink(annotators, output))


class PromptRunner(PipelineRunner):
    def __init__(self, *args, suts):
        self.suts = suts
        super().__init__(*args)

    @property
    def num_total_items(self):
        return self.num_input_items * len(self.suts)

    def _initialize_segments(self):
        self._add_prompt_segments(self.suts, include_sink=True)


class PromptPlusAnnotatorRunner(PipelineRunner):
    def __init__(self, *args, suts, annotators):
        self.suts = suts
        self.annotators = annotators
        super().__init__(*args)

    @property
    def num_total_items(self):
        return self.num_input_items * len(self.suts) * len(self.annotators)

    def _initialize_segments(self):
        # Hybrid pipeline: prompt source + annotator sink
        self._add_prompt_segments(self.suts, include_sink=False)
        self._add_annotator_segments(self.annotators, include_source=False)


class AnnotatorRunner(PipelineRunner):
    def __init__(self, *args, annotators):
        self.annotators = annotators
        super().__init__(*args)

    @property
    def num_total_items(self):
        return self.num_input_items * len(self.annotators)

    def _initialize_segments(self):
        self._add_annotator_segments(self.annotators, include_source=True)
