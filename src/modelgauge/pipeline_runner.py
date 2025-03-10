from abc import ABC, abstractmethod
import datetime
import pathlib

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
    def __init__(self, num_workers, input_path, output_dir, cache_dir, sut_options, tag=None):
        self.num_workers = num_workers
        self.input_path = input_path
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.sut_options = sut_options
        self.tag = tag
        self.pipeline_segments = []
        self.start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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
        print(f"\noutput saved to {self._create_output_path()}")

    @abstractmethod
    def _initialize_segments(self):
        pass

    def _add_prompt_segments(self, suts, include_sink=True):
        input = CsvPromptInput(self.input_path, self.sut_options)
        self.pipeline_segments.append(PromptSource(input))
        self.pipeline_segments.append(PromptSutAssigner(suts))
        self.pipeline_segments.append(PromptSutWorkers(suts, self.num_workers, cache_path=self.cache_dir))
        if include_sink:
            output = CsvPromptOutput(self._create_output_path(), suts)
            self.pipeline_segments.append(PromptSink(suts, output))

    def _add_annotator_segments(self, annotators, include_source=True):
        if include_source:
            input = CsvAnnotatorInput(self.input_path)
            self.pipeline_segments.append(AnnotatorSource(input))
        self.pipeline_segments.append(AnnotatorAssigner(annotators))
        self.pipeline_segments.append(AnnotatorWorkers(annotators, self.num_workers))
        output = JsonlAnnotatorOutput(self._create_output_path())
        self.pipeline_segments.append(AnnotatorSink(annotators, output))

    def _create_output_path(self):
        output_path = self.output_dir / self.sub_dir_name
        if not output_path.exists():
            print(f"Creating output dir {output_path}")
            output_path.mkdir(parents=True)
        return output_path / self.output_file_name


class PromptRunner(PipelineRunner):
    def __init__(self, *args, suts):
        self.suts = suts
        super().__init__(*args)

    @property
    def num_total_items(self):
        return self.num_input_items * len(self.suts)

    @property
    def output_file_name(self):
        return "prompt-responses.csv"

    @property
    def sub_dir_name(self):
        base_subdir_name = self.start_time + "-" + self.tag if self.tag else self.start_time
        sub_dir = pathlib.Path(f"{base_subdir_name}-{'-'.join(self.suts.keys())}")
        return sub_dir

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

    @property
    def output_file_name(self):
        return "prompt-responses-annotated.jsonl"

    @property
    def sub_dir_name(self):
        base_subdir_name = self.start_time + "-" + self.tag if self.tag else self.start_time
        sub_dir = pathlib.Path(f"{base_subdir_name}-{'-'.join(self.suts.keys())}-{'-'.join(self.annotators.keys())}")
        return sub_dir

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

    @property
    def output_file_name(self):
        return "annotations.jsonl"

    @property
    def sub_dir_name(self):
        base_subdir_name = self.start_time + "-" + self.tag if self.tag else self.start_time
        sub_dir = pathlib.Path(f"{base_subdir_name}-{'-'.join(self.annotators.keys())}")
        return sub_dir

    def _initialize_segments(self):
        self._add_annotator_segments(self.annotators, include_source=True)
