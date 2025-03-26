from abc import ABC, abstractmethod
import datetime
import json
import logging

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
from modelgauge.sut import SUTOptions

logger = logging.getLogger(__name__)


class PipelineRunner(ABC):
    def __init__(self, num_workers, input_path, output_dir, cache_dir=None, sut_options=SUTOptions(), tag=None):
        self.num_workers = num_workers
        self.input_path = input_path
        self.root_dir = output_dir
        self.cache_dir = cache_dir
        self.sut_options = sut_options
        self.tag = tag
        self.pipeline_segments = []
        self.start_time = datetime.datetime.now()
        self.finish_time = None

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

    def metadata(self):
        duration = self.finish_time - self.start_time
        hours, minutes, seconds = str(duration).split(":")
        duration_string = f"{hours}h{minutes}m{seconds}s"

        metadata = {
            "run_id": self.run_id,
            "run_info": {
                "started": str(self.start_time),
                "finished": str(self.finish_time),
                "duration": duration_string,
            },
            "input": {
                "source": self.input_path.name,
                "num_items": self.num_input_items,
            },
        }
        return metadata

    def output_dir(self):
        output_path = self.root_dir / self.run_id
        if not output_path.exists():
            logger.info(f"Creating output dir {output_path}")
            output_path.mkdir(parents=True)
        return output_path

    def run(self, progress_callback, debug):
        pipeline = Pipeline(
            *self.pipeline_segments,
            progress_callback=progress_callback,
            debug=debug,
        )
        pipeline.run()
        self.finish_time = datetime.datetime.now()
        logger.info(f"output saved to {self.output_dir() / self.output_file_name}")
        self._write_metadata()

    @staticmethod
    def format_date(date):
        return date.strftime("%Y%m%d-%H%M%S")

    @abstractmethod
    def _initialize_segments(self):
        pass

    def _add_prompt_segments(self, suts, include_sink=True):
        input = CsvPromptInput(self.input_path)
        self.pipeline_segments.append(PromptSource(input))
        self.pipeline_segments.append(PromptSutAssigner(suts))
        self.pipeline_segments.append(
            PromptSutWorkers(suts, sut_options=self.sut_options, workers=self.num_workers, cache_path=self.cache_dir)
        )
        if include_sink:
            output = CsvPromptOutput(self.output_dir() / self.output_file_name, suts)
            self.pipeline_segments.append(PromptSink(suts, output))

    def _add_annotator_segments(self, annotators, include_source=True):
        if include_source:
            input = CsvAnnotatorInput(self.input_path)
            self.pipeline_segments.append(AnnotatorSource(input))
        self.pipeline_segments.append(AnnotatorAssigner(annotators))
        self.pipeline_segments.append(AnnotatorWorkers(annotators, self.num_workers))
        output = JsonlAnnotatorOutput(self.output_dir() / self.output_file_name)
        self.pipeline_segments.append(AnnotatorSink(annotators, output))

    def _annotator_metadata(self):
        annotator_worker = self.pipeline_segments[-2]
        assert isinstance(
            annotator_worker, AnnotatorWorkers
        ), "Attempting to access annotator metadata without annotator workers"
        counts = annotator_worker.annotation_counts
        return {
            "annotators": [
                {
                    "uid": uid,
                }
                for uid, annotator in self.annotators.items()
            ],
            "annotations": {
                "count": sum(counts.values()),
                "by_annotator": {uid: {"count": count} for uid, count in counts.items()},
            },
        }

    def _sut_metadata(self):
        sut_worker = self.pipeline_segments[2]
        assert isinstance(sut_worker, PromptSutWorkers), "Attempting to access sut metadata without sut workers"
        counts = sut_worker.sut_response_counts
        return {
            "suts": [
                {
                    "uid": uid,
                    "initialization_record": sut.initialization_record.model_dump(),
                    "sut_options": self.sut_options.model_dump(exclude_none=True),
                }
                for uid, sut in self.suts.items()
            ],
            "responses": {
                "count": sum(counts.values()),
                "by_sut": {uid: {"count": count} for uid, count in counts.items()},
            },
        }

    def _write_metadata(self):
        with open(self.output_dir() / "metadata.json", "w") as f:
            json.dump(self.metadata(), f, indent=4)


class PromptRunner(PipelineRunner):
    def __init__(self, suts, *args, **kwargs):
        self.suts = suts
        super().__init__(*args, **kwargs)

    @property
    def num_total_items(self):
        return self.num_input_items * len(self.suts)

    @property
    def output_file_name(self):
        return "prompt-responses.csv"

    @property
    def run_id(self):
        timestamp = self.format_date(self.start_time)
        base_subdir_name = timestamp + "-" + self.tag if self.tag else timestamp
        return f"{base_subdir_name}-{'-'.join(self.suts.keys())}"

    def metadata(self):
        return {**super().metadata(), **self._sut_metadata()}

    def _initialize_segments(self):
        self._add_prompt_segments(self.suts, include_sink=True)


class PromptPlusAnnotatorRunner(PipelineRunner):
    def __init__(self, suts, annotators, *args, **kwargs):
        self.suts = suts
        self.annotators = annotators
        super().__init__(*args, **kwargs)

    @property
    def num_total_items(self):
        return self.num_input_items * len(self.suts) * len(self.annotators)

    @property
    def output_file_name(self):
        return "prompt-responses-annotated.jsonl"

    @property
    def run_id(self):
        timestamp = self.format_date(self.start_time)
        base_subdir_name = timestamp + "-" + self.tag if self.tag else timestamp
        return f"{base_subdir_name}-{'-'.join(self.suts.keys())}-{'-'.join(self.annotators.keys())}"

    def metadata(self):
        return {**super().metadata(), **self._sut_metadata(), **self._annotator_metadata()}

    def _initialize_segments(self):
        # Hybrid pipeline: prompt source + annotator sink
        self._add_prompt_segments(self.suts, include_sink=False)
        self._add_annotator_segments(self.annotators, include_source=False)


class AnnotatorRunner(PipelineRunner):
    def __init__(self, annotators, *args, **kwargs):
        self.annotators = annotators
        super().__init__(*args, **kwargs)

    @property
    def num_total_items(self):
        return self.num_input_items * len(self.annotators)

    @property
    def output_file_name(self):
        return "annotations.jsonl"

    @property
    def run_id(self):
        timestamp = self.format_date(self.start_time)
        base_subdir_name = timestamp + "-" + self.tag if self.tag else timestamp
        return f"{base_subdir_name}-{'-'.join(self.annotators.keys())}"

    def metadata(self):
        return {**super().metadata(), **self._annotator_metadata()}

    def _initialize_segments(self):
        self._add_annotator_segments(self.annotators, include_source=True)
