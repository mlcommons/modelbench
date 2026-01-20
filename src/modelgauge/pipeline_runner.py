import datetime
import json
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool

from modellogger.log_config import get_logger

from modelgauge.annotation_pipeline import AnnotatorAssigner, AnnotatorSink, AnnotatorSource, AnnotatorWorkers
from modelgauge.dataset import AnnotationDataset, PromptDataset, PromptResponseDataset
from modelgauge.pipeline import Pipeline
from modelgauge.prompt_pipeline import PromptSink, PromptSource, PromptSutAssigner, PromptSutWorkers
from modelgauge.ready import Readyable, ReadyResponses
from modelgauge.sut_capabilities_verification import MissingSUTCapabilities

logger = get_logger(__name__)


class PipelineRunner(ABC):
    def __init__(
        self,
        num_workers,
        input_dataset,
        output_dir,
        cache_dir=None,
        tag=None,
    ):
        self.num_workers = num_workers
        self.input_dataset = input_dataset
        self.root_dir = output_dir
        self.cache_dir = cache_dir
        self.tag = tag
        self.pipeline_segments = []
        self.start_time = datetime.datetime.now()
        self.finish_time = None

        self.ensure_ready()
        try:
            self._initialize_segments()
        except MissingSUTCapabilities as e:
            logger.error(f"Cannot run {self.__class__.__name__} due to missing model capabilities.")
            raise e

    def ensure_ready(self) -> None:
        """Ensure the pipeline runner is ready to start.
        Override in subclasses with specific readiness checks.
        """
        pass

    @staticmethod
    def check_readyables(readyables: dict[str, Readyable]) -> ReadyResponses:
        with ThreadPool(len(readyables)) as pool:
            results = pool.starmap(lambda uid, item: (uid, item.is_ready()), readyables.items())
        ready_responses_by_uid = dict(results)
        return ReadyResponses.from_dict(ready_responses_by_uid)

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
                "source": self.input_dataset.path.name,
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

    def _write_metadata(self):
        with open(self.output_dir() / "metadata.json", "w") as f:
            json.dump(self.metadata(), f, indent=4)


class PromptRunner(PipelineRunner):
    def __init__(self, suts, sut_options=ModelOptions(), **kwargs):
        self.sut_options = sut_options
        logger.info(f"Using SUT options: {self.sut_options}")
        self.sut_logprobs = sut_options.top_logprobs is not None
        self.suts = suts
        self.sut_worker = None  # Convenience pointer.
        super().__init__(**kwargs)

    def ensure_ready(self) -> None:
        ready_responses = self.check_readyables(self.suts)
        if not ready_responses.all_ready:
            raise RuntimeError(f"SUTs not ready: {ready_responses.responses}")

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

    def _add_prompt_segments(self, include_sink=True):
        self.pipeline_segments.append(PromptSource(self.input_dataset))
        self.pipeline_segments.append(PromptSutAssigner(self.suts))
        self.sut_worker = PromptSutWorkers(
            self.suts, sut_options=self.sut_options, workers=self.num_workers, cache_path=self.cache_dir
        )
        self.pipeline_segments.append(self.sut_worker)
        if include_sink:
            output = PromptResponseDataset(
                self.output_dir() / self.output_file_name, "w", sut_logprobs=self.sut_logprobs
            )
            self.pipeline_segments.append(PromptSink(output))

    def _sut_metadata(self):
        counts = self.sut_worker.sut_response_counts
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

    def _initialize_segments(self):
        self._add_prompt_segments(include_sink=True)


class AnnotatorRunner(PipelineRunner):
    def __init__(self, annotators, **kwargs):
        self.annotators = annotators
        self.annotator_workers = None  # Convenience pointer.
        super().__init__(**kwargs)

    def ensure_ready(self) -> None:
        ready_responses = self.check_readyables(self.annotators)
        if not ready_responses.all_ready:
            raise RuntimeError(f"Annotators not ready: {ready_responses.responses}")

    @property
    def num_total_items(self):
        return self.num_input_items * len(self.annotators)

    @property
    def output_file_name(self):
        return "annotations.csv"

    @property
    def run_id(self):
        timestamp = self.format_date(self.start_time)
        base_subdir_name = timestamp + "-" + self.tag if self.tag else timestamp
        return f"{base_subdir_name}-{'-'.join(self.annotators.keys())}"

    def metadata(self):
        return {**super().metadata(), **self._annotator_metadata()}

    def _add_annotator_segments(self, include_source=True, include_sink=True, include_sut_logprobs=False):
        if include_source:
            self.pipeline_segments.append(AnnotatorSource(self.input_dataset))
        self.pipeline_segments.append(AnnotatorAssigner(self.annotators))
        self.annotator_workers = AnnotatorWorkers(self.annotators, self.num_workers, cache_path=self.cache_dir)
        self.pipeline_segments.append(self.annotator_workers)
        if include_sink:
            jailbreak = isinstance(self.input_dataset, PromptDataset) and self.input_dataset.jailbreak
            output = AnnotationDataset(
                self.output_dir() / self.output_file_name, "w", jailbreak=jailbreak, sut_logprobs=include_sut_logprobs
            )
            self.pipeline_segments.append(AnnotatorSink(output))

    def _annotator_metadata(self):
        counts = self.annotator_workers.annotation_counts
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

    def _initialize_segments(self):
        self._add_annotator_segments(include_source=True)


class PromptPlusAnnotatorRunner(PromptRunner, AnnotatorRunner):
    def __init__(self, suts, annotators, **kwargs):
        super().__init__(suts=suts, annotators=annotators, **kwargs)

    def ensure_ready(self) -> None:
        PromptRunner.ensure_ready(self)
        AnnotatorRunner.ensure_ready(self)

    @property
    def num_total_items(self):
        return self.num_input_items * len(self.suts) * len(self.annotators)

    @property
    def output_file_name(self):
        return "prompt-responses-annotated.csv"

    @property
    def run_id(self):
        timestamp = self.format_date(self.start_time)
        base_subdir_name = timestamp + "-" + self.tag if self.tag else timestamp
        return f"{base_subdir_name}-{'-'.join(self.suts.keys())}-{'-'.join(self.annotators.keys())}"

    def metadata(self):
        return {**super().metadata(), **self._sut_metadata(), **self._annotator_metadata()}

    def _initialize_segments(self):
        # Hybrid pipeline: prompt source + annotator sink
        self._add_prompt_segments(include_sink=False)
        self._add_annotator_segments(include_source=False, include_sut_logprobs=self.sut_logprobs)


def build_runner(
    input_path,
    suts=None,
    annotators=None,
    prompt_uid_col=None,
    prompt_text_col=None,
    seed_prompt_text_col=None,
    sut_uid_col=None,
    sut_response_col=None,
    jailbreak=False,
    sut_options=None,
    **kwargs,
):
    if jailbreak and not (annotators and suts):
        raise ValueError("Jailbreak mode only applies when running both suts and annotators.")
    if sut_options and not suts:
        raise ValueError("sut_options only applies when running SUTs.")
    elif suts and not sut_options:
        sut_options = ModelOptions()
    # Build input dataset
    if suts:
        if sut_uid_col or sut_response_col:
            raise ValueError("SUT uid and SUT response input columns are not used when running SUTs.")
        dataset = PromptDataset(
            input_path,
            prompt_uid_col=prompt_uid_col,
            prompt_text_col=prompt_text_col,
            seed_prompt_text_col=seed_prompt_text_col,
            jailbreak=jailbreak,
        )
    else:
        dataset = PromptResponseDataset(
            input_path,
            mode="r",
            prompt_uid_col=prompt_uid_col,
            prompt_text_col=prompt_text_col,
            sut_uid_col=sut_uid_col,
            sut_response_col=sut_response_col,
        )
    # Build runner
    if suts and annotators:
        pipeline_runner = PromptPlusAnnotatorRunner(
            suts=suts, annotators=annotators, input_dataset=dataset, sut_options=sut_options, **kwargs
        )
    elif suts:
        pipeline_runner = PromptRunner(suts=suts, input_dataset=dataset, sut_options=sut_options, **kwargs)
    elif annotators:
        pipeline_runner = AnnotatorRunner(annotators=annotators, input_dataset=dataset, **kwargs)
    else:
        raise ValueError("Must specify at least one SUT or annotator to build a pipeline runner.")
    return pipeline_runner
