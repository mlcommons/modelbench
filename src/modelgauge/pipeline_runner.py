import datetime
import json
import logging
from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool

from modelgauge.annotation_pipeline import (
    AnnotatorAssigner,
    AnnotatorSink,
    AnnotatorSource,
    AnnotatorWorkers,
    EnsembleVoter,
)
from modelgauge.dataset import AnnotationDataset, PromptDataset, PromptResponseDataset
from modelgauge.pipeline import Pipeline
from modelgauge.prompt_pipeline import PromptSink, PromptSource, PromptSutAssigner, PromptSutWorkers
from modelgauge.ready import ReadyResponses, Readyable
from modelgauge.sut import SUTOptions

logger = logging.getLogger(__name__)


class PipelineRunner(ABC):
    def __init__(
        self,
        num_workers,
        input_dataset,
        output_dir,
        cache_dir=None,
        sut_options=SUTOptions(),
        tag=None,
    ):
        self.num_workers = num_workers
        self.input_dataset = input_dataset
        self.root_dir = output_dir
        self.cache_dir = cache_dir
        self.sut_options = sut_options
        self.tag = tag
        self.pipeline_segments = []
        self.start_time = datetime.datetime.now()
        self.finish_time = None

        self.ensure_ready()
        self._initialize_segments()

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
    def __init__(self, suts, **kwargs):
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
            output = PromptResponseDataset(self.output_dir() / self.output_file_name, "w")
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

    def _add_annotator_segments(self, include_source=True, include_sink=True):
        if include_source:
            self.pipeline_segments.append(AnnotatorSource(self.input_dataset))
        self.pipeline_segments.append(AnnotatorAssigner(self.annotators))
        self.annotator_workers = AnnotatorWorkers(self.annotators, self.num_workers, cache_path=self.cache_dir)
        self.pipeline_segments.append(self.annotator_workers)
        if include_sink:
            output = AnnotationDataset(self.output_dir() / self.output_file_name, "w")
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


class EnsembleRunner(AnnotatorRunner):
    """Runs annotators + ensemble."""

    def __init__(self, annotators, ensemble, **kwargs):
        self.ensemble = ensemble
        self.ensemble_voter = None  # Convenience pointer.
        # Make sure ensemble's annotators are requested
        missing_annotators = set(ensemble.annotators) - set(annotators.keys())
        if missing_annotators:
            raise ValueError(
                f"Ensemble annotators {missing_annotators} not found in provided annotators {set(annotators.keys())}"
            )
        super().__init__(annotators=annotators, **kwargs)

    def ensure_ready(self) -> None:
        # we should just be able to do readiness check on the
        # EnsembleAnnotatorSet, but the code is mixed between the runner and
        # the class definition.
        # for now, adding the logic here where we check both the
        # annotators and the compute_strategy separately.
        ready_responses = self.check_readyables(self.annotators)
        if not ready_responses.all_ready:
            raise RuntimeError(f"Annotators not ready: {ready_responses.responses}")
        annotation_responses = {uid: resp.response for uid, resp in ready_responses.responses.items()}
        try:
            self.ensemble.strategy.compute_response(annotation_responses)
        except Exception as e:
            raise RuntimeError(f"Error computing ensemble response: {e}")

    @property
    def run_id(self):
        timestamp = self.format_date(self.start_time)
        base_subdir_name = timestamp + "-" + self.tag if self.tag else timestamp
        annotator_uids = list(self.annotators.keys())
        # Replace ensemble's annotator UIDs with just "ensemble" shorthand.
        annotator_uids = [uid for uid in annotator_uids if uid not in self.ensemble.annotators]
        annotator_uids.append("ensemble")
        return f"{base_subdir_name}-{'-'.join(annotator_uids)}"

    def metadata(self):
        return {**super().metadata(), **self._annotator_metadata(), **self._ensemble_metadata()}

    def _ensemble_metadata(self):
        return {
            "ensemble": {"annotators": self.ensemble.annotators, "num_votes": self.ensemble_voter.num_ensemble_votes},
        }

    def _add_ensemble_segments(self):
        """Adds ensemble worker plus annotator sink."""
        self.ensemble_voter = EnsembleVoter(self.ensemble)
        self.pipeline_segments.append(self.ensemble_voter)
        output = AnnotationDataset(self.output_dir() / self.output_file_name, "w")
        self.pipeline_segments.append(AnnotatorSink(output))

    def _initialize_segments(self):
        # Add regular annotator segments
        self._add_annotator_segments(include_source=True, include_sink=False)
        self._add_ensemble_segments()


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
        self._add_annotator_segments(include_source=False)


class PromptPlusEnsembleRunner(PromptRunner, EnsembleRunner):
    def __init__(self, suts, annotators, ensemble, **kwargs):
        super().__init__(suts=suts, annotators=annotators, ensemble=ensemble, **kwargs)

    def ensure_ready(self) -> None:
        PromptRunner.ensure_ready(self)
        EnsembleRunner.ensure_ready(self)

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
        annotator_uids = list(self.annotators.keys())
        # Replace ensemble's annotator UIDs with just "ensemble" shorthand.
        annotator_uids = [uid for uid in annotator_uids if uid not in self.ensemble.annotators]
        annotator_uids.append("ensemble")
        return f"{base_subdir_name}-{'-'.join(self.suts.keys())}-{'-'.join(annotator_uids)}"

    def metadata(self):
        return {**super().metadata(), **self._sut_metadata(), **self._annotator_metadata(), **self._ensemble_metadata()}

    def _initialize_segments(self):
        # Hybrid pipeline: prompt source + ensemble + annotator sink
        self._add_prompt_segments(include_sink=False)
        self._add_annotator_segments(include_source=False, include_sink=False)
        self._add_ensemble_segments()


def build_runner(
    input_path,
    suts=None,
    annotators=None,
    ensemble=None,
    prompt_uid_col=None,
    prompt_text_col=None,
    sut_uid_col=None,
    sut_response_col=None,
    **kwargs,
):
    # Build input dataset
    if suts:
        if sut_uid_col or sut_response_col:
            raise ValueError("SUT uid and SUT response input columns are not used when running SUTs.")
        dataset = PromptDataset(input_path, prompt_uid_col=prompt_uid_col, prompt_text_col=prompt_text_col)
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
    if ensemble and suts:
        pipeline_runner = PromptPlusEnsembleRunner(
            suts, annotators=annotators, ensemble=ensemble, input_dataset=dataset, **kwargs
        )
    elif ensemble:
        pipeline_runner = EnsembleRunner(annotators=annotators, ensemble=ensemble, input_dataset=dataset, **kwargs)
    elif suts and annotators:
        pipeline_runner = PromptPlusAnnotatorRunner(suts=suts, annotators=annotators, input_dataset=dataset, **kwargs)
    elif suts:
        pipeline_runner = PromptRunner(suts=suts, input_dataset=dataset, **kwargs)
    elif annotators:
        pipeline_runner = AnnotatorRunner(annotators=annotators, input_dataset=dataset, **kwargs)
    else:
        raise ValueError("Must specify at least one SUT or annotator to build a pipeline runner.")
    return pipeline_runner
