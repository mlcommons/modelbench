import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pydantic import BaseModel

from modelgauge.data_schema import (
    AnnotationJailbreakSchema,
    AnnotationSchema,
    PromptJailbreakSchema,
    PromptResponseSchema,
    PromptSchema,
)
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import AnnotatedSUTInteraction, SUTInteraction, TestItem
from modelgauge.sut import SUTResponse


class BaseDataset(ABC):
    """This class provides common functionality for CSV file handling and context management."""

    quoting = csv.QUOTE_ALL

    def __init__(self, path: Union[str, Path], mode: str):
        """Args:
        path: Path to the dataset file
        mode: Mode to open the file in ('r' for read, 'w' for write)
        """
        self.path = Path(path)
        if self.path.suffix.lower() != ".csv":
            raise ValueError(f"Invalid dataset file {path}. Must be a CSV file.")

        self.mode = mode
        assert mode in ["r", "w"], f"Invalid dataset mode {mode}. Must be 'r' or 'w'."
        if self.mode == "r" and not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist.")
        if self.mode == "w" and self.path.exists():
            raise FileExistsError(f"File {self.path} already exists.")

        self.file = None
        self.writer = None
        self.reader = None
        self.schema = self._get_schema()  # Implemented by subclass.

    def __enter__(self):
        """Context manager entry. Opens the file and sets the reader or writer."""
        if self.file is not None:
            raise RuntimeError("Cannot enter context manager twice before exiting.")
        if self.mode == "w" and not self.path.exists():
            # New file, need to write header.
            self.file = open(self.path, mode=self.mode, newline="")
            self.writer = csv.writer(self.file, quoting=self.quoting)
            self.writer.writerow(self.header_columns())
        elif self.mode == "w":
            # Append to existing file.
            self.file = open(self.path, mode="a", newline="")
            self.writer = csv.writer(self.file, quoting=self.quoting)
        elif self.mode == "r":
            self.file = open(self.path, mode=self.mode, newline="")
            self.reader = csv.DictReader(self.file)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit. Closes the file and unsets the reader and writer."""
        if self.file:
            self.file.close()
            self.file = None
        self.reader = None
        self.writer = None

    def __iter__(self):
        """Base iterator implementation that ensures proper context management.
        Will enter the context if not already open.
        """
        if self.mode != "r":
            raise RuntimeError("Can only iterate over dataset in read mode.")

        # If we're not already in a context, create one for this iteration
        if self.file is None:
            with self:
                for row in self.reader:
                    yield self.row_to_item(row)
        else:
            # We're already in a context, just yield
            for row in self.reader:
                yield self.row_to_item(row)

    def __len__(self) -> int:
        if self.mode != "r":
            raise NotImplementedError("Length not supported in write mode")
        count = 0
        with open(self.path, newline="") as f:
            csvreader = csv.reader(f)
            next(csvreader)  # Skip header row
            for row in csvreader:
                count += 1
        return count

    @abstractmethod
    def _get_schema(self):
        """Return dataset schema. To be implemented by subclasses."""
        pass

    def _read_header(self) -> list[str]:
        """Read the header row from a CSV file."""
        if self.mode != "r":
            raise RuntimeError("Can only read header in read mode.")
        if self.file is None:
            with self:
                header = self.reader.fieldnames  # type: ignore
        else:
            header = self.reader.fieldnames  # type: ignore
        return header

    def header_columns(self) -> Sequence[str]:
        assert self.schema is not None, "Sub-classes must initialized schema."
        return self.schema.header

    def write(self, item: Any):
        """Write an item to the csv file."""
        if self.mode != "w":
            raise RuntimeError("Cannot write to dataset in read mode")
        if not self.writer:
            raise RuntimeError("Must be in a context to write.")
        self.writer.writerow(self.item_to_row(item))

    def row_to_item(self, row: dict):
        """Transform a single  dict-row from the csv file into a dataset object."""
        raise NotImplementedError("Subclasses that enable reading must implement this method.")

    def item_to_row(self, item: Any) -> list[str]:
        """Transform a dataset object into a list of strings that can be written to a csv file."""
        raise NotImplementedError("Subclasses that enable writing must implement this method.")

    def dump_sut_logprobs(self, sut_interation: SUTInteraction) -> str:
        if sut_interation.response.top_logprobs is None:
            logprobs = []
        else:
            logprobs = [[tp.model_dump() for tp in tt.top_tokens] for tt in sut_interation.response.top_logprobs]
        return str(logprobs)


class PromptDataset(BaseDataset):
    """Dataset for reading prompts as TestItems from a CSV file. Read only."""

    def __init__(
        self,
        path: Union[str, Path],
        prompt_uid_col: Optional[str] = None,
        prompt_text_col: Optional[str] = None,
        seed_prompt_text_col: Optional[str] = None,
        jailbreak=False,
    ):
        if seed_prompt_text_col and not jailbreak:
            raise ValueError("seed_prompt_text_col is only applicable if jailbreak is True.")
        self.prompt_uid_col = prompt_uid_col
        self.prompt_text_col = prompt_text_col
        self.seed_prompt_text_col = seed_prompt_text_col
        self.jailbreak = jailbreak
        super().__init__(path, "r")

    def _get_schema(self):
        if self.jailbreak:
            return PromptJailbreakSchema(
                self._read_header(),
                prompt_uid_col=self.prompt_uid_col,
                prompt_text_col=self.prompt_text_col,
                evaluated_prompt_text_col=self.seed_prompt_text_col,
            )
        return PromptSchema(
            self._read_header(),
            prompt_uid_col=self.prompt_uid_col,
            prompt_text_col=self.prompt_text_col,
        )

    def row_to_item(self, row: dict) -> TestItem:
        """Convert a single prompt row to a TestItem."""
        seed_prompt_args = {}
        if self.jailbreak:
            seed_prompt_args["evaluated_prompt"] = TextPrompt(text=row[self.schema.evaluated_prompt_text])
        return TestItem(
            prompt=TextPrompt(text=row[self.schema.prompt_text]),
            source_id=row[self.schema.prompt_uid],
            context=row,
            **seed_prompt_args,
        )


class PromptResponseDataset(BaseDataset):
    """Dataset for prompt-response CSV data. Read or write."""

    def __init__(
        self,
        path: Union[str, Path],
        mode: str,
        prompt_uid_col: Optional[str] = None,
        prompt_text_col: Optional[str] = None,
        sut_uid_col: Optional[str] = None,
        sut_response_col: Optional[str] = None,
        sut_logprobs: bool = False,
    ):
        self.prompt_uid_col = prompt_uid_col
        self.prompt_text_col = prompt_text_col
        self.sut_uid_col = sut_uid_col
        self.sut_response_col = sut_response_col
        self.sut_logprobs = sut_logprobs
        super().__init__(path, mode)

    def _get_schema(self):
        if self.mode == "r":
            if self.sut_logprobs:
                raise ValueError("There is no reason to include sut_logprobs when reading prompt-response datasets.")
            return PromptResponseSchema(
                self._read_header(),
                prompt_uid_col=self.prompt_uid_col,
                prompt_text_col=self.prompt_text_col,
                sut_uid_col=self.sut_uid_col,
                sut_response_col=self.sut_response_col,
            )
        else:
            if self.prompt_uid_col or self.prompt_text_col or self.sut_uid_col or self.sut_response_col:
                raise ValueError("Cannot specify columns in write mode.")
            return PromptResponseSchema.default(sut_logprobs=self.sut_logprobs)

    def row_to_item(self, row: dict) -> SUTInteraction:
        prompt = TestItem(
            prompt=TextPrompt(text=row[self.schema.prompt_text]),
            source_id=row[self.schema.prompt_uid],
            context=row,
        )
        response = SUTResponse(text=row[self.schema.sut_response])
        return SUTInteraction(prompt, row[self.schema.sut_uid], response)

    def item_to_row(self, item: SUTInteraction) -> list[str]:
        if not isinstance(item.prompt.prompt, TextPrompt):
            raise ValueError(f"Error handling {item}. Can only handle TextPrompts.")
        assert item.prompt.source_id is not None, "Prompt source_id is required."

        data = {
            self.schema.prompt_uid: item.prompt.source_id,
            self.schema.prompt_text: item.prompt.prompt.text,
            self.schema.sut_uid: item.sut_uid,
            self.schema.sut_response: item.response.text,
        }
        if self.sut_logprobs:
            data[self.schema.sut_logprobs] = self.dump_sut_logprobs(item)
        return self.schema.create_row(data)


class AnnotationDataset(BaseDataset):
    """Dataset for annotated prompt-response CSV data. Read or write."""

    def __init__(
        self,
        path: Union[str, Path],
        mode: str,
        prompt_uid_col: Optional[str] = None,
        prompt_text_col: Optional[str] = None,
        seed_prompt_text_col: Optional[str] = None,
        sut_uid_col: Optional[str] = None,
        sut_response_col: Optional[str] = None,
        sut_logprobs: bool = False,
        annotator_uid_col: Optional[str] = None,
        annotation_col: Optional[str] = None,
        jailbreak=False,
    ):
        if seed_prompt_text_col and not jailbreak:
            raise ValueError("seed_prompt_text_col is only applicable if jailbreak is True.")
        self.prompt_uid_col = prompt_uid_col
        self.prompt_text_col = prompt_text_col
        self.seed_prompt_text_col = seed_prompt_text_col
        self.sut_uid_col = sut_uid_col
        self.sut_response_col = sut_response_col
        self.sut_logprobs = sut_logprobs
        self.annotator_uid_col = annotator_uid_col
        self.annotation_col = annotation_col
        self.jailbreak = jailbreak
        super().__init__(path, mode)

    def _get_schema(self):
        if self.mode == "r":
            if self.sut_logprobs:
                raise ValueError("There is no reason to include sut_logprobs when reading annotation datasets.")
            if self.jailbreak:
                return AnnotationJailbreakSchema(
                    self._read_header(),
                    prompt_uid_col=self.prompt_uid_col,
                    prompt_text_col=self.prompt_text_col,
                    evaluated_prompt_text_col=self.seed_prompt_text_col,
                    sut_uid_col=self.sut_uid_col,
                    sut_response_col=self.sut_response_col,
                    annotator_uid_col=self.annotator_uid_col,
                    annotation_col=self.annotation_col,
                )
            return AnnotationSchema(
                self._read_header(),
                prompt_uid_col=self.prompt_uid_col,
                prompt_text_col=self.prompt_text_col,
                sut_uid_col=self.sut_uid_col,
                sut_response_col=self.sut_response_col,
                annotator_uid_col=self.annotator_uid_col,
                annotation_col=self.annotation_col,
            )
        elif self.jailbreak:
            return AnnotationJailbreakSchema.default(sut_logprobs=self.sut_logprobs)
        else:
            return AnnotationSchema.default(sut_logprobs=self.sut_logprobs)

    def row_to_item(self, row: dict) -> AnnotatedSUTInteraction:
        seed_prompt_args = {}
        if self.jailbreak:
            seed_prompt_args["evaluated_prompt"] = TextPrompt(text=row[self.schema.evaluated_prompt_text])
        prompt = TestItem(
            prompt=TextPrompt(text=row[self.schema.prompt_text]),
            source_id=row[self.schema.prompt_uid],
            context=row,
            **seed_prompt_args,
        )
        response = SUTResponse(text=row[self.schema.sut_response])
        interaction = SUTInteraction(prompt, row[self.schema.sut_uid], response)
        annotation = json.loads(row[self.schema.annotation])
        return AnnotatedSUTInteraction(
            sut_interaction=interaction, annotator_uid=row[self.schema.annotator_uid], annotation=annotation
        )

    def item_to_row(self, item: AnnotatedSUTInteraction) -> list[str]:
        """Write an AnnotatedSUTInteraction to a csv row. The last column is the annotation, which is a json string that can be deserialized with json.loads."""
        if not isinstance(item.sut_interaction.prompt.prompt, TextPrompt):
            raise ValueError(f"Error handling {item}. Can only handle TextPrompts.")
        assert item.sut_interaction.prompt.source_id is not None, "Prompt source_id is required."
        annotation = item.annotation
        if isinstance(annotation, BaseModel):
            annotation = annotation.model_dump(exclude_none=True)  # type: ignore
        annotation_json_str = json.dumps(annotation)

        data = {
            self.schema.prompt_uid: item.sut_interaction.prompt.source_id,
            self.schema.prompt_text: item.sut_interaction.prompt.prompt.text,
            self.schema.sut_uid: item.sut_interaction.sut_uid,
            self.schema.sut_response: item.sut_interaction.response.text,
            self.schema.annotator_uid: item.annotator_uid,
            self.schema.annotation: annotation_json_str,
        }

        if self.jailbreak:
            if not isinstance(item.sut_interaction.prompt.evaluated_prompt, TextPrompt):
                raise ValueError(f"Error handling {item}. Can only handle TextPrompts for evaluated_prompts.")
            data[self.schema.evaluated_prompt_text] = item.sut_interaction.prompt.evaluated_prompt.text
        if self.sut_logprobs:
            data[self.schema.sut_logprobs] = self.dump_sut_logprobs(item.sut_interaction)
        return self.schema.create_row(data)
