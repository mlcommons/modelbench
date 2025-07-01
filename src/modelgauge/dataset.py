import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Optional, Union, Any, Sequence

from pydantic import BaseModel

from modelgauge.data_schema import (
    DEFAULT_ANNOTATION_SCHEMA,
    DEFAULT_PROMPT_RESPONSE_SCHEMA,
    DEFAULT_PROMPT_SCHEMA,
    AnnotationSchema,
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
        self.schema = None
        self._init_schema()  # Initialized by subclass.

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
    def _init_schema(self):
        """Initialize dataset schema `self.schema`. To be implemented by subclasses."""
        pass

    def _read_header(self) -> list[str]:
        """Read the header row from a CSV file."""
        if self.mode != "r":
            raise RuntimeError("Can only read header in read mode.")
        if self.file is None:
            with self:
                header = self.reader.fieldnames
        else:
            header = self.reader.fieldnames
        return header

    def header_columns(self) -> Sequence[str]:
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


class PromptDataset(BaseDataset):
    """Dataset for reading prompts as TestItems from a CSV file. Read only."""

    def __init__(self, path: Union[str, Path]):
        super().__init__(path, "r")

    def _init_schema(self):
        self.schema = PromptSchema(self._read_header())

    def row_to_item(self, row: dict) -> TestItem:
        """Convert a single prompt row to a TestItem."""
        return TestItem(
            prompt=TextPrompt(text=row[self.schema.prompt_text]),
            source_id=row[self.schema.prompt_uid],
            context=row,
        )


class PromptResponseDataset(BaseDataset):
    """Dataset for prompt-response CSV data. Read or write."""

    def _init_schema(self):
        if self.mode == "r":
            self.schema = PromptResponseSchema(self._read_header())
        else:
            self.schema = DEFAULT_PROMPT_RESPONSE_SCHEMA

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

        return [
            item.prompt.source_id,
            item.prompt.prompt.text,
            item.sut_uid,
            item.response.text,
        ]


class AnnotationDataset(BaseDataset):
    """Dataset for annotated prompt-response CSV data. Read or write."""

    def _init_schema(self):
        if self.mode == "r":
            self.schema = AnnotationSchema(self._read_header())
        else:
            self.schema = DEFAULT_ANNOTATION_SCHEMA

    def row_to_item(self, row: dict) -> AnnotatedSUTInteraction:
        prompt = TestItem(
            prompt=TextPrompt(text=row[self.schema.prompt_text]),
            source_id=row[self.schema.prompt_uid],
            context=row,
        )
        response = SUTResponse(text=row[self.schema.sut_response])
        interaction = SUTInteraction(prompt, row[self.schema.sut_uid], response)
        annotation = row.get(self.schema.annotation)
        return AnnotatedSUTInteraction(
            sut_interaction=interaction, annotator_uid=row[self.schema.annotator_uid], annotation=annotation
        )

    def item_to_row(self, item: AnnotatedSUTInteraction) -> list[str]:
        if not isinstance(item.sut_interaction.prompt.prompt, TextPrompt):
            raise ValueError(f"Error handling {item}. Can only handle TextPrompts.")
        annotation = item.annotation.model_dump() if isinstance(item.annotation, BaseModel) else item.annotation
        return [
            item.sut_interaction.prompt.source_id,
            item.sut_interaction.prompt.prompt.text,
            item.sut_interaction.sut_uid,
            item.sut_interaction.response.text,
            item.annotator_uid,
            annotation,
        ]
