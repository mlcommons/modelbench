from abc import ABC, abstractmethod
from typing import Optional


# The first value is the preferred name.
PROMPT_UID_COLS = ["prompt_uid", "release_prompt_id"]
PROMPT_TEXT_COLS = ["prompt_text"]
SUT_UID_COLS = ["sut_uid", "sut"]
SUT_RESPONSE_COLS = ["sut_response", "response_text", "response"]
ANNOTATOR_UID_COLS = ["annotator_uid"]
ANNOTATION_COLS = ["annotation_json"]


class SchemaValidationError(ValueError):
    """Exception raised when schema validation fails."""

    def __init__(self, missing_columns):
        """missing_columns: a list where each element is a string or a list of strings. List elements are used to indicate that the column can be one of several options."""
        self.missing_columns = missing_columns
        super().__init__(str(self))

    def __str__(self):
        message = "Missing required columns:"
        for column in self.missing_columns:
            if isinstance(column, str):
                message += f"\n\t{column}"
            elif len(column) == 1:
                message += f"\n\t{column[0]}"
            else:
                message += f"\n\tone of: {column}"
        return message


class BaseSchema(ABC):
    def __init__(self, header):
        self.header = header
        self._bind_columns()
        missing_cols = self._find_missing_columns()
        if missing_cols:
            raise SchemaValidationError(missing_cols)

    def _find_column(self, columns) -> str | None:
        return next((col for col in self.header if col.lower() in columns), None)

    @abstractmethod
    def _bind_columns(self):
        pass

    @abstractmethod
    def _find_missing_columns(self) -> list[list[str]]:
        pass


class PromptSchema(BaseSchema):
    """A case-insensitive schema for a prompts file that is used as input to get SUT responses.

    Attributes:
        prompt_uid: The column name for the prompt uid.
        prompt_text: The column name for the prompt text.
    """

    def __init__(
        self,
        header: list[str],
        prompt_uid_col: Optional[str] = None,
        prompt_text_col: Optional[str] = None,
    ):
        self.expected_prompt_uid_cols = [prompt_uid_col] if prompt_uid_col else PROMPT_UID_COLS
        self.expected_prompt_text_cols = [prompt_text_col] if prompt_text_col else PROMPT_TEXT_COLS
        self.prompt_uid = None
        self.prompt_text = None
        super().__init__(header)

    def _bind_columns(self):
        self.prompt_uid = self._find_column(self.expected_prompt_uid_cols)
        self.prompt_text = self._find_column(self.expected_prompt_text_cols)

    def _find_missing_columns(self) -> list[list[str]]:
        missing_columns = []
        if not self.prompt_uid:
            missing_columns.append(self.expected_prompt_uid_cols)
        if not self.prompt_text:
            missing_columns.append(self.expected_prompt_text_cols)
        return missing_columns


class PromptResponseSchema(PromptSchema):
    """A schema for a prompt + response file that is used as prompt-response output or annotation input.
    Attributes:
        prompt_uid: The column name for the prompt uid. (same as PromptSchema)
        prompt_text: The column name for the prompt text. (same as PromptSchema)
        sut_uid: The column name for the SUT uid.
        sut_response: The column name for the SUT response.
    """

    def __init__(
        self,
        header: list[str],
        prompt_uid_col: Optional[str] = None,
        prompt_text_col: Optional[str] = None,
        sut_uid_col: Optional[str] = None,
        sut_response_col: Optional[str] = None,
    ):
        self.expected_sut_uid_cols = [sut_uid_col] if sut_uid_col else SUT_UID_COLS
        self.expected_sut_response_cols = [sut_response_col] if sut_response_col else SUT_RESPONSE_COLS
        self.sut_uid = None
        self.sut_response = None
        super().__init__(header, prompt_uid_col=prompt_uid_col, prompt_text_col=prompt_text_col)

    def _bind_columns(self):
        super()._bind_columns()
        self.sut_uid = self._find_column(self.expected_sut_uid_cols)
        self.sut_response = self._find_column(self.expected_sut_response_cols)

    def _find_missing_columns(self) -> list[list[str]]:
        missing_columns = super()._find_missing_columns()
        if not self.sut_uid:
            missing_columns.append(self.expected_sut_uid_cols)
        if not self.sut_response:
            missing_columns.append(self.expected_sut_response_cols)
        return missing_columns


class AnnotationSchema(PromptResponseSchema):
    """A schema for a prompt + response + annotation file that is used as annotation output.
    Attributes:
        prompt_uid: The column name for the prompt uid. (same as PromptSchema)
        prompt_text: The column name for the prompt text. (same as PromptSchema)
        sut_uid: The column name for the SUT uid. (same as PromptResponseSchema)
        sut_response: The column name for the SUT response. (same as PromptResponseSchema)
        annotator_uid: The column name for the annotator uid.
        annotation: The column name for the text annotation.
    """

    def __init__(
        self,
        header: list[str],
        prompt_uid_col: Optional[str] = None,
        prompt_text_col: Optional[str] = None,
        sut_uid_col: Optional[str] = None,
        sut_response_col: Optional[str] = None,
        annotator_uid_col: Optional[str] = None,
        annotation_col: Optional[str] = None,
    ):
        self.expected_annotator_uid_cols = [annotator_uid_col] if annotator_uid_col else ANNOTATOR_UID_COLS
        self.expected_annotation_cols = [annotation_col] if annotation_col else ANNOTATION_COLS
        self.annotator_uid = None
        self.annotation = None
        super().__init__(
            header,
            prompt_uid_col=prompt_uid_col,
            prompt_text_col=prompt_text_col,
            sut_uid_col=sut_uid_col,
            sut_response_col=sut_response_col,
        )

    def _bind_columns(self):
        super()._bind_columns()
        self.annotator_uid = self._find_column(self.expected_annotator_uid_cols)
        self.annotation = self._find_column(self.expected_annotation_cols)

    def _find_missing_columns(self) -> list[list[str]]:
        missing_columns = super()._find_missing_columns()
        if not self.annotator_uid:
            missing_columns.append(self.expected_annotator_uid_cols)
        if not self.annotation:
            missing_columns.append(self.expected_annotation_cols)
        return missing_columns


# Schemas with preferred names.
DEFAULT_PROMPT_SCHEMA = PromptSchema([PROMPT_UID_COLS[0], PROMPT_TEXT_COLS[0]])
DEFAULT_PROMPT_RESPONSE_SCHEMA = PromptResponseSchema(
    [PROMPT_UID_COLS[0], PROMPT_TEXT_COLS[0], SUT_UID_COLS[0], SUT_RESPONSE_COLS[0]]
)
DEFAULT_ANNOTATION_SCHEMA = AnnotationSchema(
    [
        PROMPT_UID_COLS[0],
        PROMPT_TEXT_COLS[0],
        SUT_UID_COLS[0],
        SUT_RESPONSE_COLS[0],
        ANNOTATOR_UID_COLS[0],
        ANNOTATION_COLS[0],
    ]
)
