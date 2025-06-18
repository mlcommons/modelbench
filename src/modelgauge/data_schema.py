# The first value is the preferred name.
PROMPT_UID_COLS = ["prompt_uid", "release_prompt_id"]
PROMPT_TEXT_COLS = ["prompt_text"]
SUT_UID_COLS = ["sut_uid", "sut"]
SUT_RESPONSE_COLS = ["sut_response", "response_text", "response"]


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


class PromptSchema:
    """A case-insensitive schema for a prompts file.

    Attributes:
        prompt_uid: The column name for the prompt uid.
        prompt_text: The column name for the prompt text.
    """

    def __init__(self, header: list[str]):
        self.prompt_uid = self._find_column(header, PROMPT_UID_COLS)
        self.prompt_text = self._find_column(header, PROMPT_TEXT_COLS)
        self._validate()

    def _find_column(self, header, columns):
        return next((col for col in header if col.lower() in columns), None)

    def _validate(self):
        """Validates that all required columns were found in the header.

        Raises:
            SchemaValidationError: If any required columns are missing.
        """
        missing = []
        if self.prompt_uid is None:
            missing.append(PROMPT_UID_COLS)
        if self.prompt_text is None:
            missing.append(PROMPT_TEXT_COLS)

        if missing:
            raise SchemaValidationError(missing)


class PromptResponseSchema(PromptSchema):
    """A schema for a prompt + response file that is used as annotation input.
    Attributes:
        prompt_uid: The column name for the prompt uid. (same as PromptSchema)
        prompt_text: The column name for the prompt text. (same as PromptSchema)
        sut_uid: The column name for the SUT uid.
        sut_response: The column name for the SUT response.
    """

    def __init__(self, header: list[str]):
        self.sut_uid = self._find_column(header, SUT_UID_COLS)
        self.sut_response = self._find_column(header, SUT_RESPONSE_COLS)
        super().__init__(header)  # Iniitalize the prompt schema columns and then validate.

    def _validate(self):
        missing = []
        # Validate that the prompt schema is valid
        try:
            super()._validate()
        except SchemaValidationError as e:
            missing.extend(e.missing_columns)
        # Validate that the SUT uid and response columns are present
        if self.sut_uid is None:
            missing.append(SUT_UID_COLS)
        if self.sut_response is None:
            missing.append(SUT_RESPONSE_COLS)
        if missing:
            raise SchemaValidationError(missing)


# Schemas with preferred names.
DEFAULT_PROMPT_SCHEMA = PromptSchema([PROMPT_UID_COLS[0], PROMPT_TEXT_COLS[0]])
DEFAULT_PROMPT_RESPONSE_SCHEMA = PromptResponseSchema(
    [PROMPT_UID_COLS[0], PROMPT_TEXT_COLS[0], SUT_UID_COLS[0], SUT_RESPONSE_COLS[0]]
)
