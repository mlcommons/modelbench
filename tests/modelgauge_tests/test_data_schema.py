import pytest

from modelgauge.data_schema import (
    DEFAULT_PROMPT_RESPONSE_SCHEMA,
    DEFAULT_PROMPT_SCHEMA,
    PROMPT_TEXT_COLS,
    PROMPT_UID_COLS,
    PromptResponseSchema,
    PromptSchema,
    SchemaValidationError,
)


def test_schema_validation_error():
    error = SchemaValidationError(["one", "two"])
    assert error.missing_columns == ["one", "two"]
    assert str(error) == "Missing required columns:\n\tone\n\ttwo"


def test_schema_validation_error_multiple_options():
    error = SchemaValidationError([["one", "a"], "two"])
    assert error.missing_columns == [["one", "a"], "two"]
    assert str(error) == "Missing required columns:\n\tone of: ['one', 'a']\n\ttwo"


@pytest.mark.parametrize(
    "header",
    [
        ["prompt_uid", "prompt_text"],  # Preferred names.
        ["Prompt_UID", "Prompt_Text"],  # Case-insensitive
        ["release_prompt_id", "prompt_text"],
        ["release_prompt_id", "prompt_text", "random_column"],  # Extra columns are allowed.
    ],
)
def test_valid_prompt_schema(header):
    schema = PromptSchema(header)
    assert schema.prompt_uid == header[0]
    assert schema.prompt_text == header[1]


def test_invalid_prompt_schema():
    header = ["random_column", "random_column_2"]
    with pytest.raises(SchemaValidationError) as e:
        schema = PromptSchema(header)
        assert set(e.missing_columns) == {PROMPT_UID_COLS, PROMPT_TEXT_COLS}


def test_default_prompt_schema():
    assert DEFAULT_PROMPT_SCHEMA.prompt_uid == "prompt_uid"
    assert DEFAULT_PROMPT_SCHEMA.prompt_text == "prompt_text"


@pytest.mark.parametrize(
    "header",
    [
        ["prompt_uid", "prompt_text", "sut_uid", "sut_response"],  # Preferred names.
        ["prompt_UID", "Prompt_Text", "SUT_UID", "SUT_Response"],  # Case-insensitive
        ["release_prompt_id", "prompt_text", "sut", "response"],
    ],
)
def test_valid_prompt_response_schema(header):
    schema = PromptResponseSchema(header)
    assert schema.prompt_uid == header[0]
    assert schema.prompt_text == header[1]
    assert schema.sut_uid == header[2]
    assert schema.sut_response == header[3]


def test_valid_prompt_invalid_response_schema():
    header = ["prompt_uid", "prompt_text", "random_column", "random_column_2"]
    with pytest.raises(SchemaValidationError) as e:
        schema = PromptResponseSchema(header)
        assert set(e.missing_columns) == {SUT_UID_COLS, SUT_RESPONSE_COLS}


def test_invalid_prompt_valid_response_schema():
    header = ["random_column", "random_column_2", "prompt_uid", "prompt_text"]
    with pytest.raises(SchemaValidationError) as e:
        schema = PromptResponseSchema(header)
        assert set(e.missing_columns) == {PROMPT_UID_COLS, PROMPT_TEXT_COLS}


def test_default_prompt_response_schema():
    assert DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_uid == "prompt_uid"
    assert DEFAULT_PROMPT_RESPONSE_SCHEMA.prompt_text == "prompt_text"
    assert DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_uid == "sut_uid"
    assert DEFAULT_PROMPT_RESPONSE_SCHEMA.sut_response == "sut_response"
