import pytest

from modelgauge.data_schema import (
    ANNOTATOR_UID_COLS,
    ANNOTATION_COLS,
    PROMPT_TEXT_COLS,
    PROMPT_UID_COLS,
    SEED_PROMPT_TEXT_COLS,
    SUT_UID_COLS,
    SUT_RESPONSE_COLS,
    AnnotationJailbreakSchema,
    AnnotationSchema,
    PromptJailbreakSchema,
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


def test_parameterized_prompt_schema():
    header = ["prompt_uid", "prompt_text", "uid2", "text2"]
    schema = PromptSchema(header, prompt_uid_col="uid2", prompt_text_col="text2")
    assert schema.prompt_uid == "uid2"
    assert schema.prompt_text == "text2"


def test_invalid_prompt_schema():
    header = ["random_column", "random_column_2"]
    with pytest.raises(SchemaValidationError) as e:
        schema = PromptSchema(header)
    assert e.value.missing_columns == [PROMPT_UID_COLS, PROMPT_TEXT_COLS]


def test_invalid_parameterized_prompt_schema():
    header = ["prompt_uid", "prompt_text"]
    with pytest.raises(SchemaValidationError) as e:
        schema = PromptSchema(header, prompt_uid_col="uid2", prompt_text_col="text2")
    assert e.value.missing_columns == [["uid2"], ["text2"]]


def test_default_prompt_schema():
    schema = PromptSchema.default()
    assert schema.prompt_uid == "prompt_uid"
    assert schema.prompt_text == "prompt_text"


def test_valid_prompt_jailbreak_schema():
    header = ["prompt_uid", "prompt_text", "seed_prompt_text"]
    schema = PromptJailbreakSchema(header)
    assert schema.prompt_uid == header[0]
    assert schema.prompt_text == header[1]
    assert schema.evaluated_prompt_text == header[2]


def test_parameterized_prompt_jailbreak_schema():
    header = ["prompt_uid", "prompt_text", "seed_prompt_text2"]
    schema = PromptJailbreakSchema(header, evaluated_prompt_text_col=header[2])
    assert schema.prompt_uid == header[0]
    assert schema.prompt_text == header[1]
    assert schema.evaluated_prompt_text == header[2]


def test_valid_prompt_invalid_jailbreak_schema():
    header = ["prompt_uid", "prompt_text", "random_column"]
    with pytest.raises(SchemaValidationError) as e:
        schema = PromptJailbreakSchema(header)
    assert e.value.missing_columns == [SEED_PROMPT_TEXT_COLS]


def test_default_prompt_jailbreak_schema():
    schema = PromptJailbreakSchema.default()
    assert schema.prompt_uid == "prompt_uid"
    assert schema.prompt_text == "prompt_text"
    assert schema.evaluated_prompt_text == "seed_prompt_text"


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


def test_parameterized_prompt_response_schema():
    header = ["prompt_uid", "prompt_text", "sut_uid2", "sut_response2"]
    schema = PromptResponseSchema(header, sut_uid_col=header[2], sut_response_col=header[3])
    assert schema.prompt_uid == header[0]
    assert schema.prompt_text == header[1]
    assert schema.sut_uid == header[2]
    assert schema.sut_response == header[3]


def test_valid_prompt_invalid_response_schema():
    header = ["prompt_uid", "prompt_text", "random_column", "random_column_2"]
    with pytest.raises(SchemaValidationError) as e:
        schema = PromptResponseSchema(header)
    assert e.value.missing_columns == [SUT_UID_COLS, SUT_RESPONSE_COLS]


def test_invalid_prompt_valid_response_schema():
    header = ["random_column", "random_column_2", "sut_uid", "sut_response"]
    with pytest.raises(SchemaValidationError) as e:
        schema = PromptResponseSchema(header)
    assert e.value.missing_columns == [PROMPT_UID_COLS, PROMPT_TEXT_COLS]


def test_invalid_prompt_response_schema_parameterized():
    header = ["random_column", "sut_response", "prompt_uid", "prompt_text"]
    with pytest.raises(SchemaValidationError) as e:
        schema = PromptResponseSchema(header, sut_uid_col="sut_uid2")
    assert e.value.missing_columns == [["sut_uid2"]]


def test_default_prompt_response_schema():
    def check_base_header(schema):
        assert schema.prompt_uid == "prompt_uid"
        assert schema.prompt_text == "prompt_text"
        assert schema.sut_uid == "sut_uid"
        assert schema.sut_response == "sut_response"

    schema = PromptResponseSchema.default()
    check_base_header(schema)
    assert len(schema.header) == 4

    schema = PromptResponseSchema.default(sut_logprobs=True)
    check_base_header(schema)
    assert schema.sut_logprobs == "sut_logprobs"
    assert len(schema.header) == 5


@pytest.mark.parametrize(
    "header",
    [
        # Preferred names
        ["prompt_uid", "prompt_text", "sut_uid", "sut_response", "annotator_uid", "annotation_json"],
        # Case-insensitive
        ["prompt_UID", "Prompt_Text", "SUT_UID", "SUT_Response", "Annotator_UID", "annotation_JSON"],
        # Extra columns are allowed
        ["prompt_uid", "prompt_text", "sut_uid", "sut_response", "annotator_uid", "annotation_json", "extra_col"],
    ],
)
def test_valid_annotation_schema(header):
    schema = AnnotationSchema(header)
    assert schema.prompt_uid == header[0]
    assert schema.prompt_text == header[1]
    assert schema.sut_uid == header[2]
    assert schema.sut_response == header[3]
    assert schema.annotator_uid == header[4]
    assert schema.annotation == header[5]


def test_parameterized_annotation_schema():
    header = ["prompt_uid", "prompt_text", "sut_uid", "sut_response", "a_uid", "a"]
    schema = AnnotationSchema(header, annotator_uid_col=header[4], annotation_col=header[5])
    assert schema.prompt_uid == header[0]
    assert schema.prompt_text == header[1]
    assert schema.sut_uid == header[2]
    assert schema.sut_response == header[3]
    assert schema.annotator_uid == header[4]
    assert schema.annotation == header[5]


def test_valid_prompt_response_invalid_annotation_schema():
    header = ["prompt_uid", "prompt_text", "sut_uid", "sut_response", "random_column", "random_column_2"]
    with pytest.raises(SchemaValidationError) as e:
        schema = AnnotationSchema(header)
    assert e.value.missing_columns == [ANNOTATOR_UID_COLS, ANNOTATION_COLS]


def test_invalid_prompt_response_valid_annotation_schema():
    header = ["random_1", "random_2", "random_3", "random_4", "annotator_uid", "annotation_json"]
    with pytest.raises(SchemaValidationError) as e:
        schema = AnnotationSchema(header)
    assert e.value.missing_columns == [
        PROMPT_UID_COLS,
        PROMPT_TEXT_COLS,
        SUT_UID_COLS,
        SUT_RESPONSE_COLS,
    ]


def test_invalid_annotation_schema_parameterized():
    header = ["random_column", "annotation_json", "prompt_uid", "prompt_text", "sut_uid", "sut_response"]
    with pytest.raises(SchemaValidationError) as e:
        schema = AnnotationSchema(header, annotator_uid_col="annotator_uid2")
    # When a custom column name is provided, it will be a single string in the missing_columns list
    assert e.value.missing_columns == [["annotator_uid2"]]


def test_default_annotation_schema():
    def check_base_header(schema):
        assert schema.prompt_uid == "prompt_uid"
        assert schema.prompt_text == "prompt_text"
        assert schema.sut_uid == "sut_uid"
        assert schema.sut_response == "sut_response"
        assert schema.annotator_uid == "annotator_uid"
        assert schema.annotation == "annotation_json"

    schema = AnnotationSchema.default()
    check_base_header(schema)
    assert len(schema.header) == 6

    schema = AnnotationSchema.default(sut_logprobs=True)
    check_base_header(schema)
    assert schema.sut_logprobs == "sut_logprobs"
    assert len(schema.header) == 7


def test_valid_annotation_jailbreak_schema():
    header = [
        "prompt_uid",
        "prompt_text",
        "seed_prompt_text",
        "sut_uid",
        "sut_response",
        "annotator_uid",
        "annotation_json",
    ]
    schema = AnnotationJailbreakSchema(header)
    assert schema.prompt_uid == header[0]
    assert schema.prompt_text == header[1]
    assert schema.evaluated_prompt_text == header[2]
    assert schema.sut_uid == header[3]
    assert schema.sut_response == header[4]
    assert schema.annotator_uid == header[5]
    assert schema.annotation == header[6]


def test_parameterized_annotation_jailbreak_schema():
    header = ["prompt_uid", "prompt_text", "seed", "sut_uid", "sut_response", "annotator_uid", "annotation_json"]
    schema = AnnotationJailbreakSchema(header, evaluated_prompt_text_col=header[2])
    assert schema.prompt_uid == header[0]
    assert schema.prompt_text == header[1]
    assert schema.evaluated_prompt_text == header[2]
    assert schema.sut_uid == header[3]
    assert schema.sut_response == header[4]
    assert schema.annotator_uid == header[5]
    assert schema.annotation == header[6]


def test_invalid_annotation_jailbreak_schema_parameterized():
    header = ["prompt_uid", "prompt_text", "seed", "sut_uid", "sut_response", "annotator_uid", "annotation_json"]
    with pytest.raises(SchemaValidationError) as e:
        schema = AnnotationJailbreakSchema(header, evaluated_prompt_text_col="seed2")
    # When a custom column name is provided, it will be a single string in the missing_columns list
    assert e.value.missing_columns == [["seed2"]]


def test_default_annotation_jailbreak_schema():
    def check_base_header(schema):
        assert schema.prompt_uid == "prompt_uid"
        assert schema.prompt_text == "prompt_text"
        assert schema.evaluated_prompt_text == "seed_prompt_text"
        assert schema.sut_uid == "sut_uid"
        assert schema.sut_response == "sut_response"
        assert schema.annotator_uid == "annotator_uid"
        assert schema.annotation == "annotation_json"

    schema = AnnotationJailbreakSchema.default()
    check_base_header(schema)
    assert len(schema.header) == 7

    schema = AnnotationJailbreakSchema.default(sut_logprobs=True)
    check_base_header(schema)
    assert schema.sut_logprobs == "sut_logprobs"
    assert len(schema.header) == 8
