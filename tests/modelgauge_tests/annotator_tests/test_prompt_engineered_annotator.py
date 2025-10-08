from modelgauge.annotation import SafetyModelResponse
from modelgauge.annotators.prompt_engineered_annotator import PromptEngineeredAnnotator


def test_parse_text_pure_json():
    response_text = '{"is_safe":true,"safety_categories":[],"reasoning":"very safe"}'
    response = PromptEngineeredAnnotator._parse_text_to_safety_model_response(response_text)
    assert response == SafetyModelResponse(
        is_safe=True, safety_categories=[], is_valid=True, reasoning="very safe", raw_response=response_text
    )


def test_parse_text_safety_categories_not_list():
    response_text = '{"is_safe":true,"safety_categories":"A","reasoning":"very very safe"}'
    response = PromptEngineeredAnnotator._parse_text_to_safety_model_response(response_text)
    assert response == SafetyModelResponse(
        is_safe=True, safety_categories=[], is_valid=True, reasoning="very very safe", raw_response=response_text
    )


def test_parse_text_with_json_code_block():
    response_text = (
        "```json\n\n"
        "{\n"
        '  "is_safe": false,\n'
        '  "safety_categories": [\n    "A", "B"\n  ],\n'
        '  "is_valid": true\n'
        "}\n"
        "```"
    )
    response = PromptEngineeredAnnotator._parse_text_to_safety_model_response(response_text)
    assert response == SafetyModelResponse(
        is_safe=False, safety_categories=["A", "B"], is_valid=True, raw_response=response_text
    )


def test_parse_text_parsable_malformed_json(parent_directory):
    # these are all bad responses from a particular run
    source_filename = str(parent_directory.joinpath("..", "data", "invalid_json.txt"))
    safe_count = 0
    with open(source_filename, "r") as f:
        for line in f:
            response_text = line.strip()
            response = PromptEngineeredAnnotator._parse_text_to_safety_model_response(response_text)
            assert response.is_valid
            assert response.safety_categories == []
            assert response.raw_response == response_text
            if response.is_safe:
                safe_count += 1
    assert safe_count == 3


def test_parse_text_invalid_is_safe():
    response_text = '{"is_safe":who knows}'
    response = PromptEngineeredAnnotator._parse_text_to_safety_model_response(response_text)
    assert response == SafetyModelResponse(
        is_safe=False,
        safety_categories=[],
        is_valid=False,
        raw_response=response_text,
    )
