import pytest
from modelgauge.prompt import TextPrompt
from modelgauge.single_turn_prompt_response import (
    MeasuredTestItem,
    PromptInteractionAnnotations,
    PromptWithContext,
    SUTCompletionAnnotations,
    SUTResponseAnnotations,
    TestItem,
    TestItemAnnotations,
)
from modelgauge.sut import SUTCompletion
from modelgauge.tests.discrim_eval import (
    DiscrimEval,
    DiscrimEvalResult,
    DiscrimEvalTestItemContext,
)


def _make_test_item(template_id=0, age=60, gender="male", race="white"):
    """Default parameters construct a prompt with baseline demographic values."""
    dummy_prompt = PromptWithContext(prompt=TextPrompt(text="prompt"), source_id="id")
    return TestItem(
        prompts=[dummy_prompt],
        context=DiscrimEvalTestItemContext(
            question_template_id=template_id, age=age, gender=gender, race=race
        ),
    )


def _make_measure_quality_input(sut_response_text):
    test_item = _make_test_item()
    return TestItemAnnotations(
        test_item=test_item,
        interactions=[
            PromptInteractionAnnotations(
                prompt=test_item.prompts[0],
                response=SUTResponseAnnotations(
                    completions=[
                        SUTCompletionAnnotations(
                            completion=SUTCompletion(text=sut_response_text)
                        )
                    ]
                ),
            )
        ],
    )


def _make_measured_test_item(yes=1, no=0, **kwargs):
    return MeasuredTestItem(
        test_item=_make_test_item(**kwargs),
        measurements={"yes": yes, "no": no},
    )


@pytest.mark.parametrize(
    "sut_response_text,expected_measurements",
    [
        ("Yes", {"yes": 1, "no": 0}),
        ("No", {"yes": 0, "no": 1}),
        ("Yes/no", {"yes": 0, "no": 0}),
    ],
)
def test_measure_quality_correctly_maps_decision(
    sut_response_text, expected_measurements
):
    test = DiscrimEval("uid", "explicit")
    measurements = test.measure_quality(_make_measure_quality_input(sut_response_text))
    for key in expected_measurements:
        assert measurements[key] == expected_measurements[key]


@pytest.mark.parametrize(
    "sut_response_text,decision",
    [
        ("Yes.", "yes"),
        ("Yes, \n", "yes"),
        ("No!!", "no"),
        (" yes", "yes"),
        ("\tYes", "yes"),
        ("\n\n No", "no"),
    ],
)
def test_measure_quality_ignores_whitespace_and_punctuation(
    sut_response_text, decision
):
    test = DiscrimEval("uid", "explicit")
    measurements = test.measure_quality(_make_measure_quality_input(sut_response_text))
    assert measurements[decision] == 1


def test_prepare_data_correct_baselines():
    test = DiscrimEval("uid", "explicit")
    df = test._prepare_regression_model_data([_make_measured_test_item()])
    assert df.gender.cat.categories[0] == "male"
    assert df.race.cat.categories[0] == "white"


def test_prepare_data_skips_invalid_responses():
    test = DiscrimEval("uid", "explicit")
    df = test._prepare_regression_model_data(
        [
            _make_measured_test_item(yes=0, no=0, gender="female"),
            _make_measured_test_item(yes=1, no=0, gender="non-binary"),
        ]
    )
    assert df.gender.to_list() == ["non-binary"]


def test_assertion_fails_on_all_invalid_responses():
    test = DiscrimEval("uid", "explicit")
    with pytest.raises(AssertionError, match="SUT did not return any valid responses"):
        test.aggregate_measurements([_make_measured_test_item(yes=0, no=0)])


def test_aggregate_measurements_correct_result_object(mocker):
    mock_fit = mocker.MagicMock()
    mock_fit.fit.return_value = mock_fit
    mock_fit.params = {
        "Intercept": 0,
        "age": -0.01,
        "gender[T.female]": 0.02,
        "gender[T.non-binary]": 0.03,
        "race[T.asian]": 0.04,
        "race[T.black]": 0.05,
        "race[T.hispanic]": 0.06,
        "race[T.native american]": 0.07,
    }
    mocker.patch("statsmodels.formula.api.glm", return_value=mock_fit)
    test = DiscrimEval("uid", "explicit")
    result = test.aggregate_measurements([_make_measured_test_item()])
    assert result == DiscrimEvalResult(
        discrim_score_age=-0.01,
        discrim_score_gender={"female": 0.02, "non-binary": 0.03},
        discrim_score_race={
            "asian": 0.04,
            "black": 0.05,
            "hispanic": 0.06,
            "native american": 0.07,
        },
    )
