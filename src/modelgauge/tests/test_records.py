import datetime
from modelgauge.annotation import Annotation
from modelgauge.base_test import TestResult
from modelgauge.prompt import SUTOptions, TextPrompt
from modelgauge.record_init import InitializationRecord
from modelgauge.records import TestItemRecord, TestRecord
from modelgauge.single_turn_prompt_response import (
    PromptInteractionAnnotations,
    PromptWithContext,
    SUTCompletionAnnotations,
    SUTResponseAnnotations,
    TestItem,
)
from modelgauge.sut import SUTCompletion
from pydantic import BaseModel


class MockAnnotation(BaseModel):
    mock_field: str


class MockContext(BaseModel):
    context_field: str


class MockResult(BaseModel):
    mock_result: float


def test_serialize_test_record():
    prompt = PromptWithContext(
        prompt=TextPrompt(text="some-text", options=SUTOptions(max_tokens=17)),
        source_id="id01",
        context=MockContext(context_field="prompt-context"),
    )

    record = TestRecord(
        run_timestamp=datetime.datetime(
            2017,
            8,
            21,
            11,
            47,
            0,
            123456,
            tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=61200), "MST"),
        ),
        test_uid="some-test",
        test_initialization=InitializationRecord(
            module="some-module", class_name="test-class", args=[], kwargs={}
        ),
        dependency_versions={"d1": "v1"},
        sut_uid="some-sut",
        sut_initialization=InitializationRecord(
            module="another-module", class_name="sut-class", args=["an-arg"], kwargs={}
        ),
        test_item_records=[
            TestItemRecord(
                test_item=TestItem(
                    prompts=[prompt],
                    context=MockContext(context_field="test-item-context"),
                ),
                interactions=[
                    PromptInteractionAnnotations(
                        prompt=prompt,
                        response=SUTResponseAnnotations(
                            completions=[
                                SUTCompletionAnnotations(
                                    completion=SUTCompletion(text="sut-completion"),
                                    annotations={
                                        "k1": Annotation.from_instance(
                                            MockAnnotation(mock_field="mock-value")
                                        )
                                    },
                                )
                            ]
                        ),
                    )
                ],
                measurements={"m1": 1.0},
            )
        ],
        test_item_exceptions=[],
        result=TestResult.from_instance(MockResult(mock_result=2.0)),
    )

    assert (
        record.model_dump_json(indent=2)
        == """\
{
  "run_timestamp": "2017-08-21T11:47:00.123456-07:00",
  "test_uid": "some-test",
  "test_initialization": {
    "module": "some-module",
    "class_name": "test-class",
    "args": [],
    "kwargs": {}
  },
  "dependency_versions": {
    "d1": "v1"
  },
  "sut_uid": "some-sut",
  "sut_initialization": {
    "module": "another-module",
    "class_name": "sut-class",
    "args": [
      "an-arg"
    ],
    "kwargs": {}
  },
  "test_item_records": [
    {
      "test_item": {
        "prompts": [
          {
            "prompt": {
              "text": "some-text",
              "options": {
                "num_completions": 1,
                "max_tokens": 17,
                "temperature": null,
                "top_k_per_token": null,
                "stop_sequences": null,
                "top_p": null,
                "presence_penalty": null,
                "frequency_penalty": null,
                "random": null,
                "top_logprobs": null
              }
            },
            "source_id": "id01",
            "context_internal": {
              "module": "tests.test_records",
              "class_name": "MockContext",
              "data": {
                "context_field": "prompt-context"
              }
            }
          }
        ],
        "context_internal": {
          "module": "tests.test_records",
          "class_name": "MockContext",
          "data": {
            "context_field": "test-item-context"
          }
        }
      },
      "interactions": [
        {
          "prompt": {
            "prompt": {
              "text": "some-text",
              "options": {
                "num_completions": 1,
                "max_tokens": 17,
                "temperature": null,
                "top_k_per_token": null,
                "stop_sequences": null,
                "top_p": null,
                "presence_penalty": null,
                "frequency_penalty": null,
                "random": null,
                "top_logprobs": null
              }
            },
            "source_id": "id01",
            "context_internal": {
              "module": "tests.test_records",
              "class_name": "MockContext",
              "data": {
                "context_field": "prompt-context"
              }
            }
          },
          "response": {
            "completions": [
              {
                "completion": {
                  "text": "sut-completion",
                  "top_logprobs": null
                },
                "annotations": {
                  "k1": {
                    "module": "tests.test_records",
                    "class_name": "MockAnnotation",
                    "data": {
                      "mock_field": "mock-value"
                    }
                  }
                }
              }
            ]
          }
        }
      ],
      "measurements": {
        "m1": 1.0
      }
    }
  ],
  "test_item_exceptions": [],
  "result": {
    "module": "tests.test_records",
    "class_name": "MockResult",
    "data": {
      "mock_result": 2.0
    }
  }
}"""
    )


def test_round_trip_prompt_with_context():
    prompt = PromptWithContext(
        prompt=TextPrompt(text="some-text", options=SUTOptions(max_tokens=17)),
        source_id="id01",
        context=MockContext(context_field="prompt-context"),
    )
    as_json = prompt.model_dump_json()
    returned = PromptWithContext.model_validate_json(as_json)
    assert prompt == returned
    assert type(returned.context) == MockContext
    assert returned.source_id == "id01"
