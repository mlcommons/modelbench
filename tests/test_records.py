from pydantic import BaseModel
from newhelm.annotation import Annotation
from newhelm.base_test import Result
from newhelm.prompt import Prompt, SUTOptions
from newhelm.record_init import InitializationRecord
from newhelm.records import TestItemRecord, TestRecord
from newhelm.single_turn_prompt_response import (
    PromptInteraction,
    PromptWithContext,
    TestItem,
)
from newhelm.sut import SUTCompletion, SUTResponse
from newhelm.typed_data import TypedData


class MockAnnotation(BaseModel):
    mock_field: str


class MockContext(BaseModel):
    context_field: str


def test_serialize_test_record():
    prompt = PromptWithContext(
        prompt=Prompt(text="some-text", options=SUTOptions(max_tokens=17)),
        context=TypedData.from_instance(MockContext(context_field="prompt-context")),
    )

    record = TestRecord(
        test_name="some-test",
        test_initialization=InitializationRecord(
            module="some-module", qual_name="test-class", args=[], kwargs={}
        ),
        dependency_versions={"d1": "v1"},
        sut_name="some-sut",
        sut_initialization=InitializationRecord(
            module="another-module", qual_name="sut-class", args=["an-arg"], kwargs={}
        ),
        test_item_records=[
            TestItemRecord(
                test_item=TestItem(
                    prompts=[prompt],
                    context=TypedData.from_instance(
                        MockContext(context_field="test-item-context")
                    ),
                ),
                interactions=[
                    PromptInteraction(
                        prompt=prompt,
                        response=SUTResponse(
                            completions=[SUTCompletion(text="sut-completion")]
                        ),
                    )
                ],
                annotations={
                    "k1": Annotation.from_instance(
                        MockAnnotation(mock_field="mock-value")
                    )
                },
                measurements={"m1": 1.0},
            )
        ],
        results=[Result(name="some-result", value=2.0)],
    )

    assert (
        record.model_dump_json(indent=2)
        == """\
{
  "test_name": "some-test",
  "test_initialization": {
    "module": "some-module",
    "qual_name": "test-class",
    "args": [],
    "kwargs": {}
  },
  "dependency_versions": {
    "d1": "v1"
  },
  "sut_name": "some-sut",
  "sut_initialization": {
    "module": "another-module",
    "qual_name": "sut-class",
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
                "echo_prompt": null,
                "top_p": null,
                "presence_penalty": null,
                "frequency_penalty": null,
                "random": null
              }
            },
            "context": {
              "module": "test_records",
              "class_name": "MockContext",
              "data": {
                "context_field": "prompt-context"
              }
            }
          }
        ],
        "context": {
          "module": "test_records",
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
                "echo_prompt": null,
                "top_p": null,
                "presence_penalty": null,
                "frequency_penalty": null,
                "random": null
              }
            },
            "context": {
              "module": "test_records",
              "class_name": "MockContext",
              "data": {
                "context_field": "prompt-context"
              }
            }
          },
          "response": {
            "completions": [
              {
                "text": "sut-completion"
              }
            ]
          }
        }
      ],
      "annotations": {
        "k1": {
          "module": "test_records",
          "class_name": "MockAnnotation",
          "data": {
            "mock_field": "mock-value"
          }
        }
      },
      "measurements": {
        "m1": 1.0
      }
    }
  ],
  "results": [
    {
      "name": "some-result",
      "value": 2.0
    }
  ]
}"""
    )


def test_round_trip_prompt_with_context():
    prompt = PromptWithContext(
        prompt=Prompt(text="some-text", options=SUTOptions(max_tokens=17)),
        context=TypedData.from_instance(MockContext(context_field="prompt-context")),
    )
    as_json = prompt.model_dump_json()
    returned = PromptWithContext.model_validate_json(as_json)
    assert prompt == returned
    assert type(prompt.get_context(MockContext)) == MockContext
