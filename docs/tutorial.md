# Tutorial

To help illustrate the concepts of NewHELM, we provide a series of functional (if silly) examples in the `demo_plugin` folder. This tutorial walks through those examples.

# Tests

## Creating a basic Test

[Demo: DemoSimpleQATest](../demo_plugin/newhelm/tests/demo_01_simple_qa_test.py)

Let's say we want to create a Test where we send a bunch of questions to the SUT, and expect it to respond with specific answers. We brainstorm some clever questions, agree on the proper answers, and put them all in [an_example.jsonl](https://storage.googleapis.com/mlc-aisafety-scratch-dev-7292542/an_example.jsonl).

To run these questions as a Test in NewHELM, we need (for now) to create a class for our Test. Let's call our Test `DemoSimpleQATest`.
Since this fits as a [Prompt Response Test](prompt_response_tests.md) we can have it inherit from `BasePromptResponseTest`. We now have several abstract methods we need to define.

The first phase in a PromptResponseTest is making the `TestItem`s. We want these to be our questions from `an_example.jsonl`. NewHELM uses [DependencyHelper](../newhelm/dependency_helper.py) to ensure good hygiene of data dependencies (e.g. versioning). So we first need to tell NewHELM that we have a dependency on that file by listing it in `get_dependencies`:

```py
def get_dependencies(self):
    return {
        "jsonl_questions": WebData(
            source_url="https://storage.googleapis.com/mlc-aisafety-scratch-dev-7292542/an_example.jsonl"
        ),
    }
```

We'll explore more of what DependencyHelper can do in later demos. Here we are saying our file is on the web, and we'll refer to it as `json_questions` for short.

The `make_test_items` method is where we convert our `an_example.jsonl` file into TestItems. `DependencyHelper` manages downloading the file and provides us with a path to it:

```py
with open(dependency_helper.get_local_path("jsonl_questions"), "r") as f:
```

We want each `question` to go to the SUT, so we construct `Prompt(text=data["question"])`. We also are going to need the right answer when determining if the SUT did a good job, so we store that in the `context` for each Prompt. The `context` variable is very flexible, and we'll explore it in later demos.

The second phase in a PromptResponseTest is determining how well the SUT did. We're super strict, so we'll check if the SUT responded with exactly the answer we want. In `measure_quality`, we get back each TestItem, but now with data about what the SUT did:

```py
interaction.response.completions[0].text == interaction.prompt.context
```

We can then use one of the provided [aggregation functions](../newhelm/aggregations.py) to determine how often the SUT responded correctly.

Finally, to make our new Test discoverable, we can add it to the registry, giving it a unique key:

```py
TESTS.register("demo_01", DemoSimpleQATest)
```

With our Test [installed](plugins.md), we should now be able to run our Test against any SUT in NewHELM!

```
poetry run python newhelm/main.py run-test --test demo_01 --sut demo_yes_no
```
