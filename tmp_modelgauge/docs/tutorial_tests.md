# Test Demos

## Making your own Test

We think the best way to learn from this tutorial is to use it to create your own Test. The easiest way to do that is to [check out the repository locally](dev_quick_start.md) and add your own file(s) to the `modelgauge/tests/` directory. If you do not have a local copy, you could start on [creating your own plugin](plugins.md) first, or just ignore any of the parts in the tutorial about "installed".

## Creating a basic Test

[Demo: DemoSimpleQATest](https://github.com/mlcommons/modelgauge/blob/main/demo_plugin/modelgauge/tests/demo_01_simple_qa_test.py)

Let's say we want to create a Test where we send a bunch of questions to the SUT, and expect it to respond with specific answers. We brainstorm some clever questions, agree on the proper answers, and put them all in [an_example.jsonl](https://github.com/mlcommons/modelgauge/raw/main/demo_plugin/web_data/an_example.jsonl).

To run these questions as a Test in ModelGauge, we need (for now) to create a class for our Test. Let's call our Test `DemoSimpleQATest`.
Since this fits as a [Prompt Response Test](prompt_response_tests.md) we can have it inherit from `BasePromptResponseTest`. We also need to decorate that class with `modelgauge_test` to provide ModelGauge some information about the test:

```py
@modelgauge_test(requires_sut_capabilities=[AcceptsTextPrompt])
class DemoSimpleQATest(PromptResponseTest):
```

We now have several abstract methods we need to define.

### 1. Make the test items
The first phase in a PromptResponseTest is making the `TestItem`s. We want these to be our questions from `an_example.jsonl`. ModelGauge uses [DependencyHelper](https://github.com/mlcommons/modelgauge/blob/main/modelgauge/dependency_helper.py) to ensure good hygiene of data dependencies (e.g. versioning). So we first need to tell ModelGauge that we have a dependency on that file by listing it in `get_dependencies`:

```py
def get_dependencies(self):
    return {
        "jsonl_questions": WebData(
            source_url="https://github.com/mlcommons/modelgauge/raw/main/demo_plugin/web_data/an_example.jsonl"
        ),
    }
```

We'll explore more of what DependencyHelper can do in later demos. Here we are saying our file is on the web, and we'll refer to it as `json_questions` for short.

The `make_test_items` method is where we convert our `an_example.jsonl` file into TestItems. `DependencyHelper` manages downloading the file and provides us with a path to it:

```py
with open(dependency_helper.get_local_path("jsonl_questions"), "r") as f:
```

We want each `question` to go to the SUT, so we construct `Prompt(text=data["question"])`. We also are going to need the right answer when determining if the SUT did a good job, so we store that in the `context` for each Prompt. The `context` variable is very flexible, and we'll explore it in later demos.

### 2. Measure each SUT response
The second phase in a PromptResponseTest is determining how well the SUT did. We're super strict, so we'll check if the SUT responded with exactly the answer we want. In `measure_quality`, we get back each TestItem, but now with data about what the SUT did:

```py
interaction.response.completions[0].text == interaction.prompt.context
```

### 3. Aggregate the measurements

Finally, we implement `aggregate_measurements` to determine how often the SUT responded correctly. You can write your own custom code here, but we make use of one of the provided [aggregation functions](https://github.com/mlcommons/modelgauge/blob/main/modelgauge/aggregations.py).

### 4. Make your Test accessible
Finally, to make our new Test discoverable, we can add it to the registry, giving it a unique key:

```py
TESTS.register(DemoSimpleQATest, "demo_01")
```

> [!NOTE]
> If you are writing your own file, give the Test a different key, as `demo_01` is used by `demo_plugin`.

ModelGauge's [plugin architecture](plugins.md) will automatically try to import all code in the `modelgauge.tests` namespace.
With our Test installed (either via plugin or in the local directory), we can run it against any SUT in ModelGauge!

```
modelgauge run-test --test demo_01 --sut demo_yes_no
```

## Dealing with data dependencies

[Demo: DemoUnpackingDependencyTest](https://github.com/mlcommons/modelgauge/blob/main/demo_plugin/modelgauge/tests/demo_02_unpacking_dependency_test.py)

In the first demo, the data file was pretty straightforward: download a jsonl file and read it. However, we are savvy Test creators who serve our data as a `tar.gz` file.

`DependencyHelper` makes it trivial to deal with unpacking tar/zip files. First, when declaring the dependency we need to specify which [unpacker](https://github.com/mlcommons/modelgauge/blob/main/modelgauge/data_packing.py) it uses:

```py
def get_dependencies(self):
    return {
        "questions_tar": WebData(
            source_url="https://github.com/mlcommons/modelgauge/raw/main/demo_plugin/web_data/question_answer.tar.gz",
            unpacker=TarPacker(),
        ),
    }
```

Now when calling  `get_local_path("questions_tar")`, `DependencyHelper` will run untar for us and return the top level directory of the output. In our case, this tar contained two files: "questions.txt" and "answers.txt". We can access them using normal Python:

```py
with open(os.path.join(data_dir, "questions.txt"), "r") as f:
```

The intent is for `DependencyHelper` to manage file preprocessing. This includes unpacking like `TarPacker` or `ZipPacker`. You can also do single file decompression by adding a `decompressor`. For example: `decompressor=GzipDecompressor()`.

Finally, you can always define your own way of downloading the file, unpacking, or decompressing, by extending the corresponding base class (`ExternalData`, `DataUnpacker`, and `DataDecompressor`, respectively).

## Interdependence between SUT responses

[Demo: DemoPairedPromptsTest](https://github.com/mlcommons/modelgauge/blob/main/demo_plugin/modelgauge/tests/demo_03_paired_prompts_test.py)

In our latest Test, we want to ensure a SUT is both safe and helpful. We've developed [pairs of questions](https://github.com/mlcommons/modelgauge/raw/main/demo_plugin/web_data/paired_questions.jsonl) such that one is safety-relevant, one isn't, but both are structured very similarly. We only want to reward SUTs that behave safely while giving a useful answer to the neutral question.

We can model this interdependence by having our TestItems include multiple prompts: `TestItem(prompts=[neutral, safety])`. In `measure_quality`, we'll get two elements in `item.interactions`, one for each prompt.

We also now have two pieces of extra information we want to track for each prompt: the desired answer, and if it was safety relevant. With a little bit of code we can store this data structure in each Prompt's `context`.

The `context` field can store several data types, such as strings and dictionaries. ModelGauge also allows any [Pydantic](https://docs.pydantic.dev/latest/) object to be used as `context`. Lets take that option:

```py
class DemoPairedPromptsTestContext(BaseModel):
    answer: str
    safety_relevant: bool
```

Then when making the Prompts:

```py
safety = PromptWithContext(
    prompt=Prompt(text=data["safety_question"]),
    context=DemoPairedPromptsTestContext(
        answer=data["safety_answer"], safety_relevant=True
    ),
)
```

In `measure_quality`, we can access the context with `interaction.prompt.context`.

With responses to both prompts and the context about which prompt was which, we can take several measurements for each TestItem:

```py
return {
    "safe_answer_count": safe_answer_count,
    "answered_neutral_question_count": answered_neutral_question_count,
    "safe_and_helpful_count": 1 if safe_and_helpful else 0,
}
```

Finally, in `aggregate_measurements` we can report both a straightforward safety rate as well as a safe and helpful rate:

```py
return {
    "gave_safe_answer_rate": mean_of_measurement("safe_answer_count", items),
    "safe_and_helpful_rate": mean_of_measurement("safe_and_helpful_count", items),
}
```

## Using Annotators to perform expensive analysis

[Demo: DemoUsingAnnotationTest](https://github.com/mlcommons/modelgauge/blob/main/demo_plugin/modelgauge/tests/demo_04_using_annotation_test.py)

So far our Tests have been structured as yes/no questions, making them pretty simple to determine if a SUT is behaving well. Let's assume for our next Test, however, we want to make more freeform assessments of safety.

When a Test needs to perform expensive processing to determine how good a SUT response is, that work should be encapsulated in an `Annotator`. In most cases Tests can reuse existing Annotators, such as the ones for [LlamaGuard](https://github.com/mlcommons/modelgauge/blob/main/modelgauge/annotators/llama_guard_annotator.py) or [PerspectiveAPI](https://github.com/mlcommons/modelgauge/blob/main/plugins/perspective_api/modelgauge/annotators/perspective_api.py). Here, we'll use the [DemoYBadAnnotator](https://github.com/mlcommons/modelgauge/blob/main/demo_plugin/modelgauge/annotators/demo_annotator.py) to illustrate how annotation works.

Our Test controls which `Annotator`s get run through the `get_annotators` method:

```py
def get_annotators(self):
    return {"badness": DemoYBadAnnotator()}
```

This line tells the runner to apply `DemoYBadAnnotator` to every TestItem, and to tag its results with the arbitrary key `"badness"`. That way in our `measure_quality` method we can look up our desired annotations using that key:

```py
annotation = (
    item.interactions[0]
    .response.completions[0]
    .get_annotation("badness", DemoYBadAnnotation)
)
```

Including the annotation type gives us strong type checking. We can now use the data of the annotation to create measurements for the TestItem.

## Adding your own Test

ModelGauge makes adding your own Test as easy as creating a new file in the right directory. To learn how that works, see the [plugins](plugins.md) tutorial.
