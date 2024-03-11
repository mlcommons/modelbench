# SUT Demos

## Creating a basic SUT

[Demo: DemoYesNoSUT](../demo_plugin/newhelm/suts/demo_01_yes_no_sut.py)

Before you dive into creating more realistic SUTs, let's start with with a toy example: A SUT that only answers the question "Does this prompt have an even number of words?"

To call this SUT in NewHELM, we need to implement a class for it. Let's call our SUT `DemoYesNoSUT`.
Since this SUT is able handle [Prompt Response Tests](prompt_response_tests.md) we should have it inherit from `PromptResponseSUT`. This interface requires the SUT to implement 3 high level steps:

1. Convert the prompt provided by the Test into the native representation used by the SUT.
1. Evaluate the SUT and return a detailed account of its response in the SUT's native representation.
1. Convert the SUT's native response into the SUT-agnostic form so it can be measured by the Test.

The goal of providing the SUT native representation is to allow you as the person creating the SUT to fully capture the capabilities and limitations of the SUT. This makes it easier to extend the SUT to new purposes later. It also aids in debugging as NewHELM will report these native representations on errors.

For our "Yes" "No" SUT, its native request is just going to be the text of the prompt:

```py
class DemoYesNoRequest(BaseModel):
    text: str
```

In the response, we'll need the answer. Since we calculated it, we can also include the number of words in the prompt:

```py
class DemoYesNoResponse(BaseModel):
    number_of_words: int
    text: str
```

Note that in this example inhert from [Pydantic](https://docs.pydantic.dev/latest/)'s `BaseModel` to aid in serialization and caching. This isn't strictly necessary, but we highly recommend doing so.

To tell NewHELM what your native representations are, we use Python [Generics](https://mypy.readthedocs.io/en/stable/generics.html).

```py
class DemoYesNoSUT(PromptResponseSUT[DemoYesNoRequest, DemoYesNoResponse]):
```

With that setup out of the way, we can move into the three steps mentioned earlier. First we need to translate the different kinds of Prompts into our native representation:

```py
def translate_text_prompt(self, prompt: TextPrompt) -> DemoYesNoRequest:
    return DemoYesNoRequest(text=prompt.text)

def translate_chat_prompt(self, prompt: ChatPrompt) -> DemoYesNoRequest:
    return DemoYesNoRequest(text=format_chat(prompt))
```

A benefit of translation is the ability to customize the handling of different Prompt types for your SUT. In this example `translate_chat_prompt` can decide how it wants to flatten the chat history into a single `str`.

Now that we have a `DemoYesNoRequest`, we can implement the behavior of the SUT:

```py
def evaluate(self, request: DemoYesNoRequest) -> DemoYesNoResponse:
    # Return Yes if the input is an even number of words
    number_of_words = len(request.text.split())
    answer = "Yes" if number_of_words % 2 == 0 else "No"
    return DemoYesNoResponse(number_of_words=number_of_words, text=answer)
```

Finally we convert the response back into a normalized form that Tests can understand:

```py
def translate_response(
    self, request: DemoYesNoRequest, response: DemoYesNoResponse
) -> SUTResponse:
    return SUTResponse(completions=[SUTCompletion(text=response.text)])
```

Some notes on `translate_response`:

* Our SUT isn't able to create multiple completions for the same Prompt, so we just always return 1 `SUTCompletion`.
* Even though our SUT returns `number_of_words`, that isn't currently in use by Tests, so it gets dropped.

Finally, to make our new SUT discoverable, we can add it to the registry, giving it a unique key:

```py
SUTS.register("demo_yes_no", DemoYesNoSUT)
```

With our SUT [installed](plugins.md), we can run it manually with `run-sut`:

```
poetry run python newhelm/main.py run-sut --sut demo_yes_no --prompt "One two three four"
```

We can also evaluate it using any Test in NewHELM!

```
poetry run python newhelm/main.py run-test --test demo_01 --sut demo_yes_no
```
