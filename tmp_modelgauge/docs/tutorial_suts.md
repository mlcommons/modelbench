# SUT Demos

## Making your own SUT

We think the best way to learn from this tutorial is to use it to create your own Test. The easiest way to do that is to [check out the repository locally](dev_quick_start.md) and add your own file(s) to the `modelgauge/suts/` directory. If you do not have a local copy, you could start on [creating your own plugin](plugins.md) first, or just ignore any of the parts in the tutorial about "installed".

## Creating a basic SUT

[Demo: DemoYesNoSUT](https://github.com/mlcommons/modelgauge/blob/main/demo_plugin/modelgauge/suts/demo_01_yes_no_sut.py)

Before you dive into creating more realistic SUTs, let's start with with a toy example: A SUT that only answers the question "Does this prompt have an even number of words?"

To call this SUT in ModelGauge, we need to implement a class for it. Let's call our SUT `DemoYesNoSUT`.
Since this SUT is able handle [Prompt Response Tests](prompt_response_tests.md) we should have it inherit from `PromptResponseSUT`. This interface requires the SUT to implement 3 high level steps:

1. Convert the prompt provided by the Test into the native representation used by the SUT.
1. Evaluate the SUT and return a detailed account of its response in the SUT's native representation.
1. Convert the SUT's native response into the SUT-agnostic form so it can be measured by the Test.

The goal of providing the SUT native representation is to allow you as the person creating the SUT to fully capture the capabilities and limitations of the SUT. This makes it easier to extend the SUT to new purposes later. It also aids in debugging as ModelGauge will report these native representations on errors.

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

Note that in this example we inherit from [Pydantic](https://docs.pydantic.dev/latest/)'s `BaseModel` to aid in serialization and caching. This isn't strictly necessary, but we highly recommend doing so.

To tell ModelGauge what your native representations are, we use Python [Generics](https://mypy.readthedocs.io/en/stable/generics.html).

```py
class DemoYesNoSUT(PromptResponseSUT[DemoYesNoRequest, DemoYesNoResponse]):
```

Finally we want to tell ModelGauge that this class is a SUT, and describe its capabilities:

```py
@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class DemoYesNoSUT(PromptResponseSUT[DemoYesNoRequest, DemoYesNoResponse]):
```

We'll explore capabilities more in a later tutorial. For now we are saying that this SUT can process two kinds of prompts: Text and Chat.

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
SUTS.register(DemoYesNoSUT, "demo_yes_no")
```

> [!NOTE]
> If you are writing your own file, give the SUT a different key, as `demo_yes_no` is used by `demo_plugin`.

ModelGauge's [plugin architecture](plugins.md) will automatically try to import all code in the `modelgauge.suts` namespace.
With our SUT installed (either via plugin or in the local directory), we can run it manually with `run-sut`:

```
modelgauge run-sut --sut demo_yes_no --prompt "One two three four"
```

We can also evaluate it using any Test in ModelGauge!

```
modelgauge run-test --test demo_01 --sut demo_yes_no
```

## SUTs that call an API

[Demo: DemoRandomWords](https://github.com/mlcommons/modelgauge/blob/main/demo_plugin/modelgauge/suts/demo_02_secrets_and_options_sut.py)

We expect the most common way to define a SUT is as a wrapper around an existing API. To explore this kind of SUT implementation, lets assume we've recently created a `RandomWords` SUT and set up an API for users to call it.  To implement this SUT in ModelGauge we'll need to explore two new features: Secrets and SUT Options.

### Secrets

To make sure only authorized users access our SUT, the `RandomWords` API requires the user to provide their secret API key. ModelGauge strives to handle secrets in a way that balances several competing desires:

* Secrets should stay secret to a given user.
* It should be very clear what secrets are needed for a given run.

To get started, we first need to define a wrapper for our new kind of secret:

```py
class DemoApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="demo", key="api_key", instructions="The password is 12345"
        )
```

This code is giving a name to our new secret (`demo.api_key`) and providing instructions for what a user should do if they don't already have a secret value. As we inherited from `RequiredSecret` we are also saying that code using `DemoApiKey` will fail without the secret value. If the secret was just nice to have, we could have used `OptionalSecret` instead.

Secrets should be passed into a SUT's `__init__` function:

```py
def __init__(self, uid: str api_key: DemoApiKey):
    super().__init__(uid)
    self.api_key = api_key.value
```

The `.value` property on `RequiredSecret` returns the `str` secret itself, so we can pass that to our API: `RandomWordsClient(api_key=self.api_key)`.

Finally, when we register an instance of the SUT, we need to specify which secret we need:

```py
SUTS.register(DemoRandomWords, "demo_random_words", InjectSecret(DemoApiKey))
```

When running from the command line, ModelGauge will (by default) read the `config/secrets.toml` file and look for the value matching the `DemoApiKey`.

### SUTOptions

Requests to the `RandomWords` API take the following arguments:

* source_text: All of the text from the prompt.
* num_words_desired: How many words to return per completion.
* num_completions: How many completions to create for this request.

The API returns a list of strings, one for each requested completion. We can create a native representation of this API with:

```py
class DemoRandomWordsRequest(BaseModel):
    source_text: str
    num_words_desired: int
    num_completions: int

class DemoRandomWordsResponse(BaseModel):
    completions: Sequence[str]
```

We've previously seen how Tests pass the Prompt text into the SUT. However, Tests can also provide configuration for many other parameters a SUT might want via `SUTOptions`. For example the values `prompt.options.max_tokens` and `prompt.options.num_completions` correspond well to the arguments in the `RandomWords` API. When defining your native representation, we request that you pass through all values in `SUTOptions` to your SUT that your SUT can use:

```py
def translate_text_prompt(self, prompt: TextPrompt) -> DemoRandomWordsRequest:
    return DemoRandomWordsRequest(
        source_text=prompt.text,
        num_words_desired=prompt.options.max_tokens,
        num_completions=prompt.options.num_completions,
    )
```

## Reusing SUT classes

[Demo: DemoConstantSUT](https://github.com/mlcommons/modelgauge/blob/main/demo_plugin/modelgauge/suts/demo_03_sut_with_args.py)

Many APIs allow you to interact with different models as easily as switching a request parameter. For example, TogetherAI and OpenAI both take the `model`'s name in the request. To handle this situation, ModelGauge allows a single SUT class to be reused with different configuration. To illustrate, let's create a SUT that always returns a predefined response.

```py
@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class DemoConstantSUT(PromptResponseSUT[DemoConstantRequest, DemoConstantResponse]):
    def __init__(self, uid: str, response_text: str):
        super().__init__(uid)
        self.response_text = response_text
```

We can then register multiple version of this SUT:

```py
# Everything after the class name gets passed to the class.
SUTS.register(DemoConstantSUT, "demo_always_angry", "I hate you!")
# You can use kwargs if you want.
SUTS.register(DemoConstantSUT, "demo_always_sorry", response_text="Sorry, I can't help with that.")
```

Reminder: when using ModelGauge as a library you can always skip the registration step and construct SUTs directly. Registration is only needed to make something accessible via command line.

## Capabilities

ModelGauge uses the concept of SUT capabilities to ensure SUT-Test compability. Tests that require some capability cannot be applied to SUTs that don't report that capability. SUTs must report their capabilities in the SUT decorator:

```py
@modelgauge_sut(capabilities=[...])
```

Possible capabilities include:
- `AcceptsTextPrompt`: SUT can take a `TextPrompt` as input. Must implement `translate_text_prompt()`.
- `AcceptsChatPrompt`: SUT can take a `ChatPrompt` as input. Must implement `translate_chat_prompt()`.
- `ProducesPerTokenLogProbabilities`: SUT is able return `top_logprobs`.

## Adding your own SUT

ModelGauge makes adding your own SUT as easy as creating a new file in the right directory. To learn how that works, see the [plugins](plugins.md) tutorial.
