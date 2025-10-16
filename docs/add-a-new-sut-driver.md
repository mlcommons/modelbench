# How to Create a New SUT Driver

A SUT driver is code that interfaces with a SUT provider (like Together.ai or Huggingface).
Most providers need their own driver. We provide several drivers that can be used in dynamic SUT UIDs so you don't have to write any code.

## Check This Before You Write Any Code

### Does an Existing Driver Exist?

If your SUT provider is listed as a key in the `DYNAMIC_SUT_FACTORIES` in
[sut_factory](../src/modelgauge/sut_factory.py), you don't need to write any code.

```python
DYNAMIC_SUT_FACTORIES: dict = {
    "hf": HuggingFaceSUTFactory,
    "hfrelay": HuggingFaceSUTFactory,
    "openai": OpenAICompatibleSUTFactory,
    "together": TogetherSUTFactory,
    "modelship": ModelShipSUTFactory,
}
```
Please refer to [suts-how-to.md](./suts-how-to.md#existing) for details.

### Is Your SUT Already Pre-Defined?

If your SUT provider is not listed there, before you write any code, check whether we
have already written code as a pre-defined SUT by running the following command on the CLI:

```bash
poetry run modelgauge list-suts
```

If your SUT is listed, you don't need to write any code. You can invoke it using its UID found in the command above, e.g.

```
poetry run modelgauge run-sut --sut olmo-7b-0724-instruct-hf --prompt "Why did the chicken cross the road?"
```

### Is Your SUT Compatible with the OpenAI API?

If your SUT supports the OpenAI API, you need to write very little code. Please refer to [suts-how-to.md](./suts-how-to.md#openai) for details.


## How to Write a New Driver

If you've determined a new driver is needed, here's how to do it.

Use the [HuggingFaceSUT](../src/modelgauge/suts/huggingface_api.py) class as a template.

1. Create a SUT class and a few other classes like this (imports ommitted for brevity):

```python
class MySUTRequest(BaseModel):
    inputs: str

class MySUTResponse(BaseModel):
    generated_text: str # this will depend on what your provider's API returns

@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class MySUT(PromptResponseSUT):

    def __init__(self, uid: str, api_url: str):
        # configure the provider here (API key, base URL, etc)
        super().__init__(uid)

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> MySUTRequest:
        # turn a modelgauge prompt into a provider-compatible prompt
        return MySUTRequest(inputs=prompt.text)

    def translate_response(self, request: MySUTRequest, response: MySUTResponse) -> SUTResponse:
        # turn a provider-native prompt into a modelgauge-compatible response
        return SUTResponse(text=response.generated_text)

    def evaluate(self, request: MySUTRequest) -> MySUTResponse:
        # queries your provider's API and returns the API results in a Response object
        payload = request.model_dump(exclude_none=True)
        response = requests.post(...)
        response_json = response.json()[0]
        return MySUTResponse(**response_json)
```

2. Create a factory class that creates an instance of your SUT from its UID. Look at [TogetherSUTFactory](../src/modelgauge/suts/together_sut_factory.py) for inspiration.

The `DRIVER_NAME` constant must be unique to your driver. It will be a key in a dict.

```python
DRIVER_NAME = "my_sut"

class MySUTApiKey(RequiredSecret):
    # adjust this to your specific provider
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="my_host", # name a scope in secrets.toml like this string
            key="api_key"
        )

class MySUTFactory(DynamicSUTFactory):
    def __init__(self, raw_secrets: RawSecrets):
        # RawSecrets is a dict of secrets
        super().__init__(raw_secrets)

    def get_secrets(self) -> list[InjectSecret]:
        api_key = InjectSecret(MySUTApiKey)
        return [api_key]

    def make_sut(self, sut_definition: SUTDefinition) -> MySUT:
        sut_metadata = sut_definition.to_dynamic_sut_metadata()
        return MySUT(
            sut_definition.dynamic_uid,
            *self.injected_secrets(),
        )
```

3. Add an entry for your new factory class in the `DYNAMIC_SUT_FACTORIES` dict in [sut_factory](../src/modelgauge/sut_factory.py).

```python
DYNAMIC_SUT_FACTORIES: dict = {
    ...
    "my_sut": MySUTFactory,
    ...
}
```

4. Add a scope to [secrets.toml](../config/secrets.toml) for your provider, using the `scope` you defined in the `Secret` class(es) for your SUT:

```toml
[my_host]
api_key=<your key>
```

Your SUT UID will be `maker/model:my_sut`. Once you have a driver, if you have multiple
models hosted on the same provider with the same interface, you can reference all of them the same way, e.g.

* `some/model:my_sut`
* `another/model:my_sut`
* `yet/another-model:my_sut`
* `model:my_sut`
