# Tutorial: Adding A New SUT (System Under Test)

## Dynamic SUTs With Existing Drivers

If your SUT is hosted on one of the major providers we provide a driver for, this requires no code.

The drivers are keys in the `DYNAMIC_SUT_FACTORIES` dict in [sut_factory](../src/modelgauge/sut_factory.py).

* "hf" or "huggingface" is a SUT hosted by Huggingface
* "hfrelay" is a SUT hosted by one of Huggingface's inference provider partners (e.g. nebius, sambanova) via Huggingface
* "openai" is a SUT hosted by OpenAI
* "together" is a SUT hosted by together.ai
* "modelship" is internal to MLCommons

```python
DYNAMIC_SUT_FACTORIES: dict = {
    "hf": **HuggingFaceSUTFactory**,
    "hfrelay": HuggingFaceSUTFactory,
    "huggingface": HuggingFaceSUTFactory,
    "openai": OpenAICompatibleSUTFactory,
    "together": TogetherSUTFactory,
    "modelship": ModelShipSUTFactory,
}
```

All you need is to add your credentials to [secrets.toml](../config/config/secrets.toml) and create a SUT UID like `maker/model:driver` or `maker/model:provider:driver`  where:

* `maker` is the model vendor (e.g. "meta-llama")
* `model` is the model name (e.g. "Meta-Llama-3-8B-Instruct"). This matches the Huggingface nomenclature.
* `driver` is one of the keys in `DYNAMIC_SUT_FACTORIES`
* `provider` is the provider running your SUT, if relayed through a proxy like Huggingface's relay.

#### SUT UID Examples

OLMo-2-0325-32B-Instruct on Huggingface:

`allenai/OLMo-2-0325-32B-Instruct:hf`

SmolLM3-3B on Huggingface:

`HuggingFaceTB/SmolLM3-3B:huggingface`

DeepSeek-R1 on together:

`deepseek-ai/DeepSeek-R:together`

Llama-4-Maverick-17B-128E-Instruct on sambanova via Huggingface:

`meta-llama/Llama-4-Maverick-17B-128E-Instruct:sambanova:hfrelay`

## OpenAI-Compatible SUTs

If your SUT has an OpenAI-compatible API, you can add a SUT with minimal code. VLLM and SUTs hosted by OpenAI (like the chatgpt family), as do other services, support the OpenAI API.

All you have to do is extend our existing code like this:

1. Create a subclass of `OpenAIGenericSUTFactory` in [openai_sut_factory.py](../src/modelgauge/suts/openai_sut_factory.py):
   * `base_url` is the base URL of your API.
   * `provider` is a string of your choice. It must be a valid TOML section identifier. We strongly recommend lowercase ASCII letters.
2. Add your class to the `OPENAI_SUT_FACTORIES` dict in [openai_sut_factory.py](../src/modelgauge/suts/openai_sut_factory.py). The dict key must be the same as the value set for `provider`.

```python
class MySUTFactory(OpenAIGenericSUTFactory):
    def __init__(self, raw_secrets, **kwargs):
        super().__init__(raw_secrets)
        self.provider = "mysut"
        self.base_url = "https://example.net/v1/"

OPENAI_SUT_FACTORIES: dict = {"mysut": MySUTFactory}
```

3. Set the `api_key` secret to your API key in a scope named the same as `provider` in [secrets.toml](../config/config/secrets.toml)

```toml
[mysut]
api_key=abcd1234
```

Your SUT UID (needed for modelbench and modelgauge) will be `maker/model:mysut:openai` where:

* `maker` is the model vendor (e.g. "meta-llama")
* `model` is the model name (e.g. "Meta-Llama-3-8B-Instruct"). This matches the Huggingface nomenclature.
* "mysut" is the string you chose for `provider` in your SUT class.
* "openai" indicates your SUT uses our existing OpenAI API client.

You can now use your SUT like this:

Modelgauge:

```bash
poetry run modelgauge run-sut -s maker/model:mysut:openai --prompt "Why did the chicken cross the road?"
```

Modelbench:

```bash
poetry run modelbench benchmark general --sut maker/model:mysut:openai --prompt-set practice --evaluator default -m 10
```

## Dynamic SUTs With New Drivers

If your SUT provider uses client code that isn't compatible with the above drivers or the OpenAI API, you will need to write some code. Use the [HuggingFaceSUT](../src/modelgauge/suts/huggingface_api.py) class as a template.

You will need:

1. a SUT class like this:

```python
class MySUTRequest(BaseModel):
    inputs: str

class MySUTResponse(BaseModel):
    generated_text: str # this will depend on what your provider's API returns

@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class MySUT(PromptResponseSUT):

    def __init__(self, uid: str, api_url: str):
        # configure the SUT here (API key, base URL, etc)
        super().__init__(uid)

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> MySUTRequest:
        # turns a text prompt into a pydantic object
        return MySUTRequest(inputs=prompt.text)

    def evaluate(self, request: MySUTRequest) -> MySUTResponse:
        # queries your provider's API and returns the API results in a Response object
        payload = request.model_dump(exclude_none=True)
        response = requests.post(...)
        response_json = response.json()[0]
        return MySUTResponse(**response_json)

    def translate_response(self, request: MySUTRequest, response: MySUTResponse) -> SUTResponse:
        # turns the native response from your provider's API into a standard response
        return SUTResponse(text=response.generated_text)
```

2. a factory class that creates an instance of your SUT from its UID. Look at [TogetherSUTFactory](../src/modelgauge/suts/together_sut_factory.py) for inspiration.

```python

DRIVER_NAME = "mysut"

class MySUTApiKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="mysut",
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

3. an entry for your new factory class in the `DYNAMIC_SUT_FACTORIES` dict in [sut_factory](../src/modelgauge/sut_factory.py).

```python
DYNAMIC_SUT_FACTORIES: dict = {
    ...
    "mysut": MySUTFactory,
    ...
}
```

Your SUT UID will look like `maker/model:mysut` as above.

## Pre-Registered SUTs (Deprecated)

Use the dynamic SUT mechanism above instead.

This lets you register an arbitrary string for a SUT UID in the global SUTS registry and add any options needed to make it run, including the client (driver) class, any secrets,
the model name as used internally by the provider, etc.

Look at [together_client](../src/modelgauge/suts/together_client.py) for an example.

1. At the bottom of your SUT class definition module, register your SUT:

```python
SUTS.register(SomeSUTClass, "my-arbitrary-uid-string", "maker/modelname", InjectSecret(SomeKey))
```

2. Add your string to the `LEGACY_SUT_MODULE_MAP` dict in [sut_factory](../src/modelgauge/sut_factory.py):

```python
LEGACY_SUT_MODULE_MAP = {
    ...
    "my-arbitrary-uid-string": "the_module_SometSUTClass_is_in"
    ...
```

Then you can use your arbitrary SUT UID as usual, e.g.

```bash
poetry run modelgauge run-sut -s my-arbitrary-uid-string --prompt "Why did the chicken cross the road?"
```

## Authentication

Major providers require authentication. Keys are stored in [secrets.toml](../config/config/secrets.toml).
One block per provider. E.g. for SUTs running on openai (e.g. the chatgpt family), add a section like this:

```toml
[openai]
api_key=abcd1234
```

If your own SUT requires authentication (e.g. an API key), add it to [secrets.toml](../config/config/secrets.toml) like so:

```toml
[mysut]
api_key=abcd1234
```

The string `mysut` is referred to as the "scope" and the string `api_key` is the identifier
for the secret used in the authentication process (e.g. headers, POST payload, etc. depending on
the provider hosting the SUT).

If your SUT requires more than one secret, add them all to the same scope, e.g.

```toml
[mysut]
organization=mycorp
api_key=abcd1234
username=somebody
```

### Notes

#### Huggingface

The scope for Huggingface credentials in [secrets.toml](../config/config/secrets.toml) is "hugging_face" rather than "huggingface." This may change in the future.

#### API Keys, Tokens, etc

The secret's identifier may not be `api_key`. Another common identifier is `token`. AWS uses a key ID and secret access key. Refer to that provider's documentation for details.

### Troubleshooting

If you get this error message even if you have an API key:

`modelgauge.dynamic_sut_factory.ModelNotSupportedError: Huggingface doesn't know model <model name>, or you need credentials for its repo.`

you may need to request access to the model from the provider before you can use it.
