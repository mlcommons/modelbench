# Using and Adding SUTs (System Under Test)

A SUT consists of a model and the provider it runs on. All SUTs are identified by a unique, persistent
UID. Commands in modelbench and modelgauge require users to specify the UID for the SUT they are testing.
There are two kinds of SUT UIDs:

* pre-defined SUTs with UIDs stored in the code
* SUTs specified on the fly by using a UID that follows a particular format

This document describes the various ways of specifying SUTs.

## Pre-Defined SUTs

Several SUT UIDs are ready to use. They are the keys in the `LEGACY_SUT_MODULE_MAP` dict
in [sut_factory](../src/modelgauge/sut_factory.py). You can list them with `poetry run modelgauge list-suts` on the CLI.

## <a name="dynamic"></a>Dynamic SUTs

A SUT can be specified on the fly if you use a SUT UID as follows:

`[maker/]model:[provider:]driver`

* `maker` is the model vendor (e.g. "meta-llama"; may be blank) and `model` is the model name (e.g. "Meta-Llama-3-8B-Instruct"). This matches the Huggingface nomenclature (e.g. "meta-llama/Meta-Llama-3-8B-Instruct").
* `driver` refers to the client code interfacing with the provider. We provide drivers for many major providers.
* `provider` is the provider running the model, if relayed through a proxy like Huggingface's relay or a SUT compatible with the OpenAI API. It is blank otherwise.

### <a name="existing"></a>Dynamic SUTs with an Existing Driver

These SUTs require no code if your model is hosted on one of the providers we offer a driver for.

The driver name strings in the SUT UID are the keys in the `DYNAMIC_SUT_FACTORIES` in
[sut_factory](../src/modelgauge/sut_factory.py). We may add more drivers from time to time.

```python
DYNAMIC_SUT_FACTORIES: dict = {
    "hf": HuggingFaceSUTFactory,
    "hfrelay": HuggingFaceSUTFactory,
    "huggingface": HuggingFaceSUTFactory,
    "openai": OpenAICompatibleSUTFactory,
    "together": TogetherSUTFactory,
    "modelship": ModelShipSUTFactory,
}
```

* "hf" or "huggingface" is used for models hosted by Huggingface
* "hfrelay" is used for models hosted by one of Huggingface's inference provider partners (e.g. nebius, sambanova) and proxied by Huggingface ([more info](https://huggingface.co/docs/inference-providers/en/index))
* "openai" is a model hosted by OpenAI
* "together" is a model hosted by together.ai
* "modelship" is internal to MLCommons

For models on these providers, all you need is to add your credentials to [secrets.toml](../config/secrets.toml) in a
section named after the driver name string, e.g. for OpenAI:

```toml
[openai]
api_key=<your API key>
```

Note: the Huggingface key in the TOML file should be "hugging_face" rather than "huggingface". This may change.

#### Dynamic SUT UID Examples

OLMo-2-0325-32B-Instruct on Huggingface:

`allenai/OLMo-2-0325-32B-Instruct:huggingface`

DeepSeek-R1 on together.ai:

`deepseek-ai/DeepSeek-R1:together`

Llama-4-Maverick-17B-128E-Instruct on sambanova via Huggingface:

`meta-llama/Llama-4-Maverick-17B-128E-Instruct:sambanova:hfrelay`

### <a name="openai"></a>OpenAI-Compatible Dynamic SUTs

If your SUT has an OpenAI-compatible API, you can add a SUT with minimal code. VLLM and models hosted by OpenAI
(like the chatgpt family) support the OpenAI API. Other providers offer that option. This is a good option
if you self-host a model using VLLM.

The UID for an OpenAI-compatible SUT works the same way as above, with "openai" as the driver string:

`maker/model:provider:openai`

Because these SUTs need a base URL for the API, you do need to write a little code as follows:

1. Create a subclass of `OpenAIGenericSUTFactory` in [openai_sut_factory.py](../src/modelgauge/suts/openai_sut_factory.py):
   * `base_url` is the base URL of your API.
   * `provider` is a string of your choice. It must be a valid TOML section identifier. We strongly recommend lowercase ASCII letters.
2. Add your new class to the `OPENAI_SUT_FACTORIES` dict in [openai_sut_factory.py](../src/modelgauge/suts/openai_sut_factory.py). The dict key must be the same as the value set for `provider`.

```python
class MySUTFactory(OpenAIGenericSUTFactory):
    def __init__(self, raw_secrets, **kwargs):
        super().__init__(raw_secrets)
        self.provider = "mysut"
        self.base_url = "https://example.net/v1/"

OPENAI_SUT_FACTORIES: dict = {"mysut": MySUTFactory}
```

3. Add a scope containing the `api_key` secret to your API in  [secrets.toml](../config/secrets.toml). The scope must be named the same as the `provider` in your SUT factory class.

```toml
[mysut]
api_key=<your API key>
```

Your SUT UID will be `maker/model:mysut:openai`, and you can use it with modelgauge and modelbench like this:

```bash
poetry run modelgauge run-sut --sut maker/model:mysut:openai --prompt "Why did the chicken cross the road?"

poetry run modelbench benchmark general --sut maker/model:mysut:openai --prompt-set practice --evaluator default -m 10
```

### Dynamic SUTs Without Existing Drivers

If your SUT provider uses client code that isn't compatible with the above drivers or the OpenAI API, you will need to write some code. Details are in [add-a-new-sut-driver.md](./add-a-new-sut-driver.md).

### Reference: Pre-Defined SUTs

This method should not be used. Use the dynamic SUT mechanism above instead.

If you're curious, refer to [this document](./predefined-suts.md).

## Authentication

Major providers require authentication. Keys are stored in [secrets.toml](../config/secrets.toml). One block ("scope") per provider. E.g. for models running on OpenAI (e.g. the chatgpt family), add a section like this:

```toml
[openai]
api_key=abcd1234
```

If your own SUT requires authentication, add the credentials it to [secrets.toml](../config/secrets.toml). E.g.
if the `driver` in your SUT class is the string "mysut" and the auth parameter is named "api_key":

```toml
[mysut]
api_key=abcd1234
```

If your SUT requires more than one secret, add all their values to the same scope, e.g.:

```toml
[mysut]
organization=mycorp
api_key=abcd1234
username=somebody
```

You will need one `Secret` class for each credential. Those classes roughly look like this:

```python
class MySUTAPIKey(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="mysut"
            key="api_key"
        )
```

## Troubleshooting

### API Keys, Tokens, etc

The secret's identifier may not be `api_key`. Another common identifier is `token`. AWS uses a key ID and secret access key. Refer to that provider's documentation for details.

### Access to Huggingface Models

If you get this error message even if you have an API key:

`modelgauge.dynamic_sut_factory.ModelNotSupportedError: Huggingface doesn't know model <model name>, or you need credentials for its repo.`

you may need to request access to the model from the provider before you can use it.
