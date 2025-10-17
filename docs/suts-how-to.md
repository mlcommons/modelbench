# Using and Adding SUTs (System Under Test)

A SUT consists of a model and the provider it runs on. All SUTs are identified by a unique, persistent
UID. Commands in modelbench and modelgauge require users to specify the UID for the SUT they are testing.
There are two kinds of SUT UIDs:

* pre-defined SUTs with UIDs stored in the code
* SUTs specified on the fly by using a UID that follows a particular format

This document describes the various ways of specifying SUTs.

## Pre-Defined SUTs

Several SUT UIDs are ready to use. They are the keys in the `LEGACY_SUT_MODULE_MAP` dict
in [sut_factory](../src/modelgauge/sut_factory.py). You can list them with `poetry run modelgauge list-suts` on the CLI. You may be able to add a new SUT easily if it's similar to an existing one.

More details in [this document](./predefined-suts.md).

## <a name="dynamic"></a>Dynamic SUTs

A SUT can also be specified on the fly if you use a SUT UID as follows:

`[maker/]model:[provider:]driver`

* `maker` is the model vendor (e.g. "meta-llama") and `model` is the model name (e.g. "Meta-Llama-3-8B-Instruct"), matching the Huggingface nomenclature (e.g. "meta-llama/Meta-Llama-3-8B-Instruct"). Some model names omit the `maker` part. Double-check your model name!
* `driver` refers to the client code interfacing with the provider. We provide drivers for many major providers.
* `provider` is the provider running the model, if relayed through a proxy like Huggingface's relay or a SUT compatible with the OpenAI API. It is omitted if it can be inferred from the driver name.

### <a name="existing"></a>Dynamic SUTs with an Existing Driver

A lot of new SUTs will require no code if your model is hosted on one of the providers we offer a driver for, such as Huggingface, Huggingface's inference provider partners, OpenAI, and together.ai.

Factory classes are used to create SUT objects for you, including their driver and model name, based the elements in the SUT UID.

Available drivers are identified in `DYNAMIC_SUT_FACTORIES` in
[sut_factory](../src/modelgauge/sut_factory.py). The keys correspond to the `driver` string in the SUT UID.

We may add more drivers from time to time.

```python
DYNAMIC_SUT_FACTORIES: dict = {
    "hf": HuggingFaceSUTFactory,
    "hfrelay": HuggingFaceSUTFactory,
    "openai": OpenAICompatibleSUTFactory,
    "together": TogetherSUTFactory,
    "modelship": ModelShipSUTFactory,
}
```

* "hf" is used for models hosted by Huggingface
* "hfrelay" is used for models hosted by one of Huggingface's inference provider partners (e.g. nebius, sambanova) and proxied by Huggingface ([more info](https://huggingface.co/docs/inference-providers/en/index))
* "openai" is a model hosted by OpenAI
* "together" is a model hosted by together.ai
* "modelship" is internal to MLCommons

#### Usage

For models using one of those drivers, all you need is to add your credentials to [config/secrets.toml](../config/secrets.toml) in a section named after the driver name string, e.g. for together.ai:

```toml
[together]
api_key=<your API key>
```

Note: the Huggingface key in the TOML file should be under "hugging_face" rather than "hf". This may change.

#### Dynamic SUT UID Examples

OLMo-2-0325-32B-Instruct on Huggingface:

`allenai/OLMo-2-0325-32B-Instruct:hf`

DeepSeek-R1 on together.ai:

`deepseek-ai/DeepSeek-R1:together`

Llama-4-Maverick-17B-128E-Instruct on sambanova via Huggingface:

`meta-llama/Llama-4-Maverick-17B-128E-Instruct:sambanova:hfrelay`

### <a name="openai"></a>OpenAI-Compatible Dynamic SUTs

If your SUT has an OpenAI-compatible API, you can add it with minimal code. VLLM and models hosted by OpenAI
(like the chatgpt family) support the OpenAI API. Other providers offer that option too. This is a good option
if you self-host a model using VLLM.

The UID for an OpenAI-compatible SUT works the same way as above, with "openai" as the `driver` section and a string of your choice as the `provider` section of the UID, e.g.:

`my/big_model:my_host:openai`

Because these SUTs need a base URL for the API, you do need to write a little code as follows:

1. Create a subclass of `OpenAIGenericSUTFactory` in [openai_sut_factory.py](../src/modelgauge/suts/openai_sut_factory.py):
   * `base_url` is the base URL of your API server.
   * `provider` is a string of your choice. It must be a valid TOML section identifier. We strongly recommend lowercase ASCII letters.
2. Add your new class to the `OPENAI_SUT_FACTORIES` dict in [openai_sut_factory.py](../src/modelgauge/suts/openai_sut_factory.py). The dict key must be the same as the value set for `provider`.

```python
class MySUTFactory(OpenAIGenericSUTFactory):
    def __init__(self, raw_secrets, **kwargs):
        super().__init__(raw_secrets)
        self.provider = "my_host"
        self.base_url = "https://example.net/v1/"

OPENAI_SUT_FACTORIES: dict = {"my_host": MySUTFactory}
```

3. Add a scope containing the `api_key` secret to your API in  [config/secrets.toml](../config/secrets.toml). The scope must be named the same as the `provider` in your SUT factory class.

```toml
[my_host]
api_key=<your API key>
```

Your SUT UID will look like `my/big_model:my_host:openai`, and you can use it with modelgauge and modelbench like this:

```bash
poetry run modelgauge run-sut --sut my/big_model:my_host:openai --prompt "Why did the chicken cross the road?"

poetry run modelbench benchmark general --sut my/big_model:my_host:openai --prompt-set practice --evaluator default -m 10
```

### Dynamic SUTs With New Drivers

If your SUT provider requires custom client code that isn't available in this repo, you will need to write some driver code. Details are in [add-a-new-sut-driver.md](./add-a-new-sut-driver.md).

## Authentication

Major providers require authentication. Keys are stored in [config/secrets.toml](../config/secrets.toml). One block ("scope") per provider. E.g. for models running on OpenAI (e.g. the chatgpt family), add a section like this:

```toml
[openai]
api_key=abcd1234
```

If your own SUT requires authentication, add the credentials it to [config/secrets.toml](../config/secrets.toml). E.g.
if the `provider` in your SUT class is the string "my_host" and the auth parameter is named "api_key":

```toml
[my_host]
api_key=abcd1234
```

Note that models hosted on Huggingface use "hugging_face" as the scope in secrets.toml rather than "hf". This may change.

If your SUT requires more than one secret, add all their values to the same scope, e.g.:

```toml
[my_host]
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
            scope="my_host"
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
