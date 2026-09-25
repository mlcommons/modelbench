# Using and Adding SUTs (System Under Test)

A SUT consists of a model and the provider it runs on. All SUTs are identified by a unique, persistent
UID. Commands in modelbench and modelgauge require users to specify the UID for the SUT they are testing.
There are two kinds of SUT UIDs:

* pre-defined SUTs with UIDs stored in the code
* SUTs specified on the fly by using a UID that follows a particular format

This document describes the various ways of specifying SUTs.

## Pre-Defined SUTs

Several SUT UIDs are ready to use. They are the keys in the `LEGACY_SUT_MODULE_MAP` dict
in [sut_factory](../src/modelgauge/sut_factory.py). You can list them with `uv run modelgauge list-suts` on the CLI. You may be able to add a new SUT easily if it's similar to an existing one.

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
