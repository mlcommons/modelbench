# Pre-Defined SUTs

Some SUTs are pre-defined with a specific provider, model, authentication keys, and sometimes more options. You may add new SUTs this way or use
[dynamic SUTs](./suts-how-to.md#dynamic).

## An Existing Pre-Defined SUT May Already Be Available

Pre-defined SUTs have been created for a large number of providers, including VertexAI, Azure, Mistral,
Anthropic, Huggingface, OpenAI, and Together. Look through [modelgauge/suts](../src/modelgauge/suts/)
and you may find one you can reuse.

TIP: search for the string `SUTS.register` to find the pre-defined SUTs.

## Adding Your Pre-Defined SUT

1. Add your SUT to an existing pre-defined SUT...

If there's an existing pre-defined SUT matching your needs, simply add yours to the `SUTS.register` section in its module (see below).

1. ... or create a new pre-defined SUT

If an existing pre-defined SUT doesn't exist, create one in [modelgauge/suts](../src/modelgauge/suts/) modeled after [modelgauge/suts/huggingface_api.py](../src/modelgauge/suts/huggingface_api.py). Refer to your
provider's documentation and comments in the base classes in this repo for implementation details.

2. Register Your New SUT

Your new pre-defined SUT must then be registered at the bottom of its module like this:

```python
SUTS.register(SomeSUTClass, "my-arbitrary-uid-string", "[maker/]model", InjectSecret(SomeKey))
```

`[maker/]model` is your model's standard name at the provider of your choice, e.g. "allenai/OLMo-2-0325-32B-Instruct". Some models don't use the `maker` part. Most models on Huggingface do.

3. Add your UID string as a key in the `LEGACY_SUT_MODULE_MAP` dict in [sut_factory](../src/modelgauge/sut_factory.py) pointing to the name of the module your SUT class is in:

```python
LEGACY_SUT_MODULE_MAP = {
    # ...
    "my-arbitrary-uid-string": "the_module_SometSUTClass_is_in"
    # ...
```

Then you can use your SUT UID like this:

```bash
poetry run modelgauge run-sut --sut my-arbitrary-uid-string --prompt "Why did the chicken cross the road?"
```

## Authentication

Most providers require authentication. Secrets like API tokens are stored in [secrets.toml](../config/secrets.toml) in a "scope" defined in the `Secret` subclass defined for your SUT. Those `Secret` classes look like this:

```python
class SomeSecret(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="my_provider",
            key="api_key"
        )
```

The value of `scope` is a string used to define a block in [secrets.toml](../config/secrets.toml), and the value of `key` is the identifier for that secret in that scope. E.g. if your `Secret` class includes these
values in the description:

```python
return SecretDescription(
    scope="my_special_provider",
    key="extra_secret_key"
```

then your [secrets.toml](../config/secrets.toml) file must include:

```toml
[my_special_provider]
extra_secret_key=<your key>
```

If one doesn't already exist, create one and refer to it in the `SUTS.register` statement. Refer to [modelgauge/suts/huggingface_api.py](../src/modelgauge/suts/huggingface_api.py) for an example.