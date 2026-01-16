# SUT plugins

`modelgauge` uses [namespace plugins](../../docs/plugins.md) to separate the core libraries from the implementation of less central code. That way you only have to install the dependencies you actually care about.

Any file put in this directory, or in any installed package with a namespace of `modelgauge.suts`, will be automatically loaded by the `modelgauge` command line tool via `load_plugins()`.

## Dynamic SUTs

A SUT can be created dynamically if there's a factory class for it. An invocation typically looks like:

```bash
uv run modelgauge run-sut --sut vendor/model:provider:driver --prompt "why did the chicken cross the road?"
```

where `driver` is a string identifying a SUT factory class in [src/modelgauge/sut_factory](../sut_factory.py), and `vendor/model` is a typical model identifier as used by Huggingface and others.

We currently support the following drivers for dynamic SUTs:

* huggingface
* huggingface relay (proxy to inference providers through huggingface)
* openai
* together
* openai-compatible clients

### OpenAI-Compatible Dynamic SUTs

If your SUT supports the OpenAI API, you can call it one of two ways.

#### No Code

```bash
uv run modelgauge run-sut --sut "maker/model:mysut:openai;url=https://example.com/v1/" --prompt "why did the chicken cross the road?"
```

Where `mysut` is a scope in [config/secrets.toml](../../../config/secrets.toml) including `api_key`:

```toml
[mysut]
api_key="some key"
```

#### A Little Code

1. Create a factory class for it in [openai_sut_factory.py](./openai_sut_factory.py). All you need to set is `provider` and `base_url`:

```python
class MySUTFactory(OpenAIGenericSUTFactory):
    def __init__(self, raw_secrets, **kwargs):
        super().__init__(raw_secrets)
        self.provider = "mysut" # used to find the api key in config/secrets.toml
        self.base_url = "https://example.net/v1/" # where your SUT is accessible
```

2. Add a mapping between the `provider` string and your factory to the [openai_sut_factory.OPENAI_SUT_FACTORIES](./openai_sut_factory.py) dict:

```python
OPENAI_SUT_FACTORIES: dict = {
    "demo": DemoOpenAICompatibleSUTFactory,
    "mysut": MySUTFactory
}
```

3. Add credentials to [config/secrets.toml](../../../config/secrets.toml) in a scope named after the string you set `provider` to:

```toml
[mysut]
api_key="some key"
```

Your SUT UID will look like `vendor/model:mysut:openai`:

`uv run modelgauge run-sut --sut vendor/model:mysut:openai --prompt "why did the chicken cross the road?"`

### Other Dynamic SUTs

To add support for dynamic SUTs using other client code:

1. Create a factory class for it, e.g. `src/modelgauge/suts/my_sut_factory.py`. Use the existing [together_sut_factory.py](./together_sut_factory.py) module as an example.
2. Add a mapping between a string of your choice and your new factory to the [sut_factory.DYNAMIC_SUT_FACTORIES](../sut_factory.py) dict:

```python
from modelgauge.suts.my_sut_factory import MySUTFactory

DYNAMIC_SUT_FACTORIES: dict = {
    "hf": HuggingFaceSUTFactory,
    #...
    "mysut": MySUTFactory,
}
```

Make sure your factory knows how to find any keys or other secrets it needs.

Your SUT UID will look like `vendor/model:mysut`:

`uv run modelgauge run-sut --sut vendor/model:mysut --prompt "why did the chicken cross the road?"`
