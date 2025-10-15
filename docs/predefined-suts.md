# Pre-Defined SUTs

Some SUTs are pre-defined with a specific provider, model, authentication keys, and
any other options. Adding new SUTs this way is **not encouraged**. You should use
[dynamic SUTs](./add-a-sut.md#dynamic). This document is for reference when
dealing with older SUTs.

You can define a SUT and register it with an arbitrary string as its SUT UID in the global SUTS registry and add any options needed to make it run, including the client (driver) class, any secrets, the model name as used internally by the provider, etc.

Look at [together_client](../src/modelgauge/suts/together_client.py) for an example.

1. Create a SUT class and associated classes for secrets, requests, and responses.
2. At the bottom of your SUT class definition module, register your SUT with some arbitrary UID and including any additional parameters for the SUT to run, e.g. authentication keys:

```python
SUTS.register(SomeSUTClass, "my-arbitrary-uid-string", "maker/model", InjectSecret(SomeKey))
```

3. Add your UID string as a key in the `LEGACY_SUT_MODULE_MAP` dict in [sut_factory](../src/modelgauge/sut_factory.py) pointing to the name of the module your SUT class is in:

```python
LEGACY_SUT_MODULE_MAP = {
    # ...
    "my-arbitrary-uid-string": "the_module_SometSUTClass_is_in"
    # ...
```

Then you can use your arbitrary SUT UID like this:

```bash
poetry run modelgauge run-sut --sut my-arbitrary-uid-string --prompt "Why did the chicken cross the road?"
```
