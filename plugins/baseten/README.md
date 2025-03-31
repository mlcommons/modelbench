# Baseten plugin

Plugin for running against models hosted in [Baseten](https://www.baseten.co).

## Configuring endpoints

Currently, the systems available via baseten are controlled by and environment variable
`BASETEN_MODELS` that enumerates the models being hosted at baseten. This environment variable's
value is a list of comma separate name value pairs of SUT name and the baseten model id.

One way to locate this model identifier is to locate the deployment in your workspace,
click on the deployment's card in the workspace, and you'll see the model identifier 
as a suffix to the deployed model's name:

![Locating the model id](locating-model-id.png)

Baseten will host your model endpoint at a URL like:

```
https://model-{model_id}.api.baseten.co/production/predict
```

You then register a SUT in `baseten_api.py`:

```python
 SUTS.register(
    BasetenPromptSUT,
    "baseten-gemma2-9b",
    "google/gemma2-9b",
    "https://model-2qjgeo2q.api.baseten.co/environments/production/predict",
    BASETEN_SECRET,
)
```

Then you can run the benchmark against your baseten endpoint:

```bash
poetry run modelbench --plugin-dir plugins/baseten benchmark -s baseten-gemma2-9b
```

You can choose the SUT type based on the type of interface used by the endpoint:

* `BasetenPromptSUT` - a basic prompt interface (e.g., like Gemma 2) 
* `BasetenMessagesSUT` - a "chat messages" interface (e.g., like llama 3.1).