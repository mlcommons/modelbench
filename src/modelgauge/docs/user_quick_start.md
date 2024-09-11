# User Quick Start

> [!NOTE]
> This guide assumes that you only want to use ModelGauge as a library to run evaluations, and that you do not want to contribute code to ModelGauge. If you do want to contribute code, please read the [Developer Quick Start](dev_quick_start.md) instead.

## Prerequisites

- **Python 3.10**: It is recommended to use Python version 3.10 with ModelGauge.

## Installation

Run the following (ideally inside a Python virtual environment):

```shell
pip install modelgauge
```

## Getting Started

You can run our command line tool with:

```shell
modelgauge
```

That should provide you with a list of all commands available. A useful command to run is `list`, which will show you all known Tests, System Under Tests (SUTs), and installed plugins.

```shell
modelgauge list
```

ModelGauge uses a [plugin architecture](plugins.md), so by default the list should be pretty empty. To see this in action, we can instruct poetry to install the `demo` plugin:

```shell
pip install 'modelgauge[demo]'
```

You should now see a list of all the modules in the `demo_plugin/` directory. For more info on the demo see [here](tutorial.md). 

Many SUTs and tests are provided by ModelGauge plugins. Here is a list of officially supported plugins, as well as the commands to install them:

```shell
# Hugging Face SUTs
pip install 'modelgauge[huggingface]'

# OpenAI SUTs
pip install 'modelgauge[openai]'

# Together SUTs
pip install 'modelgauge[together]'

# Perspective API
pip install 'modelgauge[perspective-api]'

# Tests used by the AI Safety Benchmark
pip install 'modelgauge[standard-tests]'
```

You can also install all plugins with the following command. Some plugins have a lot of transitive dependencies, so installation can take a while:

```shell
pip install 'modelgauge[all]'
```

## Running a Test

Here is an example of running a Test, using the `demo` plugin:

```shell
modelgauge run-test --sut demo_yes_no --test demo_01
```

If you want additional information about existing tests, you can run:

```shell
modelgauge list-tests
```

To obtain detailed information about the existing Systems Under Test (SUTs) in your setup, you can execute the following command:
```shell
modelgauge list-suts
```

## Further Questions

If you have any further questions, please feel free to ask them in the #engineering discord / file a github issue. Also if you see a way to make our documentation better, please submit a pull request. We'd love your help!
