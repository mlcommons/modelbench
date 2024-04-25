# Developer Quick Start

> [!NOTE]
> This guide assumes you want to contribute code changes to ModelGauge. If you only want to use it as a library to run
> evaluations, please read the [User Quick Start](user_quick_start.md) instead.

## Prerequisites

- **Python 3.10**: It is recommended to use Python version 3.10 with ModelGauge.
- **Poetry**: ModelGauge uses [Poetry](https://python-poetry.org/) for dependency
  management. [Install](https://python-poetry.org/docs/#installation) it if it's not already on your machine.

> [!WARNING]
> Poetry and other python virtual environment
> tooling [may not play nicely together](https://github.com/orgs/python-poetry/discussions/7767). As such we recommend you
> let Poetry manage the venv, and not try to run it within a venv.

## Installation

1. Download the repository:

        git clone https://github.com/mlcommons/modelgauge.git
        cd modelgauge

2. Install the default dependencies:

        poetry install

   This will instruct poetry to install the default dependencies into this project's environment. An isolated
   environment will be created, unless another virtual environment is already activated.
   After you install, future `poetry run` commands will use that environment.

## Getting Started

You can run our command line tool with:

```shell
poetry run modelgauge
```

That should provide you with a list of all commands available. A useful command to run is `list`, which will show you
all known Tests, System Under Tests (SUTs), and installed plugins.

```shell
poetry run modelgauge list
```

ModelGauge uses a [plugin architecture](plugins.md), so by default the list should be pretty empty. To see this in
action, we can instruct poetry to install the `demo` plugin:

```shell
poetry install --extras demo
poetry run modelgauge list
```

You should now see a list of all the modules in the `demo_plugin/` directory. For more info on the demo
see [here](tutorial.md).

The `plugins/` directory contains many useful plugins. However, those have a lot of transitive dependencies, so they can
take a while to install. To install them all:

```shell
poetry install --extras all_plugins
poetry run modelgauge list
```

Finally note that any extras not listed in a `poetry install` call will be uninstalled.

## Running a Test

Here is an example of running a Test, using the `demo` plugin:

```shell
poetry run modelgauge run-test --sut demo_yes_no --test demo_01
```

If you want additional information about existing tests, you can run:

```shell
poetry run modelgauge list-tests
```

To obtain detailed information about the existing Systems Under Test (SUTs) in your setup, you can execute the following
command:

```shell
poetry run modelgauge list-suts
```

## Using `poetry run`

When ModelGauge is installed using Poetry, in order to run the `modelgauge` command line tool, the command must be
prefixed by `poetry run` e.g. `poetry run modelgauge list`. You can also start your session with `poetry shell`, which
makes `poetry run` unnecessary thereafter. For simplicity, the rest of the documentation may omit the `poetry run`
prefix for `modelgauge` commands.

## Further Questions

If you have any further questions, please feel free to ask them in
the [#engineering-support](https://discord.com/channels/1137054779013615616/1209638758400528455) discord / file a github
issue. Also if you see a way to make our documentation better, please submit a pull request. We'd love your help!
