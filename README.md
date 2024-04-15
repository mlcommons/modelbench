
# ModelBench

Run safety benchmarks against AI models and view detailed reports showing how well they performed.

## Badges

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fmlcommons%2Fmodelbench%2Fmain%2Fpyproject.toml%3Ftoken%3DGHSAT0AAAAAACQ32W7DFHRN3LANOASRWR7MZQ5TRMA)

## Installation

> [!NOTE]
> In the future, ModelBench will be installable via PyPI.

### Install ModelBench with [Poetry](https://python-poetry.org/) for local development.

1. Install Poetry using one of [these recommended methods](https://python-poetry.org/docs/#installation).
```shell
pipx install poetry
```

2. Clone this repository.
```shell
git clone git@github.com:mlcommons/modelbench.git
```

3. Install ModelBench and dependencies.
```shell
cd modelbench
poetry install
```

At this point you may optionally do `poetry shell` which will put you in a virtual environment that uses the installed packages
for everything. If you do that, you don't have to explictly say `poetry run` in the commands below.

## Running Tests

To run all tests, cd into the `modelbench` directory and run `pytest`.

```shell
poetry run pytest
```

## Running Your First Benchmark

Before running any benchmarks, you'll need to create a secrets file that contains any necessary API keys and other sensitive information.
Create a file at `config/secrets.toml`. You can use the following as a template.

```toml
[together]
api_key = "<your key here>"

[perspective_api]
api_key = "<your key here>"
```

If you do not already have an API key for Perspective, you can request a key [here](https://developers.perspectiveapi.com/s/docs-get-started?language=en_US).
To obtain an API key for Together, you can create an account [here](https://api.together.xyz/)

With these keys in place, you are now ready to run your first benchmark!

```shell
poetry run modelbench benchmark -m 10
```
> [!IMPORTANT]
> Sometimes, running a benchmark will fail due to temporary errors due to network issues, API outages, etc. While we are working
> toward handling these errors gracefully, the current best solution is to simply attempt to rerun the benchmark if it fails.

## Viewing The Scores

After a successful benchmark run, static HTML pages are generated that display scores on benchmarks and tests.
These can be viewed by opening `web/index.html` in a web browser. E.g., `firefox web/index.html`. 

## Contributing

Coffee uses the following tools for development, code quality, and packaging:
1. [Poetry](https://python-poetry.org/) - dependency management and packaging
2. [Black](https://github.com/psf/black) - code formatting and style
3. [MyPy](https://github.com/python/mypy) - static typing

To contribute:
1. Fork the repository
2. Create your feature branch
3. Ensure there are tests for your changes and that they pass
4. Create a pull request