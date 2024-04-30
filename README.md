
# ModelBench

Run safety benchmarks against AI models and view detailed reports showing how well they performed.

## Badges

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fmlcommons%2Fmodelbench%2Fmain%2Fpyproject.toml)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mlcommons/modelbench/python-app.yml)

## Background

This is a [MLCommons project](https://mlcommons.org/ai-safety), part of the [AI Safety Working Group](https://mlcommons.org/working-groups/ai-safety/ai-safety/).
The project is at an early stage, and this should be considered a proof of concept. Results are not intended to
indicate actual levels of AI system safety. You can see sample benchmarks [here](https://mlcommons.org/benchmarks/ai-safety/) and read the white paper [here](https://arxiv.org/abs/2404.12241).

ModelBench is closely related to [ModelGauge](https://github.com/mlcommons/modelgauge). ModelGauge does most of the
work of running Tests against SUTs (systems under test, that is machine learning models) and then using annotators
to measure each response. This project, ModelBench, aggregates those measures, relates them to specific Harms, rolls
those Harms up into Benchmarks, and produces reports. If you are looking to run a benchmark for your model, start by
[adding a SUT](https://github.com/mlcommons/modelgauge/blob/main/docs/tutorial_suts.md) that works with ModelGauge. 

## Requirements

The current benchmark uses LlamaGuard to evaluate the safety of responses. For now you will need a [TogetherAI](https://www.together.ai/)
account to use it. For 0.5, all included models are also run via TogetherAI. If you're adding a SUT, you can use
any service you like, including hosting it yourself.

Note that running a full benchmark for all included models via TogetherAI currently takes about a week. Depending
on response time, running your own SUT may be faster. We aim to speed things up substantially for 1.0. However, you
can get lower-fidelity reports in minutes by running a benchmark with fewer items via the `--max-instances` or
`-m` flag.

## Installation

Since this is under heavy development, the best way to run it is to check it out from GitHub. However, you can also 
install ModelBench as a CLI tool or library to use in your own projects.

### Install ModelBench with [Poetry](https://python-poetry.org/) for local development.

1. Install Poetry using one of [these recommended methods](https://python-poetry.org/docs/#installation). For example:
```shell
pipx install poetry
```

2. Clone this repository.
```shell
git clone https://github.com/mlcommons/modelbench.git
```

3. Install ModelBench and dependencies.
```shell
cd modelbench
poetry install
```

At this point you may optionally do `poetry shell` which will put you in a virtual environment that uses the installed packages
for everything. If you do that, you don't have to explicitly say `poetry run` in the commands below.

### Install ModelBench from PyPI

1. Install ModelBench into your local environment or project the way you normally would. For example:
```shell
pip install modelbench
```

## Running Tests

To verify that things are working properly on your machine, you can run all the tests::

```shell
poetry run pytest tests
```

## Trying It Out

We encourage interested parties to try it out and give us feedback. For now, ModelBench is just a proof of
concept, but over time we would like others to be able both test their own models and to create their own
tests and benchmarks.

### Running Your First Benchmark

Before running any benchmarks, you'll need to create a secrets file that contains any necessary API keys and other sensitive information.
Create a file at `config/secrets.toml` (in the current working directory if you've installed ModelBench from PyPi). 
You can use the following as a template.

```toml
[together]
api_key = "<your key here>"
```

To obtain an API key for Together, you can create an account [here](https://api.together.xyz/).

With your keys in place, you are now ready to run your first benchmark!
Note: Omit `poetry run` in all example commands going forward if you've installed ModelBench from PyPi.

```shell
poetry run modelbench benchmark -m 10
```

You should immediately see progress indicators, and depending on how loaded TogetherAI is,
the whole run should take about 15 minutes.

> [!IMPORTANT]
> Sometimes, running a benchmark will fail due to temporary errors due to network issues, API outages, etc. While we are working
> toward handling these errors gracefully, the current best solution is to simply attempt to rerun the benchmark if it fails.

### Viewing the Scores

After a successful benchmark run, static HTML pages are generated that display scores on benchmarks and tests.
These can be viewed by opening `web/index.html` in a web browser. E.g., `firefox web/index.html`. 

If you would like to dump the raw scores, you can do:

```shell
poetry run modelbench grid -m 10 > scoring-grid.csv
```

To see all raw requests, responses, and annotations, do:

```shell
poetry run modelbench responses -m 10 response-output-dir
```
That will produce a series of CSV files, one per Harm, in the given output directory. Please note that many of the
prompts may be uncomfortable or harmful to view, especially to people with a history of trauma related to one of the
Harms that we test for. Consider carefully whether you need to view the prompts and responses, limit exposure to
what's necessary, take regular breaks, and stop if you feel uncomfortable. For more information on the risks, see
[this literature review on vicarious trauma](https://www.zevohealth.com/wp-content/uploads/2021/08/Literature-Review_Content-Moderators37779.pdf).

### Managing the Cache

To speed up runs, ModelBench caches calls to both SUTs and annotators. That's normally what a benchmark-runner wants.
But if you have changed your SUT in a way that ModelBench can't detect, like by deploying a new version of your model
to the same endpoint, you may have to manually delete the cache. Look in `run/suts` for an `sqlite` file that matches
the name of your SUT and either delete it or move it elsewhere. The cache will be created anew on the next run.

### Running the benchmark on your SUT

ModelBench uses the ModelGauge library to discover and manage SUTs. To run the benchmark on your own SUT, follow the  [instructions to add a new SUT to ModelGauge](https://modelgauge.readthedocs.io/en/latest/tutorial_suts/). You can then run the benchmark on your SUT by setting the `--sut` flag to the name of the SUT. For instance, to run the benchmark on the `demo_yes_no` SUT from the tutorial, run:

```shell
poetry run modelbench benchmark -m 10 --sut demo_yes_no
```

## Contributing

ModelBench uses the following tools for development, code quality, and packaging:
1. [Poetry](https://python-poetry.org/) - dependency management and packaging
2. [Black](https://github.com/psf/black) - code formatting and style
3. [MyPy](https://github.com/python/mypy) - static typing

To contribute:
1. Fork the repository
2. Create your feature branch
3. Ensure there are tests for your changes and that they pass
4. Create a pull request
