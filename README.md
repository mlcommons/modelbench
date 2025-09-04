
# ModelBench

Run safety benchmarks against AI models and view detailed reports showing how well they performed.

## Badges

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fmlcommons%2Fmodelbench%2Fmain%2Fpyproject.toml)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mlcommons/modelbench/python-app.yml)

## Background

This is a [MLCommons project](https://mlcommons.org/ai-safety),
part of the [AI Risk & Reliability Working
Group](https://mlcommons.org/ai-risk-and-reliability/).
The project is at an early stage. You can see sample benchmarks
[here](https://ailuminate.mlcommons.org/benchmarks/) and our 0.5 white paper
[here](https://arxiv.org/abs/2404.12241).

This project now contains both ModelGauge and ModelBench. ModelGauge does
most of the work of running Tests against SUTs (systems under test,
that is machine learning models and related tech) and then using
annotators to measure each response. ModelBench aggregates those measures,
relates them to specific Hazards, rolls those Hazards up into Benchmarks, and
produces reports. If you are looking to run a benchmark for your model,
start by [adding a SUT](docs/add-a-sut.md) that works with ModelGauge.

## Requirements

The current public practice benchmark uses LlamaGuard to
evaluate the safety of responses. For now you will need a
[Together AI](https://www.together.ai/) account to use it. For 1.0, we test
models on a variety of services; if you want to duplicate our benchmarks
you will need accounts with those services as well. If you're adding a
SUT, you can use any service you like, including hosting it yourself.

Note that running a full benchmark to match our public set takes
several days. Depending on response time, running your own SUT may be
faster. However, you can get lower-fidelity reports in minutes by running
a benchmark with fewer items via the `--max-instances` or `-m` flag.

## Installation

Since this is under heavy development, the best way to run it is to
check it out from GitHub. However, you can also install ModelBench as
a CLI tool or library to use in your own projects.

### Install ModelBench with [Poetry](https://python-poetry.org/)

1. Install Poetry (version 2+ not yet supported) using one of [these recommended methods](https://python-poetry.org/docs/#installation).  All of these methods require a recent version of python3 to be installed and available on your shell path.  For example:
```shell
pipx install poetry
```

2. Clone this repository.
```shell
git clone https://github.com/mlcommons/modelbench.git
```

3. Make sure that you have no python virtual environments activated. They will interfere with the poetry setup.

4. Install ModelBench and dependencies.
```shell
cd modelbench
poetry install
```

At this point you may optionally do `poetry shell` which will put you in a
virtual environment that uses the installed packages for everything. If
you do that, you don't have to explicitly say `poetry run` in the
commands below.

## Running Tests

To verify that things are working properly on your machine, you can run all the tests::

```shell
poetry run pytest tests
```

## Trying It Out

We encourage interested parties to try it out and give us feedback. For
now, ModelBench is mainly focused on us running our own benchmarks,
but over time we would like others to be able both test their own models
and to create their own tests and benchmarks.

### Running Your First Benchmark

Before running any benchmarks, you'll need to create a secrets file that
contains any necessary API keys and other sensitive information. Create a
file at `config/secrets.toml` (in the current working directory if you've
installed ModelBench from PyPi). You can use the following as a template.

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

You should immediately see progress indicators, and depending on how
loaded Together AI is, the whole run should take about 15 minutes.

> [!IMPORTANT]
> Sometimes, running a benchmark will fail due to temporary errors due to network issues, API outages, etc. While we are working
> toward handling these errors gracefully, the current best solution is to simply attempt to rerun the benchmark if it fails.

### Viewing the Scores

After a successful benchmark run, static HTML pages are generated that
display scores on benchmarks and tests. These can be viewed by opening
`web/index.html` in a web browser. E.g., `firefox web/index.html`.

Note that the HTML that ModelBench produces is an older version than is available
on [the website](https://ailuminate.mlcommons.org/). Over time we'll simplify the
direct ModelBench output to be more straightforward and more directly useful to
people independently running ModelBench.

### Using the Journal

As `modelbench` runs, it logs each important event to the journal. That includes
every step of prompt processing. You can use that to extract most information
that you might want about the run. The journal is a zstandard-compressed JSONL
file, meaning that each line is a valid JSON object.

There are many tools that can work with those files. In the example below, we
use [jq](https://jqlang.github.io/jq/, a JSON swiss army knife. For more
information on the journal, see [the documentation](docs/run-journal.md).

To dump the raw scores, you could do something like this

```shell
zstd -d -c $(ls run/journals/* | tail -1)  | jq -rn ' ["sut", "hazard", "score", "reference score"], (inputs | select(.message=="hazard scored") | [.sut, .hazard, .score, .reference]) | @csv'
```

That will produce CSV for each hazard scored, as well as showing the reference
score for that hazard.

Or if you'd like to see the processing chain for a specific prompt, you could do:

```shell
zstd -d -c $(ls run/journals/* | tail -1)  | jq -r 'select(.prompt_id=="airr_practice_1_0_41321")'
```

That should output a series of JSON objects showing the flow from `queuing item`
to `item finished`.

**CAUTION**: Please note that many of the prompts may be uncomfortable or
harmful to view, especially to people with a history of trauma related to
one of the hazards that we test for. Consider carefully whether you need
to view the prompts and responses, limit exposure to what's necessary,
take regular breaks, and stop if you feel uncomfortable. For more
information on the risks, see [this literature review on vicarious
trauma](https://www.zevohealth.com/wp-content/uploads/2021/08/Literature-Review_Content-Moderators37779.pdf).

### Managing the Cache

To speed up runs, ModelBench caches calls to both SUTs and
annotators. That's normally what a benchmark-runner wants. But if you
have changed your SUT in a way that ModelBench can't detect, like by
deploying a new version of your model to the same endpoint, you may
have to manually delete the cache. Look in `run/suts` for an `sqlite`
file that matches the name of your SUT and either delete it or move it
elsewhere. The cache will be created anew on the next run.

### Running the Benchmark on your SUT

ModelBench uses the ModelGauge library to discover
and manage SUTs. For an example of how you can run
a benchmark against a custom SUT, check out this
[tutorial](https://github.com/mlcommons/modelbench/blob/main/docs/add-a-sut.md).

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
