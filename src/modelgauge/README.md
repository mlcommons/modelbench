# ModelGauge

Goal: Make it easy to automatically and uniformly measure the behavior of many AI Systems.

> [!WARNING]
> This repo is still in **beta** with a planned full release in Fall 2024. Until then we reserve the right to make backward incompatible changes as needed.

ModelGauge is an evolution of [crfm-helm](https://github.com/stanford-crfm/helm/), intended to meet their existing use cases as well as those needed by the [MLCommons AI Safety](https://mlcommons.org/working-groups/ai-safety/ai-safety/) project.

## Summary

ModelGauge is a library that provides a set of interfaces for Tests and Systems Under Test (SUTs) such that:

* Each Test can be applied to all SUTs with the required underlying capabilities (e.g. does it take text input?)
* Adding new Tests or SUTs can be done without modifications to the core libraries or support from ModelGauge authors.

Currently ModelGauge is targeted at LLMs and [single turn prompt response Tests](docs/prompt_response_tests.md), with Tests scored by automated Annotators (e.g. LlamaGuard). However, we expect to extend the library to cover more Test, SUT, and Annotation types as we move toward full release.


## Docs

* [Developer Quick Start](docs/dev_quick_start.md)
* [Tutorial for how to create a Test](docs/tutorial_tests.md)
* [Tutorial for how to create a System Under Test (SUT)](docs/tutorial_suts.md)
* How we use [plugins](docs/plugins.md) to connect it all together.
