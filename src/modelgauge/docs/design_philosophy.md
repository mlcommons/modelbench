# Design Philosophy

This document tries to capture the high level principles we use in designing ModelGauge. Hopefully this can help answer questions like "why do it this way" or "how should I trade off these priorities".

## Be a library, not a framework

ModelGauge's primary objective is to facilitate the interaction of many Tests with many SUTs. There are two high level approaches you could take for doing so:

* Framework: ModelGauge owns the top of the process, with Tests/SUTs fitting into predefined boxes.
* Library: ModelGauge owns the bottom of the process, providing building blocks that can be assembled by others.

While the lines between these strategies can be blurry, some signs of a good library are that:

* Someone can use just the single class/method they want without having to set up support structures they don't.
* Users opt in to functionality, instead of opting out.
  * Example: Choosing to call a method in their code vs defining a no-op method that must exist.
* Most functions are [pure functions](https://en.wikipedia.org/wiki/Pure_function).

Libraries are fundamentally easier to reuse for multiple purposes, but with that comes extra design work. We think that for ModelGauge, that tradeoff is worth it.

### Examples in practice

* The interface for SUTs is designed to allow them to be reused by applications that ignore the Tests, Annotators, and Runners defined by ModelGauge.
* We refactored secrets management to remove global state. This allowed for more pure functions, and made the setup needed to use a class/method more explicit.
* All the logic for a Test is encapsulated into a single interface, such that to use that class you do not need to set up any other part of the infrastructure to be useful.


## Separate the required from the optional

We want ModelGauge to scale to hundreds of Tests and SUTs. We want those Tests and SUTs to be written by the community. That will invariably bring in lots of transitive dependencies, and potential trust issues, that aren't really needed by most/all users. Therefore we want to put a boundary between the core code that all users must have to use ModelGauge, and all the extras people might want.

We have approached this problem via the [plugin architecture](plugins.md). Anything not needed by (almost) all users of ModelGauge should be moved to plugins. This allows:

* The common code to be relatively small, and requiring infrequent updates.
* Only the transitive dependencies you actually want.
* Individual users to decide what code they want to trust.

### Examples in practice

* The Transformers library is enormous, but you only have to install it if you include the optional plugin (e.g. `huggingface`).

## Be extensible

The AI community is inventing and discarding use cases faster than we could hope to support in-house. Researchers will always want bespoke features we didn't forsee. So where possible, we should let users extend what we've built to suit their purposes. For example:

* A user should be able to add their Test/SUT without editing any code they don't own.
* Where possible, we should leave the door open to people adding new categories of Test/SUT.

### Examples in practice

* Users can define a new SUT in a notebook and run Tests against it using normal imports.
* A Test creator can add new ways of downloading prompts that can still track versioning without editing core libraries.
* Test authors can add arbitrary context to TestItem and define their own structured Result and have them correctly serialized in the TestRecord.
* A project can define its own cloud-based runner, reusing the existing Tests/SUTs.


## Make it hard to do the wrong thing

We expect there to be great diversity of programming skill across our community members. While good documentation and examples can help you do the right thing, it takes a little more effort to keep people from doing the wrong thing. In general:

* Try to make runtime errors into coding time errors.
* If there is only one right way to do something, don't ask users to do it.
* If a method doesn't need a value, don't give it that value.
* Try to make all data objects immutable.

### Examples in practice

* A Test never knows what SUT it is testing, and a SUT never knows what Test is being run, as that could only lead to undesirable behavior.
* Utilize `mypy` checking.
* We define interfaces and use `@abstractmethod` to enforce implementing all the features.
* Pull as much common work as possible out of Test definition and into the runner.
* There are a lot of classes to represent different stages of a TestItem instead of having a single object that gets built up over time. This makes it clear what must be set at each stage, and hides information that isn't needed at others.

## As always, the zen of Python

While all [19 aphorisms](https://peps.python.org/pep-0020/) are great, a few that we take especially to heart are:

* Explicit is better than implicit.
* There should be one-- and preferably only one --obvious way to do it.
* If the implementation is hard to explain, it's a bad idea.

### Examples in practice

* To represent structured data, use Pydantic objects instead of dicts as this makes it clear what the structures and types are.
* We discarded several "hard to explain" ways of managing secrets and ensuring compatibility between a Test and a SUT, even though they arguably met other design philosophy goals better.
