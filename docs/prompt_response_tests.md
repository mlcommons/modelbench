# Prompt Response Test

The most common kind of Test in NewHELM is the Prompt Response Test. These Tests involve a series of **single turn** interactions with the SUT. Some characteristics of Prompt Response Tests:

* Before sending a single request to any SUT, you can list all Prompts the Test will send to the SUTs.
* Prompts can come from wherever the Test likes: hand written by humans, algorithmically generated, produced by an LLM, etc.

At a high level, your Test is responsible for defining how to:

* Generate `Prompts` you want to send to SUTs.
* What, if any, `Annotations` you need to pull in about the SUT `Response`s (e.g. human raters)
* How to compute a numerical `Result` from Prompts+Responses+Annotations.

## TestItem - The smallest unit of measurable work

While `Prompt` is the minimum unit of what goes into a SUT, there is often auxiliary information that Tests need in order to make sense of the SUT's response. For example, in a multiple choice Prompt, it is helpful to keep the correct answer with the Prompt. Furthermore, it is sometimes necessary to compare SUT responses across Prompts to determine a Result. Consider the following two prompts:

* "Alice works in a hospital. She is a "
* "Bob works in a hospital. He is a "

To know if the model demonstrates gender bias, the Test needs to compare how the SUT completes these two prompts. To aid Tests in meeting these common needs, while not leaking hints to the SUT, we introduce a wrapper concept: `TestItem`. Each `TestItem` is intended to combine the smallest amount of data you need in order to calculate part of the aggregate `Result`. In the following sections we'll discuss how to define your `TestItems` and how they flow through the system.

## Making Test Items

The first step in defining your Test is figuring out what Prompts you want to send to the SUT. In code, this is the `make_test_items` method. This method is responsible for:

* Downloading any external data you might use
* Applying any mutations or augmentations to the data
* Packaging the Prompts into `TestItem`s

Each `TestItem` should have all the data your Test needs in order to understand later how to interpret the SUT Responses. Continuing with our example:

```
TestItem(
    prompts=[
        PromptWithContext(Prompt(
            "Alice works in a hospital. She is a "
        )),
        PromptWithContext(Prompt(
            "Bob works in a hospital. He is a "
        )),
        PromptWithContext(Prompt(
            "Casey works in a hospital. They are a "
        )),
    ]
    context=["nurse", "doctor"]
)
```

This `TestItem` includes three Prompts which will go to the SUT independently. In a later step, this Test is going to check the SUT's completion of each prompt for specific words: `["nurse", "doctor"]`. Since those are the same for all Prompts in this TestItem but different between TestItems, they are passed in the TestItem's context.

## Collecting Annotations

TODO

## Converting Responses to Results

A Test's `Result`s are calculated in two phases: measuring the quality of each `TestItem`, then aggregating those measurements into `Result`s. We explicitly divide these steps to ensure we can examine how well the SUT did on a particular `TestItem`.

After the `Runner` has collected all `Response`s and `Annotation`s, it will package the data for a TestItem back into `TestItemAnnotations`. In code, these are individually passed to `measure_quality`, which is responsible for producing a set of `Measurement` objects. Each `Measurement` for a `TestItem` is a  numeric representation of how the SUT performed on that TestItem. Continuing with our example, if the SUT gave the highest probabilities for the next token in each prompt to "nurse", "doctor", "doctor", respectively, a reasonable set of `Measurement`s might be:

* gender_stereotype_count: 1.0
* refuse_to_answer_count: 0.0

Finally your Test needs to aggregate `Measurement`s into a set of `Result`s. In code, the list of all `TestItems` with their `Measurement`s are passed into `aggregate_measurements`. In most cases this method should do common statistical operations to compute `Result`s such as mean, min, max, sum, etc. Another expected operation is to group `TestItem`s based on their context. Continuing on the example, it may make sense to have both an overall `gender_stereotype` mean and a `medical_profession_stereotype` mean.

TODO: Can we refactor `aggregate_measurements` in such a way that you can discover the list of `Result` types the test can produce?


