# ModelBench journal documentation

The goal of the ModelBench run journal is to provide a clear record of what happened during the
scoring of SUTs against benchmarks. It is a JSONL file consisting of one line per event during
the run.

The run journal documents one specific run of a set of Benchmarks against a set of SUTs.
It is mainly focused on the individual Item, which is one Prompt from one Test run against one SUT.
The Item flows through a series of pipeline stages, each of which takes some action on the Item.

Currently those stages are

- TestRunItemSource - from the Test, get each prompt
- TestRunSutAssigner - for each prompt, create an Item for each SUT
- TestRunSutWorker - for each Item, send the prompt to the SUT and get a response
- TestRunAnnotationWorker - for each Item, send the SUT response to all Annotators for the Test
- TestRunResultsCollector - collect the Item results for further processing

After that, the collected results have their tests, hazards, and benchmarks scored,
and the final results are output.

## message and field definitions

### Common fields

Found in all entries.

- timestamp - ISO 8601 timestamp
- message - identifies the type of message
- class - the python class generating the message
- method - the method on that class generating the message

### Item fields

Found in all item entries.

- test - the uid of the Test for the Item
- prompt_id - the uid of the Prompt for the Item
- sut - the uid of the SUT the will receive the Prompt

### Message meanings:

In order of occurrence:

- starting journal - marks the beginning of the journal
- starting run - gives basic information on the run
- hazard info - gives information on each Hazard in the run
- test info - gives information on each Test in the run
- running pipeline - marks the start of the item pipeline
- using test items - for each Test, the counts of Items available and actually used
- queuing item - the beginning of an Item's flow in the pipeline
- fetched sut response - a SUT's live raw response to a single prompt
- using cached sut response - a SUT's cached raw response to a single prompt
- translated sut response - the result of translating a raw response into a common format
- fetched annotator response - an Annotator's live raw response to a single prompt/response pair
- using cached annotator response - an Annotator's cached raw response to a single prompt/response pair
- translated annotation - the result of translating a raw response into a common format
- measured item quality - after all annotations are completed, they combined to give measurements
- finished pipeline - all Items have been run through all stages of the pipeline
- test scored - the combined Item results are used to score the Tests
- hazard scored - the Test scores are used to score the Hazards
- benchmark scored - the Hazard scores are used to score the Benchmarks
- finished run - the run is complete
- cache info - basic information on the use of the caches for SUTs and Annotators

### Message-specific fields

Listed in run order. If a message isn't listed here, its fields are covered above under "Common fields".

- starting run
    - run_id - a unique id for the run
    - benchmarks - list of uids for Benchmarks in the run
    - tests - list of uids for Tests in the run
    - suts - list of uids for the SUTs in the run
    - max_items - maximum number of Items per test to be run
    - thread_count - maximum number of threads per stage of pipeline
- hazard info
    - hazard - uid of the Hazard
    - benchmark - uid of the Benchmark the Hazard is part of
    - tests - list of uids of the Tests in the Hazard
- test info
    - test - uid of Test
    - initialization - initialization record for Test
    - sut_options - any options passed to the SUT with a prompt
    - dependencies - the external dependencies of the Test, like the source file for the prompts
- using test items
    - test - the uid of the Test
    - using - how many prompts will be used in this run
    - total - how many prompts were available total
- queuing item
    - prompt_text - the text of the Prompt
- fetched sut response
    - run_time - seconds taken to get the response
    - request - raw request sent to the SUT
    - response - raw response as received from the SUT
- using cached sut response
    - response - the raw response as previously received from the SUT
- translated sut response
    - response_text - the text of the SUT response
- fetched annotator response
    - annotator - the uid of the Annotator
    - run_time - seconds taken to get the response
    - response - raw response as received from the Annotator
- using cached annotator response
    - annotator - the uid of the Annotator
    - response - raw response as previously received from the Annotator
- translated annotation
    - annotator - the uid of the Annotator
    - annotation - the annotation as translated into a common format; for MLC Benchmarks includes `is_safe`
- measured item quality
    - measurements - the Test's measurements of an Item based on annotations
    - run_time - total time for collecting all Annotations for a test and calculating measurements
- finished pipeline
    - time - total time to run the pipeline
    - total_finished - total Items that finished the pipeline
    - finished_counts - breakdown of finished items by SUT and Test uid
- test scored
    - sut - the uid of the SUT scored
    - test - the uid of the Test scored
    - items_finished - items finished for the Test and SUT
    - result - a Test's view of the results
- hazard scored
    - benchmark - the uid of the Benchmark being scored
    - sut - the uid of the SUT scored
    - hazard - the uid of the Hazard being scored
    - score - the Hazard's raw score, the fraction of prompts that were safe
    - reference - the grading reference score for the Hazard
    - samples - the number of Items used for the scores
    - numeric_grade - the numeric version of the grade (1 to 5)
    - text_grade - the text version of the grade
- benchmark scored
    - benchmark - the uid of the Benchmark scored
    - sut - the uid of the SUT scored
    - numeric_grade - the numeric version of the grade (1 to 5)
    - text_grade - the text version of the grade
- cache info
    - type - which kind of cache
    - cache - the Python class of the cache plus key parameters
    - start_count - how many items were in the cache before the run
    - end_count - how many items were in the cache after the run
