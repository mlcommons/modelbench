# Tutorial: Adding A New SUT (System Under Test)

Adding a new SUT to ModelBench can be done in a number of ways, but here is an example of the easiest. In this example,
the assumption is that you want to create your own SUT -- a process that is described in the ModelGauge documentation --
and run a ModelBench benchmark against it. Here are the steps to do so.

## 1. Install ModelBench

First things first, install ModelBench and make sure it works. For installation instructions, look [here](https://github.com/mlcommons/modelbench?tab=readme-ov-file#modelbench)
and [here, specifically](https://github.com/mlcommons/modelbench?tab=readme-ov-file#trying-it-out) for trying it out
instructions.

This tutorial will assume you've installed ModelBench using PyPi

## 2. Create your SUT plugin

For the purposes of this tutorial, please refer to the **[Creating a basic SUT](https://github.com/mlcommons/modelgauge/blob/main/docs/tutorial_suts.md#creating-a-basic-sut)**
tutorial in the ModelGauge repository. Ignore any instructions about installing the SUT, we won't need to do that here.

At the end of this process, you should have created a python file containing your SUT (the name of the file is unimportant,
I will use `mysut.py` for this tutorial example) containing a line that will register the SUT with ModelGauge like the
following:

```python
SUTS.register(DemoYesNoSUT, "demo_yes_no")
```

If you don't want to follow the instructions on creating a basic SUT right now, don't worry about it! Use this:

```python
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import PromptResponseSUT, SUTCompletion, SUTResponse
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from pydantic import BaseModel

class DemoYesNoRequest(BaseModel):
    text: str

class DemoYesNoResponse(BaseModel):
    number_of_words: int
    text: str

@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt])
class DemoYesNoSUT(PromptResponseSUT[DemoYesNoRequest, DemoYesNoResponse]):
  def translate_text_prompt(self, prompt: TextPrompt) -> DemoYesNoRequest:
    return DemoYesNoRequest(text=prompt.text)

  def translate_chat_prompt(self, prompt: ChatPrompt) -> DemoYesNoRequest:
    return DemoYesNoRequest(text=format_chat(prompt))

  def evaluate(self, request: DemoYesNoRequest) -> DemoYesNoResponse:
    # Return Yes if the input is an even number of words
    number_of_words = len(request.text.split())
    answer = "Yes" if number_of_words % 2 == 0 else "No"
    return DemoYesNoResponse(number_of_words=number_of_words, text=answer)

  def translate_response(
    self, request: DemoYesNoRequest, response: DemoYesNoResponse
) -> SUTResponse:
    return SUTResponse(completions=[SUTCompletion(text=response.text)])

SUTS.register(DemoYesNoSUT, "demo_yes_no")
```

Put this file in a directory by itself. I'm going to use `~/modelbench_plugins` for this tutorial.

## 3. Run the benchmark using your new SUT

Run the benchmark against your new SUT by using the `modelbench benchmark` command, specifying the `--plugin-dir` option
and the specific SUT using `--sut` like this:

```shell
modelbench benchmark -m 10 --plugin-dir ~/modelbench_plugins/ --sut demo_yes_no
```
> [!NOTE]
> The `--sut` option can be used multiple times to declare multiple SUTs.

> [!WARNING]
> `--plugin-dir` will import any modules in the specified directory which can execute code that could be harmful, malicious,
> or that could have unexpected consequences. Use with caution and in a trusted environment.

## 4. View the report

After a few moments, you should see some output similar to the following:

```shell
Loading plugins: 100%|██████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 18.04it/s]

MLCommons AI Safety Benchmark v1.0 is a proof of concept only. Results are not intended to indicate actual levels of AI system safety.

Examining system "demo yes no"
  Starting run for benchmark "General Purpose Ai Chat Benchmark"
    Examining hazard "Safe Cae Hazard"
Processing TestItems for test=safe-cae sut=demo_yes_no: 100%|███████████████████████████| 1/1 [00:00<00:00, 1001.98it/s]
    Examining hazard "Safe Cbr Hazard"
Processing TestItems for test=safe-cbr sut=demo_yes_no: 100%|███████████████████████████| 1/1 [00:00<00:00, 1434.44it/s]
    Examining hazard "Safe Hat Hazard"
Processing TestItems for test=safe-hat sut=demo_yes_no: 100%|███████████████████████████| 1/1 [00:00<00:00, 1121.17it/s]
    Examining hazard "Safe Nvc Hazard"
Processing TestItems for test=safe-nvc sut=demo_yes_no: 100%|███████████████████████████| 1/1 [00:00<00:00, 1577.99it/s]
    Examining hazard "Safe Scr Hazard"
Processing TestItems for test=safe-scr sut=demo_yes_no: 100%|███████████████████████████| 1/1 [00:00<00:00, 1379.25it/s]
    Examining hazard "Safe Ssh Hazard"
Processing TestItems for test=safe-ssh sut=demo_yes_no: 100%|███████████████████████████| 1/1 [00:00<00:00, 1414.61it/s]
    Examining hazard "Safe Vcr Hazard"
Processing TestItems for test=safe-vcr sut=demo_yes_no: 100%|███████████████████████████| 1/1 [00:00<00:00, 1314.01it/s]

Benchmarking complete, rendering reports...
demo_yes_no

Reports complete, open web/index.html
```

When this process is complete, you can open `web/index.html` in a browser and view your report.