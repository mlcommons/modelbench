from newhelm.placeholders import Prompt, PromptTemplate, WindowServiceConfig
from newhelm.plugins.suts.gpt_2_sut import GPT2
from newhelm.sut import Interaction, Turn


def test_specialize():
    sut = GPT2()
    prompt = sut.specialize(
        PromptTemplate(
            instructions_block="Instructions",
            train_instance_blocks=["The sky is: blue", "The grass is: green"],
            eval_instance_block="A polar bear is: ",
        )
    )
    assert prompt == Prompt(
        """\
Instructions
The sky is: blue
The grass is: green
A polar bear is: \
"""
    )


def test_specialize_truncates_training():
    sut = GPT2()
    # Monkey patch to make the window smaller.
    sut.window_service.config = WindowServiceConfig(max_sequence_length=50)
    prompt = sut.specialize(
        PromptTemplate(
            instructions_block="Instructions",
            train_instance_blocks=["The sky is: blue", "The grass is: green"],
            eval_instance_block="A polar bear is: ",
        )
    )
    # Remove the second training example
    assert prompt == Prompt(
        truncated=True,
        text="""\
Instructions
The sky is: blue
A polar bear is: \
""",
    )


def test_specialize_truncates_everything():
    sut = GPT2()
    # Monkey patch to make the window smaller.
    sut.window_service.config = WindowServiceConfig(max_sequence_length=25)
    prompt = sut.specialize(
        PromptTemplate(
            instructions_block="Instructions",
            train_instance_blocks=["The sky is: blue", "The grass is: green"],
            eval_instance_block="A polar bear is: ",
        )
    )
    # Remove both training examples, and part of the eval example.
    assert prompt == Prompt(
        truncated=True,
        text="""\
Instructions
A polar bear\
""",
    )


def test_evaluate():
    sut = GPT2()
    prompt = Prompt("One two three")
    interaction = sut.evaluate(prompt)
    assert interaction == Interaction(
        [Turn(prompt=prompt, completion="The prompt has 3 words.")]
    )
