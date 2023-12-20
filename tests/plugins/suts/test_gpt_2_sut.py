from newhelm.placeholders import Prompt
from newhelm.plugins.suts.gpt_2_sut import GPT2
from newhelm.sut import Interaction, Turn


def test_evaluate():
    sut = GPT2()
    prompt = Prompt("One two three")
    interaction = sut.evaluate(prompt)
    assert interaction == Interaction(
        [Turn(prompt=prompt, completion="The prompt has 3 words.")]
    )
