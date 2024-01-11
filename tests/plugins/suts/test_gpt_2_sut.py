from newhelm.placeholders import Prompt
from newhelm.plugins.suts.gpt_2_sut import GPT2
from newhelm.sut import SUTResponse


def test_evaluate():
    sut = GPT2()
    prompt = Prompt("One two three")
    response = sut.evaluate(prompt)
    assert response == SUTResponse(completion="D")
