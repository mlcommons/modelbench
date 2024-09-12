import pytest
import torch
from transformers.generation.stopping_criteria import StoppingCriteriaList  # type: ignore
from transformers.utils import ModelOutput  # type: ignore

from fake_model import make_client, make_mocked_client  # type: ignore
from modelgauge.prompt import SUTOptions, ChatMessage, ChatPrompt, ChatRole, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.sut import SUTCompletion, SUTResponse, TokenProbability, TopTokens
from modelgauge.suts.huggingface_client import (
    HuggingFaceCompletion,
    HuggingFaceRequest,
    HuggingFaceResponse,
)

_DEFAULT_REQUEST_ARGS = {
    "model": "some-model",
    "generate_args": {
        "temperature": 1.0,
        "num_return_sequences": 1,
        "top_p": 1.0,
        "top_k": 1,
    },
}


_UNK_TOKEN_ID = 0
_VOCAB_MAP = {
    "<unk>": _UNK_TOKEN_ID,
    "one": 1,
    "two": 2,
    "three": 3,
}


def _make_request(
    text="some text prompt", max_new_tokens=100, stop_sequences=[], **kwargs
):
    return HuggingFaceRequest(
        prompt=text,
        max_new_tokens=max_new_tokens,
        stop_sequences=stop_sequences,
        **_DEFAULT_REQUEST_ARGS,
        **kwargs,
    )


def _make_generate_output(output_ids, num_input_ids, output_scores=None):
    """Used to patch return model.generate() return value."""

    dummy_input_ids = torch.zeros(
        (output_ids.size(0), num_input_ids), dtype=output_ids.dtype
    )
    # Prepend input-ids in output sequence
    output_data = {"sequences": torch.cat((dummy_input_ids, output_ids), dim=1)}
    if output_scores:
        output_data["scores"] = output_scores
    return ModelOutput(output_data)


def test_huggingface_translate_text_prompt_request():
    client = make_client()
    prompt = TextPrompt(text="some text prompt")
    request = client.translate_text_prompt(prompt)
    assert request == _make_request(text="some text prompt")
    assert request.num_top_logprobs is None


def test_huggingface_translate_chat_prompt_request():
    client = make_client()
    prompt = ChatPrompt(
        messages=[
            ChatMessage(text="some-text", role=ChatRole.user),
            ChatMessage(text="more-text", role=ChatRole.sut),
        ]
    )
    request = client.translate_chat_prompt(prompt)
    assert request.prompt == format_chat(prompt)


def test_huggingface_translate_request_non_default_options():
    client = make_client()
    options = SUTOptions(
        temperature=0.5,
        num_completions=2,
        top_p=0.4,
        max_tokens=15,
        top_k_per_token=3,
        stop_sequences=["stop"],
        top_logprobs=40,
    )
    request = client._translate_request("some text prompt", options)
    assert request.max_new_tokens == 15
    assert request.stop_sequences == ["stop"]
    assert request.generate_args.model_dump() == {
        "temperature": 0.5,
        "num_return_sequences": 2,
        "top_p": 0.4,
        "top_k": 3,
    }
    assert request.num_top_logprobs == 40

    # Test that temperature of 0 is adjusted.
    request = client._translate_request("some text prompt", SUTOptions(temperature=0.0))
    assert request.generate_args.temperature == 1e-7


def test_huggingface_generate_args():
    """Tests that the expected arguments are passed to the .generate() call in the evaluate method."""
    client = make_mocked_client(_VOCAB_MAP)
    request = _make_request(text="one two unknown", max_new_tokens=100)
    client.evaluate(request)

    _, generate_call_kwargs = client.model.generate.call_args
    # Need to check tensor equality separately
    input_ids = generate_call_kwargs.pop("input_ids")
    assert torch.is_tensor(input_ids)
    assert torch.equal(input_ids.to("cpu"), torch.tensor([[1, 2, _UNK_TOKEN_ID]]))
    assert generate_call_kwargs == {
        **_DEFAULT_REQUEST_ARGS["generate_args"],
        "max_new_tokens": 100,
        "do_sample": True,
        "return_dict_in_generate": True,
        "output_scores": False,
    }


@pytest.mark.parametrize(
    "num_top_logprobs,output_scores", [(50, True), (1, True), (0, False), (None, False)]
)
def test_huggingface_generate_args_logprobs(num_top_logprobs, output_scores):
    client = make_mocked_client(_VOCAB_MAP)
    request = _make_request(num_top_logprobs=num_top_logprobs)
    client.evaluate(request)

    _, generate_call_kwargs = client.model.generate.call_args
    assert generate_call_kwargs["output_scores"] == output_scores


def test_huggingface_generate_args_mask():
    """Tests that the attention mask is passed to the .generate() call if tokenizer outputs a mask."""
    client = make_mocked_client(_VOCAB_MAP, return_mask=True)
    request = _make_request(text="one two")
    client.evaluate(request)

    _, generate_call_kwargs = client.model.generate.call_args
    mask = generate_call_kwargs.get("attention_mask")
    assert torch.is_tensor(mask)
    assert torch.equal(mask.to("cpu"), torch.tensor([[1, 1]]))


def test_huggingface_generate_stopping_criteria_arg():
    client = make_mocked_client(_VOCAB_MAP)
    request = _make_request(stop_sequences=["two", "three word stop"])
    client.evaluate(request)

    _, generate_call_kwargs = client.model.generate.call_args
    stopping_criteria = generate_call_kwargs["stopping_criteria"]
    assert isinstance(stopping_criteria, StoppingCriteriaList)
    stopping_criteria_ids = [[2], [3, _UNK_TOKEN_ID, _UNK_TOKEN_ID]]
    assert [token.stop_sequence for token in stopping_criteria] == stopping_criteria_ids


def test_huggingface_generate_args_with_reduced_max_tokens():
    """If total num. tokens (max_tokens + prompt length) exceeds model's max length, the max_new_tokens is adjusted."""
    client = make_mocked_client(_VOCAB_MAP, model_max_length=52)
    request = _make_request(text="some prompt")
    client.evaluate(request)

    _, generate_call_kwargs = client.model.generate.call_args
    assert generate_call_kwargs["max_new_tokens"] == 50


@pytest.mark.parametrize("model_max_length", [2, 3, 0])
def test_huggingface_evaluate_prompt_too_large_exception(model_max_length):
    """Exception is raised if the num. of prompt tokens exceeds the the model's max length."""
    client = make_mocked_client(_VOCAB_MAP, model_max_length=model_max_length)
    request = _make_request(text="three token prompt")
    with pytest.raises(
        AssertionError,
        match=f"Prompt has 3 tokens, which is >= max length {model_max_length}",
    ):
        client.evaluate(request)


def test_huggingface_evaluate():
    client = make_mocked_client(_VOCAB_MAP)
    client.model.generate.return_value = _make_generate_output(
        torch.tensor([[3, 2, 1]]), num_input_ids=1
    )

    request = _make_request(text="input")
    response = client.evaluate(request)
    assert response == HuggingFaceResponse(
        completions=[HuggingFaceCompletion(text="three two one")]
    )


def test_huggingface_evaluate_multiple_completions():
    client = make_mocked_client(_VOCAB_MAP)
    client.model.generate.return_value = _make_generate_output(
        torch.tensor([[1, 2], [2, 1], [2, 2]]), num_input_ids=1
    )

    request = client._translate_request("input", SUTOptions(num_completions=3))
    response = client.evaluate(request)
    assert response == HuggingFaceResponse(
        completions=[
            HuggingFaceCompletion(text="one two"),
            HuggingFaceCompletion(text="two one"),
            HuggingFaceCompletion(text="two two"),
        ]
    )


def test_huggingface_evaluate_response_with_logprobs():
    num_logprobs = 3
    client = make_mocked_client(_VOCAB_MAP)
    output_token_ids = torch.tensor([[1, 2]])
    random_scores = (torch.rand(1, 4), torch.rand(1, 4))
    client.model.generate.return_value = _make_generate_output(
        output_token_ids, num_input_ids=1, output_scores=random_scores
    )
    request = client._translate_request("input", SUTOptions(top_logprobs=num_logprobs))
    response = client.evaluate(request)
    assert len(response.completions) == 1
    assert response.completions[0].text == "one two"
    logprob_dicts = response.completions[0].top_logprobs_dicts
    # One logprob dict for each token in the output sequence
    assert len(logprob_dicts) == 2
    for logprobs in logprob_dicts:
        assert isinstance(logprobs, dict)
        assert len(logprobs) == num_logprobs
        assert all(isinstance(token, str) for token in logprobs.keys())
        assert all(isinstance(logprob, float) for logprob in logprobs.values())


@pytest.mark.parametrize(
    "max_new_tokens,expected_text", [(2, "three two"), (1, "three")]
)
def test_huggingface_evaluate_truncate(max_new_tokens, expected_text):
    """Output sequence is truncated if model does not comply with max_new_tokens."""
    client = make_mocked_client(_VOCAB_MAP)
    client.model.generate.return_value = _make_generate_output(
        torch.tensor([[3, 2, 1]]), num_input_ids=1
    )

    request = _make_request(text="input", max_new_tokens=max_new_tokens)
    response = client.evaluate(request)
    assert response.completions == [HuggingFaceCompletion(text=expected_text)]


def test_huggingface_translate_response():
    client = make_client()
    request = _make_request(text="input")
    response = HuggingFaceResponse(completions=[HuggingFaceCompletion(text="output")])

    sut_response = client.translate_response(request, response)
    assert sut_response == SUTResponse(completions=[SUTCompletion(text="output")])


def test_huggingface_translate_response_logprobs():
    client = make_client()
    request = client._translate_request("input", SUTOptions(top_logprobs=3))
    response = HuggingFaceResponse(
        completions=[
            HuggingFaceCompletion(
                text="output",
                top_logprobs_dicts=[
                    {"token1": 0.1, "token2": 0.2, "token3": 0.3},
                    {"token2": -0.4, "token3": 0.5, "token4": 0.6},
                ],
            )
        ]
    )
    logprobs = [
        TopTokens(
            top_tokens=[
                TokenProbability(token="token1", logprob=0.1),
                TokenProbability(token="token2", logprob=0.2),
                TokenProbability(token="token3", logprob=0.3),
            ]
        ),
        TopTokens(
            top_tokens=[
                TokenProbability(token="token2", logprob=-0.4),
                TokenProbability(token="token3", logprob=0.5),
                TokenProbability(token="token4", logprob=0.6),
            ]
        ),
    ]

    sut_response = client.translate_response(request, response)
    assert sut_response == SUTResponse(
        completions=[SUTCompletion(text="output", top_logprobs=logprobs)]
    )
