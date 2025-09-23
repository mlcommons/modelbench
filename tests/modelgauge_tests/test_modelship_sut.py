from unittest.mock import patch

from modelgauge.prompt import ChatPrompt, ChatMessage, ChatRole
from modelgauge.sut import SUTOptions
from modelgauge.sut_definition import SUTDefinition
from modelgauge.suts.modelship_sut import ModelShipSUTFactory


def test_basic_request_with_vllm_options():
    # an actual commmand we need to run and the matching UID
    # vllm serve "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5" --trust-remote-code=Y --tensor-parallel-size=4
    uid = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5:modelship;vllm-tensor-parallel-size=4;vllm-trust-remote-code=Y"

    definition = SUTDefinition.parse(uid)
    factory = ModelShipSUTFactory(raw_secrets={"modelship": {"api_key": "whatever"}})
    sut = factory.make_sut(definition)
    prompt = ChatPrompt(messages=[ChatMessage(text="Why not?", role=ChatRole.user)])
    request = sut.translate_chat_prompt(prompt, SUTOptions())

    with patch("openai.resources.chat.completions.Completions.create") as fake_create:
        sut.evaluate(request)
        assert fake_create.call_count == 1
        kwargs = fake_create.call_args.kwargs
        assert kwargs["model"] == "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
        assert kwargs["vllm_options"]["tensor_parallel_size"] == "4"
        assert kwargs["vllm_options"]["trust_remote_code"] == "Y"
