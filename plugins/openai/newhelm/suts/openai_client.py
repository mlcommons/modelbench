from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional, Union
from newhelm.credentials import RequiresCredentials, optionally_load_credentials
from newhelm.general import asdict_without_nones, get_or_create_json_file
from newhelm.placeholders import Prompt
from newhelm.sut import PromptResponseSUT, SUTResponse
from openai import OpenAI
from openai.types.chat import ChatCompletion

from newhelm.sut_registry import SUTS

_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"
_TOOL_ROLE = "tool_call_id"


@dataclass(frozen=True)
class OpenAIChatMessage:
    content: str
    role: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


@dataclass(frozen=True)
class OpenAIChatRequest:
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[OpenAIChatMessage]
    model: str
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    # How many chat completion choices to generate for each input message.
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[int] = None
    tools: Optional[List] = None
    tool_choice: Optional[Union[str, Dict]] = None
    user: Optional[str] = None


class OpenAIChat(
    PromptResponseSUT[OpenAIChatRequest, ChatCompletion], RequiresCredentials
):
    """
    Documented at https://platform.openai.com/docs/api-reference/chat/create
    """

    def __init__(self, model: str):
        self.model = model
        self.client: Optional[OpenAI] = None

    def get_credential_instructions(self) -> Dict[str, str]:
        return {
            "openai_api_key": "See https://platform.openai.com/api-keys",
            "openai_org_id": "Optional, see https://platform.openai.com/account/organization",
        }

    def load_credentials(self, secrets_dict: Dict[str, str]) -> None:
        self.client = OpenAI(
            api_key=secrets_dict["openai_api_key"],
            organization=secrets_dict.get("openai_org_id"),
        )

    def translate_request(self, prompt: Prompt) -> OpenAIChatRequest:
        # TODO #56 - Allow Tests to specify the full message set.
        message = OpenAIChatMessage(prompt.text, role=_USER_ROLE)
        return OpenAIChatRequest(messages=[message], model=self.model)

    def evaluate(self, request: OpenAIChatRequest) -> ChatCompletion:
        assert self.client, "Must call load_credentials before evaluate."
        request_dict = asdict_without_nones(request)
        return self.client.chat.completions.create(**request_dict)

    def translate_response(
        self, prompt: Prompt, response: ChatCompletion
    ) -> SUTResponse:
        text = response.choices[0].message.content
        assert text is not None
        return SUTResponse(text)


SUTS.register("gpt-3.5-turbo", OpenAIChat, "gpt-3.5-turbo")

if __name__ == "__main__":
    import sys

    prompt_text = " ".join(sys.argv[1:])
    client = OpenAIChat("gpt-3.5-turbo")
    prompt = Prompt(prompt_text)
    request = client.translate_request(prompt)
    print(request, "\n")
    secrets_dict = get_or_create_json_file("secrets", "default.json")
    optionally_load_credentials(client, secrets_dict)
    response = client.evaluate(request)
    print(response, "\n")
    result = client.translate_response(prompt, response)
    print(result, "\n")
