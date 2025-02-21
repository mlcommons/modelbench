from typing import Optional

import requests  # type: ignore
import tenacity
from huggingface_hub import ChatCompletionOutput  # type: ignore
from modelgauge.auth.huggingface_inference_token import HuggingFaceInferenceToken
from modelgauge.prompt import TextPrompt
from modelgauge.secret_values import InjectSecret
from modelgauge.sut import PromptResponseSUT, SUTResponse
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS
from pydantic import BaseModel
from tenacity import stop_after_attempt, wait_random_exponential


class HuggingFaceChatParams(BaseModel):
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None


class HuggingFaceChatRequest(BaseModel):
    inputs: str
    parameters: HuggingFaceChatParams


class HuggingFaceResponse(BaseModel):
    generated_text: str


@modelgauge_sut(capabilities=[AcceptsTextPrompt])
class HuggingFaceSUT(PromptResponseSUT[HuggingFaceChatRequest, ChatCompletionOutput]):
    """A Hugging Face SUT that is hosted on a dedicated inference endpoint."""

    def __init__(self, uid: str, api_url: str, token: HuggingFaceInferenceToken):
        super().__init__(uid)
        self.token = token.value
        self.api_url = api_url

    def translate_text_prompt(self, prompt: TextPrompt) -> HuggingFaceChatRequest:
        return HuggingFaceChatRequest(
            inputs=prompt.text,
            parameters=HuggingFaceChatParams(
                max_new_tokens=prompt.options.max_tokens, temperature=prompt.options.temperature
            ),
        )

    @tenacity.retry(stop=stop_after_attempt(7), wait=wait_random_exponential())
    def evaluate(self, request: HuggingFaceChatRequest) -> HuggingFaceResponse:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = request.model_dump(exclude_none=True)
        response = requests.post(self.api_url, headers=headers, json=payload)
        try:
            if response.status_code != 200:
                response.raise_for_status()
            response_json = response.json()[0]
            return HuggingFaceResponse(**response_json)
        except Exception as e:
            print(f"Unexpected failure for {payload}: {response}:\n {response.content}\n{response.headers}")
            raise e

    def translate_response(self, request: HuggingFaceChatRequest, response: HuggingFaceResponse) -> SUTResponse:
        return SUTResponse(text=response.generated_text)


HF_SECRET = InjectSecret(HuggingFaceInferenceToken)

SUTS.register(
    HuggingFaceSUT,
    "olmo-7b-0724-instruct-hf",
    "https://flakwttqzmq493dw.us-east-1.aws.endpoints.huggingface.cloud",
    HF_SECRET,
)

SUTS.register(
    HuggingFaceSUT,
    "olmo-2-1124-7b-instruct-hf",
    "https://l2m28ramsifovtf6.us-east-1.aws.endpoints.huggingface.cloud",
    HF_SECRET,
)
