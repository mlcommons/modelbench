from typing import List
from newhelm.huggingface_client import HuggingFaceClient, Request, RequestResult
from newhelm.placeholders import Prompt
from newhelm.sut import SUTResponse, PromptResponseSUT
from newhelm.sut_registry import SUTS


class GPT2(PromptResponseSUT):
    """The SUT should have all the details currently spread across model_deployment and model_metadata."""

    def __init__(self):
        self.model = HuggingFaceClient("gpt2")

    def evaluate(self, prompt: Prompt) -> SUTResponse:
        response: RequestResult = self.model.make_request(
            Request(
                prompt=prompt.text,
                # Configured to match BBQ right now.
                temperature=0.0,
                num_completions=1,
                top_k_per_token=5,
                max_tokens=1,
                stop_sequences=[],
                echo_prompt=False,
                top_p=1,
            )
        )
        return SUTResponse(response.completions[0].text)


SUTS.register("gpt2", GPT2())
