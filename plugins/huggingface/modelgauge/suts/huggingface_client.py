import logging
import os

import torch
from pydantic import BaseModel
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.generation.stopping_criteria import (  # type: ignore
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Any, Dict, List, Optional, Tuple

from modelgauge.concurrency import ThreadSafeWrapper
from modelgauge.general import value_or_default
from modelgauge.prompt import ChatPrompt, TextPrompt
from modelgauge.prompt_formatting import format_chat
from modelgauge.secret_values import InjectSecret, OptionalSecret, SecretDescription
from modelgauge.sut import (
    PromptResponseSUT,
    SUTCompletion,
    SUTResponse,
    TokenProbability,
    TopTokens,
)
from modelgauge.sut_capabilities import (
    AcceptsChatPrompt,
    AcceptsTextPrompt,
    ProducesPerTokenLogProbabilities,
)
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS


WrappedPreTrainedTokenizer = ThreadSafeWrapper[PreTrainedTokenizerBase]
"""Thread safe wrapper around Hugging Face PreTrainedTokenizerBase.

Hugging Face PreTrainedTokenizerBase is thread-hostile and using it from multiple threads
simultaneously can result in an "Already borrowed" error (#1421). This wrapper ensures
that a lock is held when using the PreTrainedTokenizerBase.

Example usage:

    with wrapped_tokenizer as tokenizer:
        tokenizer.encode("...")
"""


def create_tokenizer(
    pretrained_model_name_or_path: str, hugging_face_token: Optional[str], **kwargs
) -> WrappedPreTrainedTokenizer:
    """Loads tokenizer using files from disk if they exist. Otherwise, downloads from HuggingFace."""
    # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
    # TODO: Figure out if we actually need this.
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    try:
        # From the Hugging Face documentation, "local_files_only(defaults to False) —
        # Whether or not to only look at local files".
        # Running `local_files_only=False` requires an internet connection even if the files are downloaded
        # and cached. We need to first run with `local_files_only=True` just in case the machine
        # we are running this code has connection issues. If the tokenizer files are not cached,
        # we attempt to download them from HuggingFace.
        # From https://huggingface.co/course/chapter6/3, "slow tokenizers are those written in Python inside
        # the Hugging Face Transformers library, while the fast versions are the ones provided by Hugging Face
        # Tokenizers, which are written in Rust." So, use the "fast" version of the tokenizers if available.
        return WrappedPreTrainedTokenizer(
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=True,
                use_fast=True,
                token=hugging_face_token,
                **kwargs,
            )
        )
    except OSError:
        return WrappedPreTrainedTokenizer(
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=False,
                use_fast=True,
                token=hugging_face_token,
                **kwargs,
            )
        )


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence: List[int]):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(
            self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype
        )

        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())


class HuggingFaceRequest(BaseModel):
    class GenerateArgs(BaseModel):
        temperature: float
        num_return_sequences: int
        top_p: float
        top_k: int

    prompt: str
    model: str  # Included to help uniquely identify the request (e.g in caching).
    max_new_tokens: int
    stop_sequences: List[str]
    generate_args: GenerateArgs
    num_top_logprobs: Optional[int] = None


class HuggingFaceCompletion(BaseModel):
    text: str
    top_logprobs_dicts: Optional[List[Dict[str, float]]] = None


class HuggingFaceResponse(BaseModel):
    completions: List[HuggingFaceCompletion]


class HuggingFaceToken(OptionalSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(
            scope="hugging_face",
            key="token",
            instructions="You can create tokens at https://huggingface.co/settings/tokens.",
        )


@modelgauge_sut(
    capabilities=[
        AcceptsTextPrompt,
        AcceptsChatPrompt,
        ProducesPerTokenLogProbabilities,
    ]
)
class HuggingFaceSUT(PromptResponseSUT[HuggingFaceRequest, HuggingFaceResponse]):
    def __init__(
        self,
        uid: str,
        pretrained_model_name_or_path: str,
        token: HuggingFaceToken,
        **kwargs,
    ):
        super().__init__(uid)
        if torch.cuda.is_available():
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
        self.token = token.value
        self.hg_kwargs = kwargs
        self.model_path = pretrained_model_name_or_path
        # Model and tokenizer are lazy loaded.
        self.model: Optional[Any] = None
        self.wrapped_tokenizer: Optional[WrappedPreTrainedTokenizer] = None

    def _load_model(self) -> Tuple[Any, WrappedPreTrainedTokenizer]:
        # WARNING this may fail if your GPU does not have enough memory
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            token=self.token,
            **self.hg_kwargs,
        ).to(self.device)
        wrapped_tokenizer = create_tokenizer(
            self.model_path, self.token, **self.hg_kwargs
        )
        return model, wrapped_tokenizer

    def evaluate(self, raw_request: HuggingFaceRequest) -> HuggingFaceResponse:
        assert self.model_path == raw_request.model
        if not self.model or not self.wrapped_tokenizer:
            self.model, self.wrapped_tokenizer = self._load_model()
        with self.wrapped_tokenizer as tokenizer:
            encoded_input = tokenizer(
                raw_request.prompt, return_tensors="pt", return_token_type_ids=False
            ).to(self.device)
            num_input_tokens = encoded_input.input_ids.nelement()
            # Ensure the total tokens is within the model's max length.
            max_new_tokens = min(
                raw_request.max_new_tokens,
                tokenizer.model_max_length - num_input_tokens,
            )
            if raw_request.max_new_tokens != max_new_tokens:
                logging.warning(
                    f"Had to reduce max_new_tokens from "
                    f"{raw_request.max_new_tokens} to {max_new_tokens}"
                )
            assert max_new_tokens > 0, (
                f"Prompt has {num_input_tokens} tokens, which is >= "
                f"max length {tokenizer.model_max_length}"
            )
        optional_args = {}
        # Convert stop sequences to token ids
        if len(raw_request.stop_sequences) > 0:
            with self.wrapped_tokenizer as tokenizer:
                stop_sequence_ids = tokenizer(
                    raw_request.stop_sequences,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                )
                stopping_criteria = StoppingCriteriaList()
                for stop_sequence_input_ids in stop_sequence_ids.input_ids:
                    stopping_criteria.append(
                        StopAtSpecificTokenCriteria(
                            stop_sequence=stop_sequence_input_ids
                        )
                    )
                optional_args["stopping_criteria"] = stopping_criteria
        # Some models do not have a `pad_token_id`. For example gpt2
        # This prevents a warning message
        if self.model.config.pad_token_id is None:
            if self.model.config.eos_token_id is not None:
                optional_args["pad_token_id"] = self.model.config.eos_token_id
        output_scores = (
            raw_request.num_top_logprobs is not None
            and raw_request.num_top_logprobs > 0
        )
        output = self.model.generate(
            **encoded_input,
            **raw_request.generate_args.model_dump(),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=output_scores,
            **optional_args,
        )

        completions: List[HuggingFaceCompletion] = []
        for completion_id in range(raw_request.generate_args.num_return_sequences):
            # Remove input prompt and truncate if length exceeds max_new_tokens.
            sequence = output.sequences[completion_id][num_input_tokens:]
            sequence = sequence[:max_new_tokens]

            text = tokenizer.decode(sequence)
            sequence_logprobs_dicts = None
            if raw_request.num_top_logprobs:
                scores = output.scores
                sequence_logprobs_dicts = []
                # Get dictonary of top-k logprobs for each token in the sequence.
                for i in range(len(sequence)):
                    logprobs = torch.nn.functional.log_softmax(
                        scores[i][completion_id], dim=0
                    )
                    # Get top tokens in terms of log probability.
                    topk_logprobs = torch.topk(logprobs, k=raw_request.num_top_logprobs)
                    logprobs_dict = {}
                    with self.wrapped_tokenizer as tokenizer:
                        for token_id, logprob in zip(
                            topk_logprobs.indices, topk_logprobs.values
                        ):
                            token = tokenizer.convert_ids_to_tokens(token_id.item())
                            logprobs_dict[token] = logprob.item()
                    sequence_logprobs_dicts.append(logprobs_dict)
            completions.append(
                HuggingFaceCompletion(
                    text=text, top_logprobs_dicts=sequence_logprobs_dicts
                )
            )
        return HuggingFaceResponse(completions=completions)

    def translate_text_prompt(self, prompt: TextPrompt) -> HuggingFaceRequest:
        return self._translate_request(prompt.text, prompt.options)

    def translate_chat_prompt(self, prompt: ChatPrompt) -> HuggingFaceRequest:
        return self._translate_request(format_chat(prompt), prompt.options)

    def _translate_request(self, text, options):
        temperature = value_or_default(options.temperature, 1.0)
        if temperature == 0:
            temperature = 1e-7
        generate_args = {
            "temperature": temperature,
            "num_return_sequences": options.num_completions,
            "top_p": value_or_default(options.top_p, 1.0),
            "top_k": value_or_default(options.top_k_per_token, 1),
        }
        return HuggingFaceRequest(
            prompt=text,
            model=self.model_path,
            max_new_tokens=options.max_tokens,
            stop_sequences=value_or_default(options.stop_sequences, []),
            generate_args=generate_args,
            num_top_logprobs=options.top_logprobs,
        )

    def translate_response(
        self, request: HuggingFaceRequest, response: HuggingFaceResponse
    ) -> SUTResponse:
        completions = []
        for raw_completion in response.completions:
            logprobs: Optional[List[TopTokens]] = None
            if request.num_top_logprobs:
                assert (
                    raw_completion.top_logprobs_dicts is not None
                ), "Expected logprobs, but not returned."
                logprobs = []
                for top_logprobs in raw_completion.top_logprobs_dicts:
                    top_tokens: List[TokenProbability] = []
                    for token, logprob in top_logprobs.items():
                        top_tokens.append(
                            TokenProbability(token=token, logprob=logprob)
                        )
                    logprobs.append(TopTokens(top_tokens=top_tokens))

            completions.append(
                SUTCompletion(text=raw_completion.text, top_logprobs=logprobs)
            )
        return SUTResponse(completions=completions)


SUTS.register(HuggingFaceSUT, "gpt2", "gpt2", InjectSecret(HuggingFaceToken))
