from copy import deepcopy
from dataclasses import dataclass, field
import torch
from transformers import AutoModelForCausalLM  # type: ignore
from transformers.generation.stopping_criteria import (  # type: ignore
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore
from typing import Any, Dict, List, Optional, TypedDict
import os
from typing import Any, Dict, Optional

from newhelm.concurrency import ThreadSafeWrapper
from newhelm.placeholders import Prompt, SUTOptions
from newhelm.sut import PromptResponseSUT, SUTResponse
from newhelm.sut_registry import SUTS

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
    pretrained_model_name_or_path: str, **kwargs
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
                **kwargs,
            )
        )
    except OSError:
        return WrappedPreTrainedTokenizer(
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=False,
                use_fast=True,
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


class HuggingFaceRequest(TypedDict):
    """Data passed between make_request and serve_request. Used as the cache key."""

    prompt: str
    temperature: float
    num_return_sequences: int
    max_new_tokens: int
    top_p: float
    echo_prompt: bool
    top_k_per_token: int
    stop_sequences: List


class HuggingFaceCompletion(TypedDict):
    text: str
    tokens: List[str]
    logprobs: List[float]
    top_logprobs_dicts: List[Dict[str, float]]
    prompt_logprobs: List[float]
    prompt_top_logprobs_dicts: List[Dict[str, float]]


class HuggingFaceResponse(TypedDict):
    completions: List[HuggingFaceCompletion]
    input_length: int


@dataclass(frozen=True)
class Token:
    """
    A `Token` represents one token position in a `Sequence`, which has the
    chosen `text` as well as the top probabilities under the model.

    Note: (text, logprob) could exist or not exist in `top_logprobs`.
    """

    # Text that was chosen
    text: str

    # Log probability of generating that
    logprob: float

    # text -> log probability of generating that
    top_logprobs: Dict[str, float]


@dataclass(frozen=True)
class Sequence:
    """A `Sequence` is a sequence of tokens."""

    # The concatenation of all the tokens
    text: str

    # The sum of the log probabilities of all tokens
    logprob: float

    # The tokens
    tokens: List[Token]

    def __add__(self, other: "Sequence") -> "Sequence":
        return Sequence(
            self.text + other.text,
            self.logprob + other.logprob,
            self.tokens + other.tokens,
        )


@dataclass(frozen=False)
class RequestResult:
    """What comes back due to a `Request`."""

    success: bool
    """Whether the request was successful"""

    completions: List[Sequence]
    """List of completion"""

    error: Optional[str] = None
    """If `success` is false, what was the error?"""


def truncate_sequence(
    sequence: Sequence, options: SUTOptions, print_warning: bool = True
) -> Sequence:
    """
    Certain providers have bugs where they aren't respecting max_tokens,
    stop_sequences and the end of text token, so as a hack, we have to manually
    truncate the suffix of `sequence` and `tokens` as a post-hoc process.
    """
    # TODO: if echo_prompt, then we should only ignore the prompt, but we don't
    # know how many tokens the prompt takes up.
    # In the benchmark, usually echo_prompt is only used for language modeling,
    # where max_tokens = 0, so there's nothing to truncate.
    if options.echo_prompt:
        if options.max_tokens != 0:
            return sequence

    for stop in options.stop_sequences:
        # Find `stop` in the text
        try:
            new_text = sequence.text[: sequence.text.index(stop)]
        except ValueError:
            # The stop sequence doesn't exist, but it might exist in the list of tokens.
            new_text = sequence.text

        # Strip `stop` off the tokens
        new_tokens: List[Token] = []
        # Need to start
        for token in sequence.tokens:
            # Note: we can only strip at token boundaries
            if token.text.startswith(stop):
                break
            new_tokens.append(token)

        # Recompute log probability
        new_logprob = sum(token.logprob for token in new_tokens)
        sequence = Sequence(text=new_text, logprob=new_logprob, tokens=new_tokens)

    # Truncate based on the max number of tokens.
    if len(sequence.tokens) > options.max_tokens:
        new_tokens = sequence.tokens[: options.max_tokens]

        # This is imperfect stitching together of tokens, so just to make sure this is okay
        # TODO: should use the proper detokenizer since T5-style models.
        # Usually, in our benchmark, max_tokens is active when it's 1, so hopefully this isn't an issue.
        new_text = "".join(token.text for token in new_tokens)
        new_logprob = sum(token.logprob for token in new_tokens)

        sequence = Sequence(text=new_text, logprob=new_logprob, tokens=new_tokens)

    return sequence


TORCH_DTYPE_KEY = "torch_dtype"
TORCH_DTYPE_VALUE_PREFIX = "torch."


def _process_huggingface_client_kwargs(raw_kwargs: Dict[str, Any]):
    """Process the kwargs for HuggingFaceClient.

    The kwargs passed to HuggingFaceClient will eventually be passed to AutoModel.from_pretrained().
    Since the kwargs from HuggingFaceClient may be derived from configuration YAML,
    they may contain primitive types instead of the unserializable types that
    AutoModel.from_pretrained() expects (e.g. torch_dtype). This function converts values of
    primitive types to values of the unserializable types."""
    processed_kwargs = deepcopy(raw_kwargs)

    # Convert torch_dtype string value to actual dtypes
    # e.g. the string "torch.bfloat16" is converted to torch.bfloat16
    torch_dtype = processed_kwargs.get(TORCH_DTYPE_KEY)
    if torch_dtype and isinstance(torch_dtype, str):
        if not torch_dtype.startswith(TORCH_DTYPE_VALUE_PREFIX):
            raise ValueError(
                f'Unknown dtype "{torch_dtype}"; expected a string such as "torch.bfloat16"'
            )
        processed_kwargs[TORCH_DTYPE_KEY] = getattr(
            torch, torch_dtype[len(TORCH_DTYPE_VALUE_PREFIX) :]
        )

    return processed_kwargs


class HuggingFaceSUT(PromptResponseSUT[HuggingFaceRequest, HuggingFaceResponse]):
    """A thin wrapper around a Hugging Face AutoModelForCausalLM for HuggingFaceClient to call."""

    def __init__(self, pretrained_model_name_or_path: str, **kwargs):
        if torch.cuda.is_available():
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
        hg_kwargs = _process_huggingface_client_kwargs(kwargs)
        # WARNING this may fail if your GPU does not have enough memory
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **hg_kwargs
        ).to(self.device)
        self.wrapped_tokenizer: WrappedPreTrainedTokenizer = create_tokenizer(
            pretrained_model_name_or_path, **hg_kwargs
        )

    def evaluate(self, raw_request: HuggingFaceRequest) -> HuggingFaceResponse:
        with self.wrapped_tokenizer as tokenizer:
            encoded_input = tokenizer(
                raw_request["prompt"], return_tensors="pt", return_token_type_ids=False
            ).to(self.device)
        top_k_per_token: int = raw_request["top_k_per_token"]
        stopping_criteria: Optional[StoppingCriteriaList] = None
        optional_args = {}
        if len(raw_request["stop_sequences"]) > 0:
            with self.wrapped_tokenizer as tokenizer:
                stop_sequence_ids = tokenizer(
                    raw_request["stop_sequences"],
                    return_token_type_ids=False,
                    add_special_tokens=False,
                )
            if (
                len(stop_sequence_ids.input_ids) == 1
                and len(stop_sequence_ids.input_ids[0]) == 1
            ):
                optional_args["eos_token_id"] = stop_sequence_ids.input_ids[0][0]
            else:
                stopping_criteria = StoppingCriteriaList()
                for stop_sequence_input_ids in stop_sequence_ids.input_ids:
                    stopping_criteria.append(
                        StopAtSpecificTokenCriteria(
                            stop_sequence=stop_sequence_input_ids
                        )
                    )

        # Check if we need to compute the perplexity of the prompt (#1497)
        compute_logprobs_only = (
            raw_request["max_new_tokens"] == 0
            and raw_request["num_return_sequences"] == 1
            and raw_request["echo_prompt"]
        )

        # Use HuggingFace's `generate` method.
        if compute_logprobs_only:
            with torch.no_grad():
                output = self.model(encoded_input["input_ids"])
            sequences = encoded_input["input_ids"]
            scores = output.logits
        else:
            output = self.model.generate(
                **encoded_input,
                temperature=raw_request["temperature"],
                num_return_sequences=raw_request["num_return_sequences"],
                max_new_tokens=raw_request["max_new_tokens"],
                top_p=raw_request["top_p"],
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                **optional_args,
                stopping_criteria=stopping_criteria,
            )
            sequences = output.sequences
            scores = output.scores

        prompt_tokens_logprobs = []
        prompt_tokens_top_logprobs_dicts: List[Dict] = []
        if compute_logprobs_only:
            # Append the logprob of the first token of the prompt.
            prompt_tokens_logprobs.append(0.0)
            prompt_tokens_top_logprobs_dicts.append({})

            # Compute logprobs of prompt tokens.
            for completion_id in range(raw_request["num_return_sequences"]):
                for i in range(len(sequences[completion_id]) - 1):
                    logprobs = torch.nn.functional.log_softmax(
                        scores[completion_id][i], dim=0
                    )
                    topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                    with self.wrapped_tokenizer as tokenizer:
                        prompt_tokens_top_logprobs_dicts.append(
                            {
                                tokenizer.convert_ids_to_tokens(k.item()): v.item()
                                for (k, v) in zip(
                                    topk_logprobs.indices, topk_logprobs.values
                                )
                            }
                        )
                    prompt_tokens_logprobs.append(
                        logprobs[sequences[completion_id][i + 1]].item()
                    )

        # Compute logprobs of generated tokens for each completed sequence.
        all_generated_tokens_logprobs = []
        all_generated_tokens_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            generated_tokens_logprobs = []
            generated_tokens_top_logprobs_dicts = []
            for i in range(
                len(sequences[completion_id]) - len(encoded_input.input_ids[0])
            ):
                logprobs = torch.nn.functional.log_softmax(
                    scores[i][completion_id], dim=0
                )
                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                with self.wrapped_tokenizer as tokenizer:
                    generated_tokens_top_logprobs_dicts.append(
                        {
                            tokenizer.convert_ids_to_tokens(k.item()): v.item()
                            for (k, v) in zip(
                                topk_logprobs.indices, topk_logprobs.values
                            )
                        }
                    )
                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                generated_tokens_logprobs.append(
                    logprobs[sequences[completion_id][j]].item()
                )
            all_generated_tokens_logprobs.append(generated_tokens_logprobs)
            all_generated_tokens_top_logprobs_dicts.append(
                generated_tokens_top_logprobs_dicts
            )

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [
                sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences
            ]

        with self.wrapped_tokenizer as tokenizer:
            all_tokens = [
                [tokenizer.decode(token) for token in sequence_tokens]
                for sequence_tokens in sequences
            ]
            all_decoded_text = tokenizer.batch_decode(sequences)

        completions: List[HuggingFaceCompletion] = []
        for (
            decoded_text,
            tokens,
            generated_tokens_logprobs,
            generated_tokens_top_logprobs_dicts,
        ) in zip(
            all_decoded_text,
            all_tokens,
            all_generated_tokens_logprobs,
            all_generated_tokens_top_logprobs_dicts,
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": generated_tokens_logprobs,
                    "top_logprobs_dicts": generated_tokens_top_logprobs_dicts,
                    "prompt_logprobs": prompt_tokens_logprobs,
                    "prompt_top_logprobs_dicts": prompt_tokens_top_logprobs_dicts,
                }
            )

        return {
            "completions": completions,
            "input_length": len(encoded_input.input_ids[0]),
        }

    def translate_request(self, prompt: Prompt) -> HuggingFaceRequest:
        options = prompt.options
        return {
            "prompt": prompt.text,
            "temperature": 1e-7 if options.temperature == 0 else options.temperature,
            "num_return_sequences": options.num_completions,
            "max_new_tokens": options.max_tokens,
            "top_p": options.top_p,
            "echo_prompt": options.echo_prompt,
            "top_k_per_token": options.top_k_per_token,
            "stop_sequences": options.stop_sequences,
        }

    def translate_response(
        self, prompt: Prompt, response: HuggingFaceResponse
    ) -> SUTResponse:
        options = prompt.options
        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if options.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                if raw_completion.get("prompt_logprobs") and raw_completion.get(
                    "prompt_top_logprobs_dicts"
                ):
                    for token_text, logprob, top_logprobs_dict in zip(
                        raw_completion["tokens"][: response["input_length"]],
                        raw_completion["prompt_logprobs"][: response["input_length"]],
                        raw_completion["prompt_top_logprobs_dicts"][
                            : response["input_length"]
                        ],
                    ):
                        tokens.append(
                            Token(
                                text=token_text,
                                logprob=logprob,
                                top_logprobs=top_logprobs_dict,
                            )
                        )
                        sequence_logprob += logprob
                else:
                    for token_text in raw_completion["tokens"][
                        : response["input_length"]
                    ]:
                        tokens.append(
                            Token(text=token_text, logprob=0.0, top_logprobs={})
                        )

            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens,
                raw_completion["logprobs"],
                raw_completion["top_logprobs_dicts"],
            ):
                tokens.append(
                    Token(
                        text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict
                    )
                )
                sequence_logprob += logprob

            completion = Sequence(
                text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens
            )
            completion = truncate_sequence(completion, options)
            completions.append(completion)

        return SUTResponse(completions[0].text)


SUTS.register("gpt2", HuggingFaceSUT, "gpt2")
